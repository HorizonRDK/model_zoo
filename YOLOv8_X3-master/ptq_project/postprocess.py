## Postprocess for YOLOv8: feature decoding, nms, visulization and saving, evaluation
import numpy as np
import torch
import cv2
from coco_metric import MSCOCODetMetric
from multiprocessing.pool import ApplyResult
import logging
from easydict import EasyDict
import re
import time
import colorsys
from horizon_tc_ui.utils import tool_utils


def calc_accuracy(annotation_path, accuracy):
    metric = MSCOCODetMetric(annotation_path, with_mask=False)
    if isinstance(accuracy[0], ApplyResult):
        batch_result = [i.get() for i in accuracy]
    else:
        batch_result = list(accuracy)
    total_samples = 0
    with open("yolov8_eval.log", "w") as eval_log_handle:
        logging.info("start to calc eval info...")
        for pred_result, filename in batch_result:
            metric.update(pred_result, filename)
            pred_result.sort(key=lambda x: x['bbox'][4], reverse=True)
            eval_log_handle.write(f"input_image_name: {filename} ")
            for one_result in pred_result:
                det_item = one_result['bbox']
                eval_log_handle.write(
                    f"{det_item[0]:6f},{det_item[1]:6f},{det_item[2]:6f},{det_item[3]:6f},{det_item[4]:6f},{det_item[5]:0.0f} "
                )
            eval_log_handle.write("\n")
    return metric.get()


def gen_report(metric_result):
    names, values = metric_result
    summary = values[0]
    summary = summary.splitlines()
    pattern = re.compile(r'(IoU.*?) .* (.*)$')
    tool_utils.report_flag_start('MAPPER-EVAL')
    for v in summary[0:2]:
        valid_data = pattern.findall(v)[0]
        logging.info("[%s] = %s" % (valid_data[0], valid_data[1]))
    tool_utils.report_flag_end('MAPPER-EVAL')


def prepare_output(res):
    # Dimension squeeze
    res_new = [res_ori.squeeze() for res_ori in res]
    return res_new


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    y = e_x / e_x.sum(axis=axis, keepdims=True)
    return y


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        remains = []
        for index in sorted_indices[1:]:
            remains.append(boxes[index])
        if len(remains) == 0:
            return keep_boxes
        ious = compute_iou(boxes[box_id], np.array(remains))
        
        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # Update sorted scores
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes


def process_output(output, original_img_shape, conf_threshold, iou_threshold, reg_max=16):
    output = prepare_output(output)

    dfl = np.arange(0, reg_max, dtype=np.float32)
    
    num_classes = 80
    img_height, img_width = original_img_shape[0], original_img_shape[1]
    model_height, model_width = 640, 640
    ratio_h, ratio_w = img_height / model_height, img_width / model_width 
    dh, dw = (0, 0)

    boxes = []
    confidences = []
    classes = []
    
    for i in range(len(output) // 2):
        bboxes_feat = output[i * 2 + 1]
        scores_feat = sigmoid(output[i * 2 + 0])
        scores_max = scores_feat.max(-1)
        scores_max_indices = scores_feat.argmax(-1)
        indices = np.where(scores_max > conf_threshold)
        hIdx, wIdx = indices
        num_proposal = hIdx.size
        if not num_proposal:
            continue
        assert scores_feat.shape[-1] == num_classes
        scores = scores_max[hIdx, wIdx]
        bboxes = bboxes_feat[hIdx, wIdx].reshape(-1, 4, reg_max)
        bboxes = softmax(bboxes, -1) @ dfl
        class_ids = scores_max_indices[hIdx, wIdx]
        
        for k in range(num_proposal):
            x0, y0, x1, y1 = bboxes[k]
            score = scores[k]
            class_id = class_ids[k]
            h, w, stride = hIdx[k], wIdx[k], 1 << (i + 3)
            x0 = ((w + 0.5 - x0) * stride - dw) * ratio_w
            y0 = ((h + 0.5 - y0) * stride - dh) * ratio_h
            x1 = ((w + 0.5 + x1) * stride - dw) * ratio_w
            y1 = ((h + 0.5 + y1) * stride - dh) * ratio_h
            # Clip
            x0 = min(max(x0, 0), img_width)
            y0 = min(max(y0, 0), img_height)
            x1 = min(max(x1, 0), img_width)
            y1 = min(max(y1, 0), img_height)
            confidences.append(float(score))
            boxes.append(np.array([x0, y0, x1, y1]))
            classes.append(int(class_id))
    
    nms_idx = nms(boxes, confidences, iou_threshold)
    new_boxes = []
    new_confidences = []
    new_classes = []
    for idx in nms_idx:
        new_boxes.append(boxes[idx])
        new_confidences.append(confidences[idx])
        new_classes.append(classes[idx])
    return new_boxes, new_confidences, new_classes


def draw_detections(image, boxes, scores, class_ids, mask_alpha):
    # Initializations
    mask_img = image.copy()
    det_img = image.copy()
    class_names = get_classes()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    # Create a list of colors for each class where each color is a tuple of 3 integer values
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(class_names), 3))

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


def save_image(file_path, result_image):
    cv2.imwrite(file_path, result_image)


def get_classes(class_file_name="coco_classes.names"):
    # Load MSCOCO class names
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def postprocess(outputs, image, confidence, iou, save_path):
    boxes, scores, class_ids = process_output(output=outputs, original_img_shape=image.shape,
                                              conf_threshold=confidence, iou_threshold=iou)
    res_raw = draw_detections(image=image, boxes=boxes,
                              scores=scores, class_ids=class_ids, mask_alpha=0.3)
    res = np.squeeze(res_raw)
    save_image(file_path=save_path, result_image=res)


def eval_process(model_output,
                 model_hw_shape,
                 origin_img_shape,
                 score_threshold,
                 nms_threshold,
                 dump_image=False):
    boxes, scores, class_ids = process_output(output=model_output, original_img_shape=origin_img_shape,
                                              conf_threshold=score_threshold, iou_threshold=nms_threshold)
    
    boxes = np.array(boxes)
    scores = np.array(scores)[:, np.newaxis]
    class_ids = np.array(class_ids)[:, np.newaxis]

    if len(boxes.shape) == 1:
        boxes = boxes[:, np.newaxis]
    
    bboxes = np.concatenate([boxes, scores, class_ids], axis=-1)
    return bboxes


def eval_postprocess(model_output,
                     model_hw_shape,
                     entry_dict,
                     score_threshold=0.001,
                     nms_threshold=0.65):

    bboxes_pr = eval_process(model_output,
                             model_hw_shape,
                             origin_img_shape=entry_dict[0]['origin_shape'],
                             score_threshold=score_threshold,
                             nms_threshold=nms_threshold,
                             dump_image=False)
    pred_result = []
    for one_bbox in bboxes_pr:
        one_result = {'bbox': one_bbox, 'mask': False}
        pred_result.append(one_result)
    return pred_result, entry_dict[0]['image_name']
