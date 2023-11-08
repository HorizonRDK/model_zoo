# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
"""MS COCO Instance Segmentation Evaluate Metrics."""
from __future__ import absolute_import

import sys
import io
import os
import warnings
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class MSCOCODetMetric(object):
    class_names = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",  # noqa
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
        self,
        annotation_file,
        with_mask=False,
        overwrite=True,
        cleanup=False,
    ):

        save_filename = "./tmp_file"
        anno_file = os.path.abspath(os.path.expanduser(annotation_file))
        self._coco_anno = COCO(anno_file)
        if with_mask:
            import pycocotools.mask as cocomask
            self._cocomask = cocomask
        else:
            self._cocomask = None

        self._contiguous_id_to_json = {}
        self._contiguous_id_to_name = {}
        class_cat = self._coco_anno.dataset["categories"]
        name2jsonID = {}
        for (i, cat) in enumerate(class_cat):
            name2jsonID[cat["name"]] = cat["id"]
        self._with_bg = False
        for (i, name) in enumerate(MSCOCODetMetric.class_names):
            name = name.split("|")[-1]
            if name not in ["background", "__background__"]:
                assert name in name2jsonID
                self._contiguous_id_to_json[i] = name2jsonID[name]
                self._contiguous_id_to_name[i] = name
            else:
                self._with_bg = True
        self._filename = os.path.abspath(os.path.expanduser(save_filename))
        if os.path.exists(self._filename) and not overwrite:
            raise RuntimeError(
                "%s already exists, set overwrite=True to overwrite" %
                self._filename)
        self._results = {}
        self._with_mask = with_mask
        self._cleanup = cleanup
        self.IoU_lo_thresh = 0.5
        self.IoU_hi_thresh = 0.95

    def reset(self):
        self._results = {}

    def update(self, pred_result, image_name):
        """
        Parameters
        ----------
        pred_result: list of dict
            Each element is a dict, with key ``bbox``, ``mask`` is required
            if with_mask is True.
            bbox is an array with shape (6, ), where 6 represents
            (x1, y1, x2, y2, score, cls_id).
            mask is an array with shape (H, W), which is the same to
            original image.
        image_name: str
            Image name
        """
        assert isinstance(pred_result, list)
        for pred in pred_result:
            assert isinstance(pred, dict)
            assert "bbox" in pred, "missing bbox for prediction"
            if self._with_mask:
                assert "mask" in pred, "missing mask for prediction"
        if image_name in self._results:
            warnings.warn("warning: you are overwriting {}".format(image_name))

        parsed_name = image_name.strip()
        parsed_name = parsed_name.split(".")[0]
        image_id = int(parsed_name[-12:])
        inst_list = []
        for pred in pred_result:
            coco_inst = {}
            bbox = pred["bbox"].reshape((-1, ))
            assert bbox.shape == (6, ), (
                "bbox should with shape (6,), get %s" % bbox.shape)
            coco_inst.update({
                "image_id":
                    image_id,
                "category_id":
                    self._contiguous_id_to_json[int(bbox[5])],
                "score":
                    float(bbox[4]),
                "bbox": [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2] - bbox[0]),
                    float(bbox[3] - bbox[1]),
                ],
            })
            if self._with_mask:
                mask = pred["mask"]
                rle = self._cocomask.encode(
                    np.array(mask[:, :, np.newaxis], order="F"))[0]
                rle["counts"] = rle["counts"].decode("ascii")
                coco_inst.update({"segmentation": rle})
            inst_list.append(coco_inst)
        self._results[image_name] = inst_list

    def get(self):
        self._dump_json()
        bbox_names, bbox_values = self._update("bbox")
        if self._with_mask:
            mask_names, mask_values = self._update("segm")
        else:
            mask_names = []
            mask_values = []
        names = bbox_names + mask_names
        values = bbox_values + mask_values
        return (names, values)

    def __del__(self):
        if os.path.exists(self._filename):
            try:
                os.remove(self._filename)
            except IOError as err:
                warnings.warn(str(err))

    # internal utils
    def _dump_json(self):
        recorded_size = len(self._results)
        anno_size = len(self._coco_anno.getImgIds())
        if not anno_size == recorded_size:
            warnings.warn("Recorded {} out of {} validation images, "
                          "incompelete results".format(recorded_size,
                                                       anno_size))
        try:
            ret = []
            for (_, v) in self._results.items():
                ret.extend(v)
            with open(self._filename, "w") as f:
                json.dump(ret, f)
        except IOError as e:
            raise RuntimeError(
                "Unable to dump json file, ignored. What(): {}".format(str(e)))

    def _update(self, anno_type):
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5)
                           & (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        pred = self._coco_anno.loadRes(self._filename)
        coco_eval = COCOeval(self._coco_anno, pred, anno_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        ind_lo = _get_thr_ind(coco_eval, self.IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, self.IoU_hi_thresh)
        precision = coco_eval.eval["precision"][ind_lo:(ind_hi + 1), :, :, 0,
                                                2]
        ap_default = np.mean(precision[precision > -1])
        names, values = ([], [])
        names.append("====== Summary {} metrics ======\n".format(anno_type))
        # catch coco print string, don't want directly print here
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        coco_eval.summarize()
        coco_summary = sys.stdout.getvalue()
        sys.stdout = _stdout
        values.append(str(coco_summary).strip())
        # collect MAP for each class
        for (cls_ind, cls_name) in self._contiguous_id_to_name.items():
            precision = coco_eval.eval["precision"][ind_lo:(ind_hi + 1), :,
                                                    cls_ind -
                                                    int(self._with_bg), 0, 2]
            ap = np.mean(precision[precision > -1])
            names.append(cls_name)
            values.append("{:.1f}".format(100 * ap))
        # put mean AP at last, for comparing perf
        names.append(
            "====== MeanAP @ IoU=[{:.2f},{:.2f} for {} ======\n".format(
                self.IoU_lo_thresh, self.IoU_hi_thresh, anno_type))
        values.append("{:.1f}".format(100 * ap_default))
        return (names, values)
