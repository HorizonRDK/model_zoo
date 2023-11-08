## Single-image inference using original best.onnx
from horizon_tc_ui import HB_ONNXRuntime
import numpy as np
import cv2

from postprocess import postprocess


def preprocess(image):
    height, width = 640, 640
    # Resize
    image = cv2.resize(image, (height, width))
    # BGR->RGB
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # HWC->CHW
    image = image[:, :, ::-1].transpose((2, 0, 1))
    # Scale
    image = image / 255
    # Expand to NCHW
    image_data = np.expand_dims(image, axis=0)
    return image_data


def main():
    # Read image in bgr
    image = cv2.imread("kite.jpg")
    # Load model for inference
    sess = HB_ONNXRuntime(model_file="./best.onnx")
    # (Optional) GPU acceleration
    sess.set_providers(["CUDAExecutionProvider"])
    # Get input and output names
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]
    # Data preprocess before inference
    feed_dict = dict()
    for input_name in input_names:
        feed_dict[input_name] = preprocess(image)
    # Inference for model with input datatype in float32
    outputs = sess.run_feature(output_names, feed_dict, input_offset=0)
    # Postprocess and visualize results
    postprocess(outputs, image, confidence=0.4, iou=0.4, save_path="./best_onnx.jpg")


if __name__ == '__main__':
    main()