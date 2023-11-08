## Single-image inference using modified_model_output/yolov8s_640x640_nv12_original_float_model.onnx
import sys
sys.path.append("./python/data/")
from transformer import *
from dataloader import *
from horizon_tc_ui import HB_ONNXRuntime
import numpy as np
import cv2

from postprocess import postprocess


# Create preprocess transformer
def infer_transformers(input_shape, input_layout):
    transformers = [
        # Resize
        ResizeTransformer(target_size=input_shape, mode="opencv"),
        # HWC->CHW
        HWC2CHWTransformer(),
        # BGR->RGB
        BGR2RGBTransformer(data_format=input_layout[1:]),
        # RGB->NV12
        RGB2NV12Transformer(data_format=input_layout[1:]),
        # NV12->YUV444
        NV12ToYUV444Transformer(target_size=input_shape, yuv444_output_layout=input_layout[1:]),
    ]
    return transformers


def preprocess(image_name, is_processed):
    input_shape = (640, 640)
    input_layout = "NCHW"
    
    transformers = infer_transformers(input_shape, input_layout)
    origin_image, processed_image = SingleImageDataLoaderWithOrigin(transformers, image_name, imread_mode="opencv")
    return processed_image if is_processed else origin_image


def main():
    image_name = "kite.jpg"
    # Load model for inference
    sess = HB_ONNXRuntime(model_file="./modified_model_output/yolov8s_640x640_nv12_original_float_model.onnx")
    # (Optional) GPU acceleration
    sess.set_providers(["CUDAExecutionProvider"])
    # Get input and output names
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]
    # Data preprocess before inference
    feed_dict = dict()
    for input_name in input_names:
        feed_dict[input_name] = preprocess(image_name, is_processed=True)
    # Inference for model with input in (RGB/BGR/NV12/YUV444/GRAY, dtype=uint8)
    outputs = sess.run(output_names, feed_dict, input_offset=128)
    # Postprocess and visualize results
    origin_image = np.squeeze(preprocess(image_name, is_processed=False))
    postprocess(outputs, origin_image, confidence=0.4, iou=0.4, save_path="./original_float_model.jpg")


if __name__ == '__main__':
    main()