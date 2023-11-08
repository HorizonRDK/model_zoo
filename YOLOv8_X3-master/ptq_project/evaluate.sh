## Accuracy evaluation for original_float_model and quantized_model
## Usage: sh evaluate.sh quanti/origin (num)

# Quantized_model evaluation
quanti_model_file="./modified_model_output/yolov8s_640x640_nv12_quantized_model.onnx"
quanti_input_layout="NHWC"

# Original_float_model evaluation
original_model_file="./modified_model_output/yolov8s_640x640_nv12_original_float_model.onnx"
original_input_layout="NCHW"

# Specify model
if [[ $1 =~ "origin" ]]; then
  model=$original_model_file
  transformer="origin"
  layout=$original_input_layout
  input_offset=128
else
  model=$quanti_model_file
  transformer="quanti"
  layout=$quanti_input_layout
  input_offset=128
fi

# Specify data path
image_path="../../data/coco_val2017/images/"
anno_path="../../data/coco_val2017/annotations/instances_val2017.json"

# Specify data num
if [ -z $2 ]; then 
  total_image_number=5000
else
  total_image_number=$2
fi

python3 -u ./det_evaluate.py \
  --model=${model} \
  --image_path=${image_path} \
  --annotation_path=${anno_path} \
  --input_layout=${layout} \
  --total_image_number=${total_image_number} \
  --input_offset ${input_offset} \
  --load_transformer ${transformer}