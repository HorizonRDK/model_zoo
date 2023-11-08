# 导入 YOLOv8
from ultralytics import YOLO

# 载入预训练权重
model = YOLO("best.pt")

# 对 Head 做修改，指定 opset=11，并且导出 ONNX
success = model.export(format="onnx", horizon=True, opset=11, simplify=True)
