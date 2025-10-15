import torch
from ultralytics import YOLO

# 加载 YOLOv5su 模型
model = YOLO("./model/yolov5su.pt")

# 获取模型
pt_model = model.model

# 设置为评估模式
pt_model.eval()

# 指定 batch size = 1 
batch_size = 1
img_size = 640

# 创建虚拟输入
dummy_input = torch.randn(batch_size, 3, img_size, img_size)

# 导出为 ONNX
onnx_file_path = "./model/yolov5su.onnx"
torch.onnx.export(
    pt_model,
    dummy_input,
    onnx_file_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['output0'],
)

print(f"YOLOv5su (batch_size={batch_size}) has been converted to {onnx_file_path}")
