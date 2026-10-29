# 支持的模型规格

本目录集中维护仓库支持模型、任务、输入规格和适用芯片平台。仓库根目录 README 负责操作流程，本页只维护模型规格信息。

| 任务 | 模型 | 输入规格 | 校准数据 | 模型及配置 | 支持的平台 |
| --- | --- | --- | --- | --- | --- |
| 图像分类 | ResNet-18 | 单输入，224×224 RGB 图像；推理示例为 UINT8 输入 | ImageNet 子集，`dataset/imagenet_random_10/` | [ONNX、tsmodel、配置与脚本](resnet18/) | TX5215x、TX5239x200/300、TX5112x、TX5239x201、TX5336x、TX5110x、TX5326x/TX5329x、TX5212x |
| 图像分类 | Mobilenetv2 | 单输入，224×224 RGB 图像；推理示例为 UINT8 输入 | ImageNet 子集，`dataset/imagenet_random_10/` | [ONNX、tsmodel、配置与脚本](mobilenet/) | TX5215x、TX5239x200/300、TX5112x、TX5239x201、TX5336x、TX5110x、TX5326x/TX5329x、TX5212x |
| 目标检测 | YOLOv5s | 单输入，640×640 RGB 图像；80 类 COCO 检测 | COCO 子集，`dataset/coco_random_10/` | [ONNX、tsmodel、配置与脚本](yolov5s/) | TX5215x、TX5239x200/300、TX5112x、TX5239x201、TX5336x、TX5110x、TX5326x/TX5329x、TX5212x |
| 目标检测 | YOLOv8s | 单输入，640×640 RGB 图像；80 类 COCO 检测 | COCO 子集，`dataset/coco_random_10/` | [ONNX、tsmodel、配置与脚本](yolov8s/) | TX5215x、TX5239x200/300、TX5112x、TX5239x201、TX5336x、TX5110x、TX5326x/TX5329x、TX5212x |
| 目标检测 | YOLOWorld | 单输入，640×640 RGB 图像；80 类 COCO 检测 | COCO 子集，`dataset/coco_random_10/` | [ONNX、tsmodel、配置与脚本](yolo_world/) |TX5336x|
| 目标检测 | YOLOxs | 单输入，640×640 RGB 图像；80 类 COCO 检测 | COCO 子集，`dataset/coco_random_10/` | [ONNX、tsmodel、配置与脚本](yoloxs/) |TX5336x|

## 示例源码

- ResNet-18：[`model_deploy/resnet18/main.cpp`](../examples/resnet18/main.cpp)
- Mobilenetv2：[`model_deploy/mobilenet/main.cpp`](../examples/mobilenet/main.cpp)
- YOLOv5s：[`model_deploy/yolov5s/main.cpp`](../examples/yolov5s/main.cpp)
- YOLOv8s：[`model_deploy/mobilenet/main.cpp`](../examples/yolov8s/main.cpp)
- YOLOWorld：[`model_deploy/mobilenet/main.cpp`](../examples/yolov_world_/main.cpp)
模型输入规格依据仓库配置和前后处理脚本整理；
芯片平台沿用仓库原有支持列表。新增模型时，请同步更新本表、根目录 `CMakeLists.txt`、一键脚本和根 README 的流程说明。
