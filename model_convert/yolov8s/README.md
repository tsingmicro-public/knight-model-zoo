# onnx 模型导出说明

Github工程地址：https://github.com/ultralytics/ultralytics

pth 模型转换为onnx 需要依赖模型定义代码，以版本ultralytics-8.3.23为例，需修改
在ultralytics/nn/modules/head.py里_inference函数修改如下图所示

![alt text](assets/image.png)


导出后为3路输出，示例如下

![alt text](assets/image-1.png)