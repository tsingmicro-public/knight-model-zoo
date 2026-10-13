# Knight Model Zoo

## 仓库介绍

本仓库是 TSingMicro（清微智能）芯片的模型仓库，集中收录各支持平台的模型文件、模型转换配置与脚本，以及板端测试代码，便于用户进行模型验证和部署。

## 支持的模型

| 任务 | 模型 | 示例代码 | 模型文件与转换资源 |
| --- | --- | --- | --- |
| 图像分类 | ResNet-18 | [model_deploy/resnet18](model_deploy/resnet18) | [model_convert/resnet18](model_convert/resnet18) |
| 目标检测 | YOLOv5s | [model_deploy/yolov5s](model_deploy/yolov5s) | [model_convert/yolov5s](model_convert/yolov5s) |

## 支持的平台

当前模型支持的平台包括：**TX5215x、TX5239x200/300、TX5112x、TX5239x201、TX5336x、TX5110x、TX5326x/TX5329x、TX5212x**。各模型对应的输入规格、校准数据和平台详情见[模型规格目录](model_convet/README.md)。

**本仓库支持功能：**

1. **测试已有模型**：准备板端编译环境，编译示例程序，然后在目标设备上运行仓库提供的 tsmodel 和测试图片。见[测试已有模型](#1-板载模型测试)。
2. **转换模型**：准备 Knight Docker 环境，在容器中运行一键转换脚本。见[转换模型](#2-模型转换)。

当前支持模型、输入规格和芯片平台请查阅[模型规格目录](model_convet/README.md)。数据集统一位于 [dataset](dataset)，模型原始文件、转换配置/脚本及转换产物按模型归档于 [model_convet](model_convet)。

## 1. 板载模型测试

### 1.1 环境准备

在编译机上准备以下依赖：

- **RNE-RT-Lib**：放在仓库根目录，或稍后通过 `RNE_RT_ROOT` 指定其路径。
- **目标平台交叉编译工具链**：将工具链的 `bin` 目录加入 `PATH`，或设置 `TOOLCHAIN_ROOT`。下载信息见本节下方。
- **目标设备运行环境**：设备上需有匹配的 RNE 运行库及其动态依赖。

工具链下载信息：

- `gcc-ts-10.3-2023.2-x86_64-arm-none-linux-uclibcgnueabihf.tar.xz`：[下载链接](https://pan.baidu.com/s/1Kq6JJICwhXa0yD-RsOdXLA?pwd=tsnn)，提取码 `tsnn`。
- `gcc-ts-10.3-2023.2-x86_64-aarch64-none-linux-uclibc.tar.xz`：[下载链接](https://pan.baidu.com/s/1O84K6LQMAgPsZTqI1BfR4g?pwd=tsnn)，提取码 `tsnn`。

CMake 使用仓库内 `3rdparty/` 中与目标架构对应的 OpenCV 库；RNE 运行库目录由根目录 `CMakeLists.txt` 检查。编译机和目标设备需使用匹配的 RNE 运行库版本。

### 1.2 一键编译

从仓库根目录运行脚本。默认编译所有示例，也可只编译指定模型：

```bash
export ARCH=linux_a35_uclibc_32bit
export TOOLCHAIN_ROOT="/absolute/path/to/toolchain"
export RNE_RT_ROOT="/absolute/path/to/RNE-RT-Lib"

./scripts/build.sh                 # 编译全部模型
./scripts/build.sh resnet18        # 只编译 ResNet-18
./scripts/build.sh yolov5s         # 只编译 YOLOv5s
```

默认输出在 `build/<ARCH>/bin/`。脚本支持 `ARCH`、`TOOLCHAIN_ROOT`、`RNE_RT_ROOT`、`BUILD_DIR`、`BUILD_TYPE`、`JOBS` 和 `USE_DYNAMIC_LIB` 环境变量。当前 `ARCH` 仅支持 `linux_a35_uclibc_32bit` 和 `linux_a53_uclibc` 两种构建目标；各模型支持的芯片平台请查阅[模型规格表](model_convert/README.md)。切换架构时建议使用新的 `BUILD_DIR`。

### 1.3 在目标设备运行

将对应 ELF、tsmodel 和输入图片目录复制到目标设备。以下命令以仓库目录结构和示例输入为例：

```bash
./build/linux_a35_uclibc_32bit/bin/resnet18_demo.elf \
  model_convert/resnet18/resnet18_quantize_r.tsmodel dataset/test_data

./build/linux_a35_uclibc_32bit/bin/yolov5s_demo.elf \
  model_convert/yolov5s/yolov5s_quantize_r.tsmodel dataset/test_data
```

程序参数依次为 tsmodel 文件和图片目录。运行目录中的模型、图片路径需按设备上的实际部署位置调整；YOLOv5s 检测结果图片写入程序的当前工作目录。若要测试转换后新生成的模型，请将命令中的 tsmodel 路径改为 `model_convert/<模型名>/output/compile/` 下对应产物。

## 2. 模型转换

### 2.1 环境准备

模型转换使用 Knight 工具链，需先准备 Knight Docker 镜像。示例镜像为 `ts.knight:3.8.0.9`：

```bash
docker -v
docker load -i ts.knight-3.8.0.9.tar.gz
```

转换脚本要求仓库在容器内挂载到 `/data`。从宿主机启动容器：

```bash
cd /path/to/knight-model-zoo
docker run -it -v "${PWD}:/data" ts.knight:3.8.0.9
```

### 2.2 一键转换

在容器内执行。

```bash
cd /data
Knight build --run-config ./model_convert/resnet18/resnet18_config.json 
Knight build --run-config ./model_convert/yolov5s/yolov5s_config.json
```

脚本按序调用各模型配置中的 `Knight build` 命令。ONNX 原始模型、配置和脚本位于 `model_convert/<模型名>/`，校准/测试数据位于 `dataset/`，转换产物写入各模型配置指定的 `model_convert/<模型名>/output/`。


## 3. 仿真测试与性能分析

仿真和性能分析使用 Knight 工具链，在仓库挂载到 `/data` 的 Knight 容器中运行。脚本会使用仓库自带的 tsmodel 和默认测试图片；仿真输入和结果保存在对应的 `model_convert/<模型名>/output/` 目录。

```bash
cd /data
# 仿真模型
sh ./model_convert/resnet18/run_sim.sh 
sh ./model_convert/yolov5s/run_sim.sh 

# 性能分析
Knight profiling --run-config ./model_convert/resnet18/resnet18_config.json 
Knight profiling --run-config ./model_convert/yolov5s/yolov5s_config.json

```

仿真默认使用 ResNet-18 的 `dataset/imagenet_random_10/ILSVRC2012_val_00000056_n02930766.JPEG` 和 YOLOv5s 的 `dataset/bus.jpg`。也可传入自定义图片路径,修改相应模型目录下run_sim.sh 脚本即可。 `CHIP` 可覆盖默认芯片型号 `TX5326DV500`，性能分析结果由 Knight 写入各模型的 `output/profiling/`。
