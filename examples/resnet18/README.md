# resnet18

​

## 模型介绍

**YOLOv5** 是一种在图像检测中广泛使用的深度卷积神经网络，其速度快和检测精度高，是最著名的物体检测算法之一。

本示例中使用的模型来自以下开源项目：[https://github.com/ultralytics/yolov5/](https://github.com/ultralytics/yolov5/releases/tag/v7.0)​

**Resnet**（Residual Network， 残差网络）系列网络是一种在图像分类中广泛使用的深度卷积神经网络，同时也作为计算机视觉任务主干经典神经网络的一部分。

论文：[《Deep Residual Learning for Image Recognition》](https://arxiv.org/abs/1512.03385)​

开源模型链接：[https://github.com/pytorch/vision/blob/v0.11.0/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/v0.11.0/torchvision/models/resnet.py)​

数据集（ImageNet）：[http://www.image-net.org/](http://www.image-net.org/)​

## 模型转换

### 1）Knight环境准备

```bash
# 检查docker环境默认服务器中已安装docker
docker -v   
# 加载镜像
docker load -i ts.knight-3.8.0.9.tar.gz
#启动docker容器，localhost_dir为宿主机目录。
docker run -it -v ${localhost_dir}/knight-model-zoo:/data  ts.knight:3.8.0.9
```

### 2）预训练模型准备

resnet18.onnx 下载链接: [https://pan.baidu.com/s/1ou8UpnI-sWuRczFC\_nIZ9Q?pwd=tsnn ](https://pan.baidu.com/s/1cWKS3XT4jeGLv-ZxxEv29w?pwd=ic1j)​

并将模型放在目录 knight-model-zoo/examples/resnet18/model 下

### 3）转换为tsmodel &#x20;

```bash
# 容器内执行命令
cd /data/examples/resnet18/convert
Knight build --run-config resnet18_config.json 
```

### 4）仿真

```bash

#数据前处理
cd /data/examples/resnet18/convert
 
python3 make_image_input.py \
--input /data/examples/dataset/imagenet_random_10/ILSVRC2012_val_00000056_n02930766.JPEG \
--outpath /data/examples/resnet18/output/sim

#模型仿真推理
Knight --chip TX5326DV500 run --input /data/examples/resnet18/output/sim/model_input.bin \
--model /data/examples/resnet18/output/compile/resnet18_quantize_r.tsmodel  \
--format nchw --save-dir /data/examples/resnet18/output/sim

#后处理
python post_process.py --input /data/examples/resnet18/output/sim

```

### 5）性能分析

```bash
Knight profiling --chip TX5326DV500 --model /data/examples/resnet18/output/compile/resnet18_quantize_r.tsmodel \
--save-dir /data/examples/resnet18/output/compile
```

## 板端部署

### 1）准备RNE-RT-Lib

获取RNE Runtime库RNE-RT-Lib, 放至knight-model-zoo 目录下。

### 2) 配置交叉编译环境

根据不同芯片型号设置不同的交叉编译工具链，比如TX5326DV500对应的
gcc-ts-10.3-2023.2-x86\_64-arm-none-linux-uclibcgnueabihf.tar.xz 解压后的存放路径

* **方式一**

修改knight-model-zoo/RNE-RT-Lib/settings\_path\_linux.sh脚本，将tools\_dir设置成交叉编译工具链解压后的存放路径（最好是绝对路径）。

```bash
# 修改后 source 使该文件生效  
source settings_path_linux.sh  
```

* **方式二**

```bash
# 直接设置环境变量，其中TOOLCHAIN_ROOT是交叉编译工具链的绝对路径
export TOOLCHAIN_ROOT="<TOOLCHAIN_ROOT>"
export PATH="$TOOLCHAIN_ROOT/bin:$PATH"
```

### 2）编译生成板端部署程序&#x20;

修改 knight-model-zoo/examples/yolov5s/cpp/Makefile中的 ARCH 和ROOT\_PATH, 示例如下

```bash
ARCH ?= linux_a35_uclibc_32bit
ROOT_PATH ?= ../../../RNE-RT-Lib
```

执行make

```bash
make clean & make
```

此时在cpp目录下会生成文件build\_linux\_a35\_uclibc\_32bit/Release/resnet18\_demo.elf

### 3) 运行

把该elf文件，模型输入文件，模型tsmodel 挂载到开发板上，执行如下命令：

也可直接下载编译后的模型 ：resnet18\_quantize\_r.tsmodel

链接: [https://pan.baidu.com/s/1RB6L-2y-8N8dxvHMb8aTuQ?pwd=tsnn](https://pan.baidu.com/s/1RB6L-2y-8N8dxvHMb8aTuQ?pwd=tsnn) 提取码: tsnn

```bash

# ./resnet18_demo.elf 参数1 参数2
# 参数1   resnet18_quantize_r.tsmodel 网络配置和权重文件 
# 参数2  ILSVRC2012_val_00000056_n02930766.JPEG网络输入images目录，目录下可以多张或者单张，结果只保持最后推理的一张 

./build_linux_a35/Release/resnet18_demo.elf resnet18_quantize_r.tsmodel ILSVRC2012_val_00000056_n02930766.JPEG
```

​
