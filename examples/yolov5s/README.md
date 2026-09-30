# yolov5

## 模型介绍

**YOLOv5** 是一种在图像检测中广泛使用的深度卷积神经网络，其速度快和检测精度高，是最著名的物体检测算法之一。

本示例中使用的模型来自以下开源项目：[https://github.com/ultralytics/yolov5/](https://github.com/ultralytics/yolov5/releases/tag/v7.0)​


## 模型转换

### 1）Knight环境准备

```bash
# 检查docker环境默认服务器中已安装docker
docker -v   
# 加载镜像
docker load -i ts.knight-3.8.0.9.tar.gz
#启动docker容器，localhost_dir为宿主机目录。
docker run -v ${localhost_dir}/knight-model-zoo:/data -it ts.knight:3.8.0.9
```

### 2）预训练模型准备

yolov5s.onnx 下载链接: [https://pan.baidu.com/s/1cWKS3XT4jeGLv-ZxxEv29w?pwd=ic1j](https://pan.baidu.com/s/1cWKS3XT4jeGLv-ZxxEv29w?pwd=ic1j)&#x20;

并将模型放在目录 knight-model-zoo/examples/yolov5s/model 下

### 3）转换为tsmodel &#x20;

```bash
# 容器内执行命令
cd /data/examples/yolov5s/convert
Knight build --run-config yolov5s_config.json 
```

### 4）仿真

```bash

#前处理
cd /data/examples/yolov5s/convert
python3 yolov5_pre_post.py \
--input /data/examples/dataset/bus.jpg \
--outpath /data/examples/yolov5s/output/pre --pre_processing

#仿真推理
Knight --chip TX5326DV500 run \
--input /data/examples/yolov5s/output/pre/model_input.bin \
--model /data/examples/yolov5s/output/compile/yolov5s_quantize_r.tsmodel \
--save-dir /data/examples/yolov5s/output/sim -fmt nchw

#后处理
python3 yolov5_pre_post.py \
--input  /data/examples/yolov5s/output/sim  --outpath /data/examples/yolov5s/output/sim \
--post_processing \
--data /data/examples/dataset/bus.jpg

```

### 5）性能分析

```bash
Knight profiling --chip TX5326DV500 --model /data/examples/yolov5s/output/compile/yolov5s_quantize_r.tsmodel 
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

此时在cpp目录下会生成文件build\_linux\_a35\_uclibc\_32bit/Release/yolov5s\_demo.elf

### 3) 运行

把该elf文件，模型输入文件，模型tsmodel 挂载到开发板上，执行如下命令：

也可直接下载编译后的模型 yolov5s\_quantize\_r.tsmodel

下载链接: [https://pan.baidu.com/s/18-YzF7pQO1EDVpWi9c8Mkg?pwd=xf4a](https://pan.baidu.com/s/18-YzF7pQO1EDVpWi9c8Mkg?pwd=xf4a)&#x20;

```bash

# ./rne_yolov5_detection.elf 参数1 参数2
# 参数1   yolov5s_quantize_r.tsmodel  网络配置和权重文件 
# 参数2  /knight-model-zoo/examples/dataset 网络输入images目录，目录下可以多张或者单张，结果只保持最后推理的一张 

./build_linux_a35/Release/yolov5s_demo.elf yolov5s_quantize_r.tsmodel examples/dataset
```

输出结果为图片，保存在程序执行的当前目录下，图片名为default\_output.jpg

​

​

​
