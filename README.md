# Knight Model Zoo

## 简介

Knight Model Zoo基于 Knight 工具链开发, 提供了常见算法的部署例程. 例程包含导出tsmodel模型, 板端推理 tsmodel模型的流程。

## Knight工具链下载

​

### 镜像环境准备

```bash
# 检查docker环境默认服务器中已安装docker
docker -v   
# 加载镜像
docker load -i ts.knight-3.8.0.9.tar.gz
#启动docker容器，localhost_dir为宿主机目录。
docker run -it -v ${localhost_dir}/knight-model-zoo:/data  ts.knight:3.8.0.9
```

​

## 交叉编译工具链下载

* gcc-ts-10.3-2023.2-x86\_64-arm-none-linux-uclibcgnueabihf.tar.xz&#x20;

链接: [https://pan.baidu.com/s/1Kq6JJICwhXa0yD-RsOdXLA?pwd=tsnn](https://pan.baidu.com/s/1Kq6JJICwhXa0yD-RsOdXLA?pwd=tsnn) 提取码: tsnn

* gcc-ts-10.3-2023.2-x86\_64-aarch64-none-linux-uclibc.tar.xz

链接: [https://pan.baidu.com/s/1O84K6LQMAgPsZTqI1BfR4g?pwd=tsnn](https://pan.baidu.com/s/1O84K6LQMAgPsZTqI1BfR4g?pwd=tsnn) 提取码: tsnn

## 模型支持说明.

| **类别** | **模型名称** | **支持的芯片平台**                                                                                                       |
| ------ | -------- | ----------------------------------------------------------------------------------------------------------------- |
| 图像分类   | resnet18 | TX5215x\_TX5239x200\_TX5239x300&#xA;TX5112x\_TX5239x201&#xA;TX5336x &#xA;TX5110x&#xA;TX5326x\_TX5329x&#xA;TX5212x |
| 物体检测   | yolov5   | TX5215x\_TX5239x200\_TX5239x300&#xA;TX5112x\_TX5239x201&#xA;TX5336x &#xA;TX5110x&#xA;TX5326x\_TX5329x&#xA;TX5212x |

​
