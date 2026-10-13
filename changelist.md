# Changelist

## 2026-10-10

- 在仓库根目录新增统一的 `CMakeLists.txt`，集中配置交叉编译架构、RNE-RT-Lib/OpenCV 依赖，并在一次构建中生成 ResNet-18 和 YOLOv5s 示例程序。
- 删除各模型目录下独立的 Makefile 和 CMakeLists.txt，避免分模块构建入口。
- 将示例程序源码收拢到 `examples/<model>/`，模型原始 ONNX、转换脚本/配置及已转换 tsmodel 按模型归档到顶层模型资源目录。
- 将数据集从 `examples/dataset/` 移到仓库一级目录 `dataset/`。
- 更新 Knight 配置和根目录 README 中的路径与使用说明，补充支持模型总表，以及模型转换、统一编译和板端执行流程。
- 新增根目录 `.gitignore` 规则，忽略 CMake 构建文件、Python 缓存和模型转换中间输出。
- 新增 `scripts/convert.sh` 与 `scripts/build.sh`，支持一次处理全部模型或通过参数指定单个模型；转换脚本在 Knight 容器内运行，编译脚本集中接收架构、工具链及 RNE 运行库路径。
- 删除模型目录下各自的 README，统一由根目录 README 维护使用说明。
- 按用户工作流重排根 README：先提供“测试已有模型”和“转换模型”两个入口，再分别说明环境准备、脚本用法和执行方法；模型规格说明单独维护。
- 在根 README 工作流程之前补充仓库定位、支持模型列表和芯片平台列表，并保留到详细模型规格目录的链接。
- 将模型资源目录 `convert/` 重命名为 `models/`，并将规格文档并入 `models/README.md`；同步更新配置、转换脚本和文档中的目录路径。
- 精简根目录 CMake 配置，仅保留当前使用的 `linux_a35_uclibc_32bit` 与 `linux_a53_uclibc` 两种架构，移除其他未使用的架构分支。
- 新增 `scripts/test_model.sh`，将 ResNet-18/YOLOv5s 的输入预处理、Knight 仿真、结果后处理和性能分析封装为可选择全部模型或指定模型执行的流程；在根 README 末尾补充使用说明。