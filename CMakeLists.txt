cmake_minimum_required(VERSION 3.13)

# 头文件与运行库的根目录（包含 include/ 和 libs/），默认仓库根目录。
set(RNE_RT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "RNE runtime root (include/ + libs/)")

# 交叉编译器必须在 project() 之前选定，否则 CMake 会先用主机编译器。
if(ARCH STREQUAL "linux_a35")
    set(CMAKE_C_COMPILER   "aarch64-none-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-none-linux-gnu-g++")
elseif(ARCH STREQUAL "linux_a35_32bit")
    set(CMAKE_C_COMPILER   "arm-none-linux-gnueabihf-gcc")
    set(CMAKE_CXX_COMPILER "arm-none-linux-gnueabihf-g++")
elseif(ARCH STREQUAL "linux_a35_uclibc")
    set(CMAKE_C_COMPILER   "aarch64-ts-linux-uclibc-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-ts-linux-uclibc-g++")
elseif(ARCH STREQUAL "linux_a35_uclibc_32bit")
    set(CMAKE_C_COMPILER   "arm-ts-linux-uclibcgnueabihf-gcc")
    set(CMAKE_CXX_COMPILER "arm-ts-linux-uclibcgnueabihf-g++")
endif()

project(knight-model-zoo)

add_subdirectory(model_deploy/resnet18)
add_subdirectory(model_deploy/yolov5s)
