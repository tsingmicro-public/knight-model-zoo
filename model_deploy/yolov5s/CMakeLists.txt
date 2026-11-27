cmake_minimum_required(VERSION 3.13)

project(sample)

set(ARCH "linux_a35" CACHE STRING "Target architecture")

if(NOT DEFINED USE_DYNAMIC_LIB)
    set(USE_DYNAMIC_LIB 0)
endif()

# 本示例目录：include / libs / open_source 均拷贝到该目录下
set(SAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# knight-model-zoo 仓库根目录：OpenCV 位于 3rdparty 下
set(REPO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(ARCH STREQUAL "linux_a35")
    set(CC_PREFIX "aarch64-none-linux-gnu-")
    set(CPU_FLAGS "-march=armv8-a -mcpu=cortex-a35")
    set(X_LIB_L_FLAGS rne_pal_linux_a35 gcc)
    set(OPENCV_DIR_NAME "opencv4.5.5")
    set(CV_3RD_PARTY_LIBS libtiff)
elseif(ARCH STREQUAL "linux_a35_32bit")
    set(CC_PREFIX "arm-none-linux-gnueabihf-")
    set(CPU_FLAGS "-mcpu=cortex-a35 -mthumb -mfpu=neon -mfloat-abi=hard")
    set(X_LIB_L_FLAGS rne_pal_linux_a35 gcc)
    set(OPENCV_DIR_NAME "opencv4.5.5")
    set(CV_3RD_PARTY_LIBS libtiff)
elseif(ARCH STREQUAL "linux_a35_uclibc")
    set(CC_PREFIX "aarch64-ts-linux-uclibc-")
    set(CPU_FLAGS "-march=armv8-a -mcpu=cortex-a35")
    set(X_LIB_L_FLAGS rne_pal_linux_a35 gcc)
    set(OPENCV_DIR_NAME "opencv4.5.5")
    set(CV_3RD_PARTY_LIBS libtiff)
elseif(ARCH STREQUAL "linux_a35_uclibc_32bit")
    set(CC_PREFIX "arm-ts-linux-uclibcgnueabihf-")
    set(CPU_FLAGS "-mcpu=cortex-a35 -mthumb -mfpu=neon -mfloat-abi=hard")
    set(X_LIB_L_FLAGS rne_pal_linux_a35 gcc)
    set(OPENCV_DIR_NAME "opencv4.5.5_uclibc")
    set(CV_3RD_PARTY_LIBS tegra_hal)
else()
    set(CC_PREFIX "")
    set(CPU_FLAGS "")
    set(X_LIB_L_FLAGS rne_pal_sim rne_vm gcc stdc++)
    set(OPENCV_DIR_NAME "opencv4.5.5")
    set(CV_3RD_PARTY_LIBS libtiff)
endif()

if(ARCH MATCHES "32bit")
    set(LIBS_BIT "32bit")
else()
    set(LIBS_BIT "64bit")
endif()

set(CMAKE_C_COMPILER "${CC_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${CC_PREFIX}g++")

set(LIBS_DIR "${RNE_RT_ROOT}/libs/${LIBS_BIT}")
set(OPENCV_DIR "${REPO_DIR}/3rdparty/${OPENCV_DIR_NAME}")

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CPU_FLAGS_COMMON "-Wall -Werror -Wno-format-truncation -fPIC -ffunction-sections -fdata-sections")
set(CMAKE_C_FLAGS_RELEASE "${CPU_FLAGS} -std=c99 -O2 ${CPU_FLAGS_COMMON}")
set(CMAKE_CXX_FLAGS_RELEASE "${CPU_FLAGS} -std=c++11 -O2 ${CPU_FLAGS_COMMON}")
set(CMAKE_C_FLAGS_DEBUG "${CPU_FLAGS} -std=c99 -O0 -g ${CPU_FLAGS_COMMON}")
set(CMAKE_CXX_FLAGS_DEBUG "${CPU_FLAGS} -std=c++11 -O0 -g ${CPU_FLAGS_COMMON}")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# common 为各模型共享，放在 model_deploy 目录下
set(OPEN_SOURCE_DIR ${SAMPLE_DIR}/../common)

include_directories(
    ${RNE_RT_ROOT}/include
    ${OPEN_SOURCE_DIR}/crc
    ${OPEN_SOURCE_DIR}/gp_layers
    ${OPEN_SOURCE_DIR}/record_file
    ${OPEN_SOURCE_DIR}/common
    ${OPENCV_DIR}/include/opencv4
)

# 主程序 + open_source 中的辅助源码
file(GLOB OPEN_SOURCE_C_FILES
    ${OPEN_SOURCE_DIR}/crc/*.c
    ${OPEN_SOURCE_DIR}/gp_layers/*.c
    ${OPEN_SOURCE_DIR}/record_file/*.c)
file(GLOB OPEN_SOURCE_CXX_FILES ${OPEN_SOURCE_DIR}/common/*.cpp)

link_directories(${LIBS_DIR} ${OPENCV_DIR}/libs ${OPENCV_DIR}/libs/opencv4/3rdparty)

add_compile_definitions(DISABLE_PRINT_DEBUG_LAYER RESULT_ALL)

get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(PROJECT_NAME ${CURRENT_DIR_NAME}_demo.elf)
add_executable(${PROJECT_NAME}
    ${SAMPLE_DIR}/main.cpp
    ${OPEN_SOURCE_C_FILES}
    ${OPEN_SOURCE_CXX_FILES})

set(LINKER_GROUP_START -Wl,--start-group)
set(LINKER_GROUP_END -Wl,--end-group)
set(LL_LINK_LIBS ${X_LIB_L_FLAGS} rne_common rne_rt_g3 c m pthread opencv_world ${CV_3RD_PARTY_LIBS} zlib libjpeg-turbo ade libopenjp2 libprotobuf libwebp quirc libpng dl)

target_link_libraries(${PROJECT_NAME} ${LINKER_GROUP_START} ${LL_LINK_LIBS} ${LINKER_GROUP_END})
