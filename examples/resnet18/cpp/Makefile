ARCH ?= linux_a35_uclibc_32bit
ROOT_PATH ?= ../../../RNE-RT-Lib
OPENCV_ROOT_PATH ?= ../../../3rdparty
USE_DYNAMIC_LIB ?= 0

ifeq ($(strip $(ARCH)),linux_a35)
	CC_PREFIX := aarch64-none-linux-gnu-
	CPU_FLAGS := -march=armv8-a -mcpu=cortex-a35
	LIBS_DIR := $(if $(filter 1,$(USE_DYNAMIC_LIB)),bin,libs)
else ifeq ($(strip $(ARCH)),linux_a35_32bit)
	CC_PREFIX := arm-none-linux-gnueabihf-
	CPU_FLAGS := -mcpu=cortex-a35 -mthumb -mfpu=neon -mfloat-abi=hard
	LIBS_DIR := $(if $(filter 1,$(USE_DYNAMIC_LIB)),bin,libs)
else ifeq ($(strip $(ARCH)),linux_a35_uclibc)
	CC_PREFIX := aarch64-ts-linux-uclibc-
	CPU_FLAGS := -march=armv8-a -mcpu=cortex-a35
	LIBS_DIR := $(if $(filter 1,$(USE_DYNAMIC_LIB)),uclibc_bin,uclibc_libs)
else ifeq ($(strip $(ARCH)),linux_a35_uclibc_32bit)
	CC_PREFIX := arm-ts-linux-uclibcgnueabihf-
	CPU_FLAGS := -mcpu=cortex-a35 -mthumb -mfpu=neon -mfloat-abi=hard
	LIBS_DIR := $(if $(filter 1,$(USE_DYNAMIC_LIB)),uclibc_bin,uclibc_libs)
else
	CC_PREFIX :=
	CPU_FLAGS :=
	LIBS_DIR := $(if $(filter 1,$(USE_DYNAMIC_LIB)),bin,libs)
endif

ifneq ($(findstring 32bit, $(ARCH)),)
    LIBS_BIT := 32bit
else
    LIBS_BIT := 64bit
endif

ifeq ($(strip $(ARCH)),linux_a35_uclibc_32bit)
    X_LIB_L_FLAGS := -lrne_pal_linux_a35,-lgcc,-lc,-lm,-lpthread,-lopencv_world,-lzlib,-llibjpeg-turbo,-lade,-llibopenjp2,-llibprotobuf,-llibwebp,-lquirc,-llibpng,-ltegra_hal,-ldl
else
    X_LIB_L_FLAGS := -lrne_pal_linux_a35,-lgcc,-lc,-lm,-lpthread,-lopencv_world,-llibtiff,-lzlib,-llibjpeg-turbo,-lade,-llibopenjp2,-llibprotobuf,-llibwebp,-lquirc,-llibpng,-ldl
endif

BUILD_PATH := build_$(ARCH)
DEF_FLAGS := -DLOG_TIME


ifeq ($(strip $(ARCH)),linux_a35_uclibc_32bit)
    INC_FLAGS := -I$(ROOT_PATH)/include/ \
                 -I$(ROOT_PATH)/open_source/crc \
                 -I$(ROOT_PATH)/open_source/gp_layers \
                 -I$(ROOT_PATH)/open_source/record_file \
                 -I$(ROOT_PATH)/open_source/common \
                 -I$(OPENCV_ROOT_PATH)/opencv4.5.5_uclibc/include/opencv4
    OPENCV_DIR := $(OPENCV_ROOT_PATH)/opencv4.5.5_uclibc
else
    INC_FLAGS := -I$(ROOT_PATH)/include/ \
                 -I$(ROOT_PATH)/open_source/crc \
                 -I$(ROOT_PATH)/open_source/gp_layers \
                 -I$(ROOT_PATH)/open_source/record_file \
                 -I$(ROOT_PATH)/open_source/common \
                 -I$(OPENCV_ROOT_PATH)/opencv4.5.5/include/opencv4
    OPENCV_DIR := $(OPENCV_ROOT_PATH)/opencv4.5.5
endif

CC := $(CC_PREFIX)gcc
CXX := $(CC_PREFIX)g++
LINKER := $(CXX)
RM := rm -rf

ifeq ($(MAKECMDGOALS),release)
	BUILD_DIR := $(BUILD_PATH)/Release
	OPT_LEVEL := -O2
else ifeq ($(MAKECMDGOALS),debug)
	BUILD_DIR := $(BUILD_PATH)/Debug
	OPT_LEVEL := -O0
	else
	BUILD_DIR := $(BUILD_PATH)/Release
	OPT_LEVEL := -O2
endif

CC_FLAGS := $(CPU_FLAGS) $(OPT_LEVEL) $(DEF_FLAGS) -std=c99 -Wall -Werror -fPIC -ffunction-sections -fdata-sections $(INC_FLAGS)
CXX_FLAGS := $(CPU_FLAGS) $(OPT_LEVEL) $(DEF_FLAGS) -std=c++11 -Wall -Werror -fPIC -ffunction-sections -fdata-sections $(INC_FLAGS)
LINK_FLAGS := $(OPT_LEVEL)

#SRC_DIRS := src ../../open_source/crc ../../open_source/gp_layers ../../open_source/record_file ../../open_source/common

SRC_DIRS := src \
    $(ROOT_PATH)/open_source/crc \
    $(ROOT_PATH)/open_source/gp_layers \
    $(ROOT_PATH)/open_source/record_file \
    $(ROOT_PATH)/open_source/common \
    $(ROOT_PATH)/open_source/utils

#C_SRCS := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c))
#CC_SRCS := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cpp))

C_SRCS  := $(shell find $(SRC_DIRS) -name '*.c'   2>/dev/null)
CC_SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' 2>/dev/null)

C_OBJS := $(addprefix $(BUILD_DIR)/,$(C_SRCS:.c=.o))
CC_OBJS := $(addprefix $(BUILD_DIR)/,$(CC_SRCS:.cpp=.o))

#TARGET_NAME := $(notdir $(CURDIR)).elf
TARGET_NAME := resnet18_demo.elf
TARGET := $(BUILD_DIR)/$(TARGET_NAME)
ELFSIZE := $(TARGET).size

.PHONY: all release debug clean run pre_build mkdir_build

# 默认目标
all: pre_build release

# Release 目标
release: $(TARGET) $(ELFSIZE)

# Debug 目标
debug: CC_FLAGS += -g
debug: $(TARGET) $(ELFSIZE)

# 链接规则
$(TARGET): $(C_OBJS) $(CC_OBJS)
	@echo 'Linking target: $@'
	$(LINKER) $(LINK_FLAGS) -o $@ $^ \
    -L$(ROOT_PATH)/$(LIBS_DIR)/$(LIBS_BIT) \
    -L$(ROOT_PATH)/libs/$(LIB_SUB_DIR) \
    -L$(OPENCV_DIR)/libs/ \
    -L$(OPENCV_DIR)/libs/opencv4/3rdparty \
    -Wl,--start-group,-lrne_rt_g3,-lrne_common,$(X_LIB_L_FLAGS),-end-group -fopenmp
# 编译规则
$(BUILD_DIR)/%.o: %.c
	@echo 'Compiling file: $<'
	@mkdir -p $(dir $@)
	$(CC) $(CC_FLAGS) -MMD -MP -MF"$(@:.o=.d)" -MT"$@" -c -o $@ $<

$(BUILD_DIR)/%.o: %.cpp
	@echo 'Compiling file: $<'
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -MMD -MP -MF"$(@:.o=.d)" -MT"$@" -c -o $@ $<

# 计算目标文件大小
$(ELFSIZE): $(TARGET)
	@echo 'Calculating size: $<'
	$(CC_PREFIX)size $< | tee $@


clean:
	@-$(RM) build*
	@-$(RM) open_source
	@echo 'Clean done'

run: all
	$(TARGET)

pre_build: mkdir_build

mkdir_build:
	@mkdir -p $(BUILD_DIR)

.PHONY: clean all pre_build mkdir_build
