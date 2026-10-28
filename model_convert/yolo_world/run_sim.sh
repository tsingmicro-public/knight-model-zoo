#!/usr/bin/env bash

cd /data/model_convert/yolo_world
#前处理
python3 yoloword_onnx_pre_post.py \
--input /data/dataset/bus.jpg \
--outpath /data/model_convert/yolo_world/output/pre --pre_processing


#仿真推理
Knight --chip TX5336AV200 run \
--input /data/model_convert/yolo_world/output/pre/model_input.bin:/data/model_convert/yolo_world/coco80-norm-nch.bin \
--model /data/model_convert/yolo_world/output/compile/yolo-world-v2-s-image_quantize_r.tsmodel \
--save-dir /data/model_convert/yolo_world/output/sim -fmt nchw


#后处理
python3 yoloword_onnx_pre_post.py  \
--input  /data/model_convert/yolo_world/output/sim --outpath  /data/model_convert/yolo_world/output/sim \
--post_processing \
--data /TS-KnightDemo/Samples/yoloworld/data/test_data/bus.jpg
