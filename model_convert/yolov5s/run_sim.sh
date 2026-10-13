#前处理
cd /data/model_convert/yolov5s
python3 yolov5_pre_post.py \
--input /data/dataset/bus.jpg \
--outpath /data/model_convert/yolov5s/output/pre --pre_processing

#仿真推理
Knight --chip TX5326DV500 run \
--input /data/model_convert/yolov5s/output/pre/model_input.bin \
--model /data/model_convert/yolov5s/output/compile/yolov5s_quantize_r.tsmodel \
--save-dir /data/model_convert/yolov5s/output/sim -fmt nchw

#后处理
python3 yolov5_pre_post.py \
--input  /data/model_convert/yolov5s/output/sim  --outpath /data/model_convert/yolov5s/output/sim \
--post_processing \
--data /data/dataset/bus.jpg

