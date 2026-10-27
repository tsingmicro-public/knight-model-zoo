#前处理
cd /data/model_convert/yolov8s

python3 make_image_input.py --input /data/dataset/bus.jpg --outpath /data/model_convert/yolov8s/output/pre


#仿真推理
Knight --chip TX5326DV500 run \
--input /data/model_convert/yolov8s/output/pre/model_input.bin \
--model /data/model_convert/yolov8s/output/compile/yolov8s_quantize_r.tsmodel \
--save-dir /data/model_convert/yolov8s/output/sim -fmt nchw

#后处理

#仿真输出txt文件转numpy
show_sim_result --sim-data /data/model_convert/yolov8s/output/sim/result-699_p.txt --save-dir /data/model_convert/yolov8s/output/sim
show_sim_result --sim-data /data/model_convert/yolov8s/output/sim/result-719_p.txt --save-dir /data/model_convert/yolov8s/output/sim
show_sim_result --sim-data /data/model_convert/yolov8s/output/sim/result-739_p.txt --save-dir /data/model_convert/yolov8s/output/sim

python post_process.py --image /data/dataset/bus.jpg --img-size 640 \
--numpys /data/model_convert/yolov8s/output/sim/result-699_p.npy /data/model_convert/yolov8s/output/sim/result-719_p.npy \
/data/model_convert/yolov8s/output/sim/result-739_p.npy --scales  1 1 1 --save_dir /data/model_convert/yolov8s/output/sim


