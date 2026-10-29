cd /data/model_convert/yoloxs

#准备bin数据
python make_image_input.py --input /data/dataset/bus.jpg --outpath /data/model_convert/yoloxs/output/pre


#仿真
Knight --chip TX5336AV200 run  \
--input /data/model_convert/yoloxs/output/pre/model_input.bin \
--model /data/model_convert/yoloxs/output/compile/yolox_s_quantize_r.tsmodel  \
--save-dir /data/model_convert/yoloxs/output/sim -fmt nchw

#仿真输出txt文件转numpy
show_sim_result --sim-data /data/model_convert/yoloxs/output/sim/result-817_p.txt --save-dir /data/model_convert/yoloxs/output/sim
show_sim_result --sim-data /data/model_convert/yoloxs/output/sim/result-843_p.txt --save-dir /data/model_convert/yoloxs/output/sim
show_sim_result --sim-data /data/model_convert/yoloxs/output/sim/result-869_p.txt --save-dir /data/model_convert/yoloxs/output/sim

#模型后处理。 scales为模型输出top_scale，需要根据实际量化结果指定该值
python post_process.py --image /data/dataset/bus.jpg --numpys  /data/model_convert/yoloxs/output/sim/result-817_p.npy /data/model_convert/yoloxs/output/sim/result-843_p.npy /data/model_convert/yoloxs/output/sim/result-869_p.npy  \
--scales 1 1 1 --save_dir /data/model_convert/yoloxs/output/sim


