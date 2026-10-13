#数据前处理
cd /data/model_convert/resnet18
 
python3 make_image_input.py \
--input /data/dataset/imagenet_random_10/ILSVRC2012_val_00000056_n02930766.JPEG \
--outpath /data/model_convert/resnet18/output/sim

#模型仿真推理
Knight --chip TX5326DV500 run --input /data/model_convert/resnet18/output/sim/model_input.bin \
--model /data/model_convert/resnet18/output/compile/resnet18_quantize_r.tsmodel  \
--format nchw --save-dir /data/model_convert/resnet18/output/sim

#后处理
python post_process.py --input /data/model_convert/resnet18/output/sim
