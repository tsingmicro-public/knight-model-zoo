# 操作说明
## step 1: 确保配置好交叉编译工具链

## step 2: 编译 sample

`make clean; make`

## step 3: 将资源copy至板端

* 将输出 build_linux_a53/Release/rne_yoloworld_detection.elf 传输到板端
* 将 resource 目录传输到板端

## step 4: 运行sample

- ./rne_yoloworld_detection.elf 参数1 参数2 参数3

`./build_linux_a53/Release/rne_yoloworld_detection.elf resource/yolo-world-v2-s-image_quantize_r.tsmodel  resource/bus.jpg resource/coco80-norm-nch.bin`
`./build_linux_a53/Release/rne_yoloworld_detection.elf resource/yolo-world-v2-s-image_quantize_r_od.tsmodel  resource/bus.jpg resource/coco80-norm-nch.bin`


# sample 参数说明
| 参数 | 示例 | 说明|
| :- |:- |:- |
| 参数1|resource/yolo-world-v2-s-image_quantize_r.tsmodel 或 resource/yolo-world-v2-s-image_quantize_r_od.tsmodel | 网络配置和权重文件|
| 参数2|resource/1.jpg 或 resource/2.jpg 或 resource/3.jpg 或 resource/4.jpg 或resource/bus.jpg| 网络图片输入文件|
| 参数3|resource/coco80-norm-nch.bin| 网络text输入文件|


# 输出结果说明

* 输出结果为图片，保存在resource文件夹中，图片名为result.jpg
