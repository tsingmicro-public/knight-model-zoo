# -*- coding: UTF-8 -*-
'''将指定的单张 image 文件转化为 .bin 文件
'''
from pathlib import Path
import numpy as np
import argparse
import cv2

def __parse_args():
    parser = argparse.ArgumentParser(description='make input data scripts')
    parser.add_argument('--input',
            type=str,
            default='',
            help='The path of a single image file')
    parser.add_argument('--outpath',
            type=str,
            default='',
            help='The path of output bin')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = __parse_args()

    img_file = Path(args.input)
    assert img_file.is_file(), f"expect an image file, but got {img_file}"

    # 单张图片前处理
    input_data = np.zeros((1, 3, 224, 224), dtype=np.float32)

    img = cv2.imread(str(img_file))
    assert img is not None, f"failed to read image: {img_file}"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = max(img.shape) / 256
    interp = cv2.INTER_LINEAR if r < 1 else cv2.INTER_AREA
    img = cv2.resize(img, (256, 256), interpolation=interp)
    input_data[0] = img.transpose(2, 0, 1)[:, 15:239, 15:239]

    # 保存为 .bin
    outpath = Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)   # exist_ok 拼写修正

    input_data = input_data.astype(np.uint8)
    input_data.flatten().tofile(outpath / "model_input.bin")
    print("Success to save the picture to bin.")

if __name__ == '__main__':
    main()