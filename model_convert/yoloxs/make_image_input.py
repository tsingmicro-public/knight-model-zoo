# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import cv2
#from yolox.data.data_augment import preproc as preprocess

IMAGE_SIZE = (640,640)
DEBUG = 0

# 支持的图片扩展名
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

parser = argparse.ArgumentParser(description='make input data scripts')
parser.add_argument('--input', type=str, default='./',
                    help='path of image file OR images dir')
parser.add_argument('--outpath', type=str, default='./',
                    help='the path of output bin dir')
parser.add_argument('--proc_mode', type=str, default='onnx',
                    help='the proc mode as onnx2onnx')
parser.add_argument('--img-size', type=int, default=(640, 640),
                    help='inference size (pixels)')

FLAGS, unparsed = parser.parse_known_args()

def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r



def preprocess_image(img_path):
    """读取并预处理单张图片，失败返回 (None, None)"""
    origin_img = cv2.imread(img_path)
    if origin_img is None:
        print(f"[WARN] skip unreadable file: {img_path}")
        return None, None
    img, ratio = preprocess(origin_img, IMAGE_SIZE)
    img = img.astype(np.uint8)
    img = img[None, :, :, :]
    return img, ratio


def process_single_file(img_path, out_dir):
    """单张图片模式：输出固定名 model_input.bin"""
    img, ratio = preprocess_image(img_path)
    if img is None:
        return
    bin_path = os.path.join(out_dir, "model_input.bin")
    img.flatten().tofile(bin_path)
    print(f"save model_input.bin to {bin_path} (ratio={ratio})")


def process_directory(input_dir, out_dir):
    """批量目录模式：按原文件名逐张输出 <name>.bin"""
    img_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(IMG_EXTS)
    ])

    if len(img_files) == 0:
        print(f"[WARN] no image files found in {input_dir}")
        return

    for fname in img_files:
        img_path = os.path.join(input_dir, fname)
        img, ratio = preprocess_image(img_path)
        if img is None:
            continue

        bin_name = os.path.splitext(fname)[0] + ".bin"
        bin_path = os.path.join(out_dir, bin_name)
        img.flatten().tofile(bin_path)
        print(f"save {bin_name} to {bin_path} (ratio={ratio})")


if __name__ == '__main__':
    input_path = FLAGS.input
    out_dir = FLAGS.outpath

    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(input_path):
        # 目录 → 批量模式（等价原 make_input.py）
        process_directory(input_path, out_dir)
    elif os.path.isfile(input_path):
        # 单文件 → 单图模式（等价原 make_image_input.py）
        process_single_file(input_path, out_dir)
    else:
        raise ValueError(f"input path not found: {input_path}")