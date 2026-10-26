# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import glob
import cv2
import torch
import csv
import time
import torchvision
import sys
from classes import *

classes = 1000
result_shape = [-1, classes]

parser = argparse.ArgumentParser(description='post txt data')
parser.add_argument(
    '--input',
    type=str,
    default='',
    help='the input npy')

FLAGS, unparsed = parser.parse_known_args()

def txt2npy(result_file, result_shape):
    act_list = []
    has_point = 0
    no_point = 0
    with open (result_file, 'r') as act_re:
        lines = csv.reader(act_re)
        for row in lines:
            if len(row) != 0 and "[" not in row[0] and 'SHAPE' not in row[0] and"#LAYER_NAME" not in row[0]:
                act_list.append(row[:-1])
                for x in row[:-1]:
                    if '.' in x:
                        has_point += 1
                    else:
                        no_point += 1
    if has_point > 0 and no_point > 0:
        raise ValueError('result file:{result_file} should only has float type numbers or int type numbers.')

    is_float = False
    if has_point > 0:
        is_float = True

    if is_float:
        act_list = [[float(x) for x in row] for row in act_list]
    else:
        act_list = [[int(x) for x in row] for row in act_list]
    return torch.tensor(act_list).reshape(result_shape), is_float

def get_file_name(input):
    txt_files = [f for f in os.listdir(input) if f.endswith('.txt')]
    if not txt_files:
        print(f'no .txt file in {input}')
    return txt_files

# 仅将第一条数据进行转换
if __name__ == '__main__':
    input = FLAGS.input
    txt_list = ['result-output_p.txt', 'result-output_hwc_p.txt','result-fc_p.txt', 'result-prob_p.txt', 'result-linear_1.tmp_3_p.txt', 'result-191_p.txt', 'result-output:0_p.txt']
    lable_list=["n02930766","n02963159","n03733805","n02268443","n01828970","n01910747","n02095889","n01843065","n01695060","n02037110"]
    files = get_file_name(input)
    if not files:
        # 当没有找到文件时直接退出
        exit(1)
    valid_file = None
    for file in files:
        if file in txt_list:
            valid_file = file
            break
    if valid_file is None:
        raise FileNotFoundError(f'no valid file in {txt_list}')
    output, is_float = txt2npy(os.path.join(input, valid_file), result_shape)
    output=output.numpy()
    pred = np.argmax(output, axis=1)
    np.savetxt(os.path.join(input,'pred_sim.txt'),pred,fmt='%d')
    index = np.argmax(output, axis=-1)
    print("\ninfer result:")
    for i, idx in enumerate(index):
        lable_name=get_class_lable(idx)
        if lable_name in lable_list:
            print(f"\033[3;32m<predict_{i}  idx:{idx}\tclass: {get_classes(idx)}>\033[0m")
        else:
            print(f"\033[3;31m<predict_{i}  idx:{idx}\tclass: {get_classes(idx)}>\033[0m")
