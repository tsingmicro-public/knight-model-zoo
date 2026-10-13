import torch
import numpy as np
import cv2
import sys
import os
import glob
import csv
import argparse
import contextlib
import glob
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile

import cv2
#import IPython
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml




# 图像预处理config
default_model_width = 640
default_model_height = 640
# config_nc = 4
config_nc = 80
config_c = 3*(5+config_nc)

m0_shape = [1, config_c, int(default_model_height/8), int(default_model_width/8)]
m1_shape = [1, config_c, int(default_model_height/16), int(default_model_width/16)]
m2_shape = [1, config_c, int(default_model_height/32), int(default_model_width/32)]




DEBUG = 0                   # 1-开启debug, 0-关闭debug
USE_OPENCV2_NMS = 0         # 1-后处理中nms使用cv2, 0-后处理使用源码nms
conf_thres = 0.25           # nms confidence门限
iou_thres=0.45              # nms IOU门限
DEBUG_USE_TENSOR_INPUT = 0  # 1-直接读model_input.bin作为推理输入, 0-从原始图片开始处理
USE_SENIOR_CHANGE = 0

config_yolov5s={
    'nc': config_nc,                # number of classes
    'anchors':[
        [10,13, 16,30, 33,23],      # P3/8
        [30,61, 62,45, 59,119],     # P4/16
        [116,90, 156,198, 373,326]  # P5/32
    ],
    'data': {
        'names': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'},
        'nc': config_nc
    }
}
# 后处理配置
COLOR = [(56, 243, 185), (115, 75, 60), (144, 104, 184), (169, 44, 185), (120, 114, 14), (175, 148, 64), (185, 128, 123), (128, 20, 242), (46, 142, 155), (235, 190, 146), (159, 153, 22), (200, 147, 12), (62, 204, 55), (195, 222, 102), (102, 157, 69), (190, 150, 129), (5, 196, 168), (45, 252, 198), (189, 188, 31), (75, 94, 195), (240, 207, 111), (216, 47, 134), (75, 213, 43), (130, 73, 139), (14, 151, 212), (81, 138, 111), (76, 119, 95), (98, 68, 64), (39, 172, 252), (4, 93, 90)]
COLOR_LENS = len(COLOR)

# out files list
RST_ONNX_SIM=['result-_model.24_m.0_Conv_output_0_p.txt','result-_model.24_m.1_Conv_output_0_p.txt','result-_model.24_m.2_Conv_output_0_p.txt']

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output



########################## preprocess ##########################################
class Dataloader:
    def __init__(self, input_path, batch_size=1, input_dtype='float', imgsz=[640], shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.input_dtype = input_dtype
        self.imgsz = (imgsz[0], imgsz[0]) if len(imgsz) == 1 else tuple(imgsz[:2])
        self.drop_last = drop_last
        self.input_data, self.targets, self.shapes = self.load_data(input_path)
        self.data_lens = len(self.input_data)
        self.total_iter_num = self.data_lens // batch_size if drop_last else (self.data_lens + batch_size - 1) // batch_size
        self.index = [i for i in range(self.data_lens)]
        if shuffle:
            random.shuffle(self.index)
        # to ndarray
        self.input_data = np.array(self.input_data)
        self.index = np.array(self.index)
    def __iter__(self):
        self.iter = 0
        return self
    def __next__(self):
        if self.iter == self.total_iter_num:
            raise StopIteration
        start = self.iter * self.batch_size
        idx = self.index[start: start + self.batch_size]
        input_data = self.input_data[idx]
        C, H, W = input_data.shape[-3:]
        input_data = input_data.reshape(-1, C, H, W)
        targets, shapes = [], []
        for i in idx:
            targets.append(self.targets[i])
            shapes.append(self.shapes[i])
        self.iter += 1
        return input_data, targets, shapes
    ###### get img data from a list of jpg path ###############
    def get_images(self, test_image_list, save_path=None):
        img_list,shapes = [],[]
        for i in range(len(test_image_list)):
            data,shape = self.load_image(test_image_list[i], save_path)
            img_list.append(data)
            shapes.append(shape)
        return img_list,shapes
    def load_data(self, input_path):
        ########## get jpg list and label list ###################################
        image_list = glob.glob(input_path + '/*.jpg')
        label_list = [path.replace('images', 'labels').replace('jpg', 'txt') for path in image_list]
        test_image_list, shapes = self.get_images(image_list)
        label_classes=[]
        for i, label_path in enumerate(label_list):
            label_classes.append([])
            with open(label_path) as act_re:
                rows = act_re.readlines()
                for i, row in enumerate(rows):
                    row = list(map(float, row.strip().split(' ')))
                    rows[i] = row
                label_classes[-1] = np.array(rows)
        return test_image_list, label_classes, shapes
    def load_image(self,img_path, save_path=None):
        ori_img = cv2.imread(img_path)      # BGR HWC
        shape=(ori_img.shape[0],ori_img.shape[1])
        assert ori_img is not None, f'Image Not Found {img_path}'
        if USE_SENIOR_CHANGE:
            h0, w0 = ori_img.shape[:2]  # orig hw
            r2 = 640 / max(h0, w0)  # resize image to img_size
            if r2 != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r2 < 1 else cv2.INTER_LINEAR
                ori_img = cv2.resize(ori_img, (int(w0 * r2), int(h0 * r2)), interpolation=interp)
            r1 = min(h0 / w0, w0 / h0)
            reshape0 = np.ceil(r1 * 640 / 32 +0.5) * 32
            reshape1 = 672
            reshape_h0 = reshape1
            reshape_w0 = reshape0
            if h0 < w0:
                reshape_h0 = reshape0
                reshape_w0 = reshape1
            img = letterbox(ori_img, new_shape=(reshape_h0, reshape_w0), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=False, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB hwc to chw
            img = np.ascontiguousarray(img)
        img = letterbox(ori_img, new_shape=(default_model_height, default_model_width), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32)[0]
        #save resized image
        if save_path:
            name = os.path.basename(os.path.abspath(img_path))
            cv2.imwrite(os.path.join(save_path, name), img)
            cv2.imwrite(os.path.join(save_path, 'yolov5.jpeg'), img)
            if DEBUG:
                print(f'[Info]load_image :{img_path} shape:{ori_img.shape}')
                print(f'[Info]load_image :{img_path} after letterbox shape:{img.shape}')
        img = img.transpose((2, 0, 1))[::-1]     # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        channel, ht, wth = img.shape[:]
        data = img.reshape(1, channel, ht, wth)
        if self.input_dtype=='float':
            data = data.astype(np.float32) / 255.0
        if save_path:
            data.flatten().tofile(os.path.join(save_path, 'model_input.bin'))
        return data, shape
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]    # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: 
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r            # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:                # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:         # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    dw /= 2                 # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


########################## postprocess##########################################
class Detect_process():
    def __init__(self, anchors, nc=1, nl=3, ch=3, na=3, inplace=True):
        self.nc = nc                            # number of classes
        self.no = nc + 5                        # number of outputs per anchor
        self.nl = len(anchors)                  # number of detection layers
        self.na = len(anchors[0]) // 2          # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl                       # init anchor grid
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)   # shape(nl,na,2)
        self.inplace = inplace                  # use in-place ops (e.g. slice assignment)
        self.stride = torch.Tensor([8, 16, 32])
        self.anchors /= self.stride.view(-1, 1, 1)

    def post_process(self, x):
        z = []  # inference output
        for i in range(len(x)):
            bs, _, ny, nx = x[i].shape
            # x(bs,255,20,20) to x(bs,3,85,20,20) and to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i].sigmoid()
            if self.inplace:
                y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]       # wh
            z.append(y.view(bs, -1, self.no))
        return torch.cat(z, 1)

    @staticmethod
    def _make_grid_add(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)],indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t=self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2                                   # grid shape
        yv, xv = torch.meshgrid(torch.arange(ny, device=d,dtype=t), torch.arange(nx, device=d,dtype=t),indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape).float() - 0.5     # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape).float()
        return grid, anchor_grid

def non_max_suppression_cv2(boxes, conf_thres=0.25, iou_thres=0.45):
    # 将边界框和置信度转换为OpenCV 所需的格式(cx,cy,h,w)-->(x1,y1,x2,y2)
    boxes_ltrd = []
    for i in boxes:
        boxes_ltrd.append([int(i[0] - i[2]/2.0), int(i[1] - i[3]/2.0), int( i[2]), int(i[3])])
    confidences = boxes[:,4]
    size_conf=len(confidences)
    cnt_box=0
    for i in range(size_conf):
        if confidences[i]>conf_thres:
            boxes_class=boxes[i,5:]
            max_score=torch.max(boxes_class)
            confidences[i]=confidences[i]*max_score
            cnt_box=cnt_box+1
    # 应用非最大抑制
    indices = cv2.dnn.NMSBoxes(np.array(boxes_ltrd), np.array(confidences), conf_thres, iou_thres)
    nms_results = []
    for i in indices:
        if DEBUG:
            print(f'--------cv2 nms:{i}, {boxes_ltrd[i]}, {confidences[i]}, {np.argmax(boxes[i][5:])}')
        box_result = boxes_ltrd[i]
        box_result.append(confidences[i])
        box_result.append(np.argmax(boxes[i][5:]))
        nms_results.append(box_result)
    # 返回保留的边界框索引列表
    return [torch.tensor(nms_results)],cnt_box

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def scale_boxes(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old 640 / new 424
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    #different to opensource ,same to RT lib code
    clip_boxes(boxes, img0_shape)
    boxes[..., :4] /= gain
    return boxes
def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
def draw_bboxes(image, im_size, bboxes):
    for bbox in bboxes:
        for (x1, y1, x2, y2, conf, class_id) in bbox:

            name = config_yolov5s['data']['names']
            name = name[int(class_id)]
            color = COLOR[int(class_id) % COLOR_LENS]   # (0, 255, 0)
            c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
            text = name + ' ' + str(format(conf.item(),'.2f'))
            cv2.rectangle(image, c1, c2, color, 3)
            cv2.putText(image, text, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 3)
    return image


def save_pred_to_file(pred, savedir,pic_order):
    file_name=f'infer_rst_pic_{pic_order}.txt'
    txt_pred = np.array(pred[0],dtype=np.float32)
    np.savetxt(os.path.join(savedir,file_name),txt_pred,fmt='%f')
    if pic_order==0:
        print(f'Save infer result to {savedir}/infer_rst_pic_*.txt')


def txt2npy(result_file, result_shape):
    if DEBUG:
        print(f'result_file:{result_file} result_shape:{result_shape}')
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

def detection_post_process(savedir,in_list,output_dtype=None,is_dir=bool,test_data=None,proc_mode="yolov5_org",proc_step=None):
    if type(in_list) == str:

        m0, is_float = txt2npy(os.path.join(in_list, RST_ONNX_SIM[0]), m0_shape)

        m1, is_float = txt2npy(os.path.join(in_list, RST_ONNX_SIM[1]), m1_shape)

        m2, is_float = txt2npy(os.path.join(in_list, RST_ONNX_SIM[2]), m2_shape)

    else:
        m0=torch.from_numpy(in_list[0])
        m1=torch.from_numpy(in_list[1])
        m2=torch.from_numpy(in_list[2])
    if DEBUG:
        print(f'detection_post_process proc_mode:{proc_mode},proc_step:{proc_step}')


    ################# do Coordinate restoration(out cx,cy,h,w,score,c0prob,c1prob...)##########################
    x_list=[]
    x_list.append(m0)
    x_list.append(m1)
    x_list.append(m2)
    detect_process = Detect_process(config_yolov5s['anchors'], nc=config_yolov5s['nc'])
    pred = detect_process.post_process(x_list)
    ################# do NMS ##########################
    if USE_OPENCV2_NMS:
        pred,cnt_box_beforeNms = non_max_suppression_cv2(pred[0], conf_thres, iou_thres)

        if not is_dir:
            print(f'Before nms, The boxes number is {cnt_box_beforeNms}')
        cnt_box=pred[0].shape[0]
        for i in range(cnt_box):
            rect_x2=int(pred[0][i][0])+int(pred[0][i][2])
            rect_y2=int(pred[0][i][1])+int(pred[0][i][3])
            pred[0][i][2]=rect_x2
            pred[0][i][3]=rect_y2

    else:
        pred = non_max_suppression(pred, conf_thres, iou_thres)

    if not is_dir:
        cnt_box=pred[0].shape[0]
        print(f'After nms, The boxes number is {cnt_box}')
        im_dst = cv2.imread(test_data)
        if  USE_OPENCV2_NMS:
            img1_shape=[default_model_height, default_model_width]
            img0_shape=im_dst.shape[:2]
            for i in range(cnt_box):
                conf=pred[0][i][4]
                conf=conf.numpy().astype(float)
                conf=format(conf,'.2f')
                score=int(pred[0][i][5])
            ############ REMAP rect ################################################
            scale_boxes(img1_shape, pred[0], img0_shape)
            for i in range(cnt_box):
                conf=pred[0][i][4]
                conf=conf.numpy().astype(float)
                conf=format(conf,'.2f')
                score=int(pred[0][i][5])
                print(f'draw box {i}: x1:{int(pred[0][i][0])}, y1:{int(pred[0][i][1])}, w:{int(pred[0][i][2]-pred[0][i][0])}, h:{int(pred[0][i][3]-pred[0][i][1])}, conf:{conf}, class_id:{score}')

        ############################################################
        if  not USE_OPENCV2_NMS:
            img1_shape=[default_model_height, default_model_width]
            img0_shape=im_dst.shape[:2]

            for i in range(cnt_box):
                conf=pred[0][i][4]
                conf=conf.numpy().astype(float)
                conf=format(conf,'.2f')
                score=int(pred[0][i][5])

            scale_boxes(img1_shape, pred[0], img0_shape)
            print("----------------------------------------")
            for i in range(cnt_box):
                conf=pred[0][i][4]
                conf=conf.numpy().astype(float)
                conf=format(conf,'.2f')
                score=int(pred[0][i][5])
                print(f'draw box {i}: x1:{int(pred[0][i][0])}, y1:{int(pred[0][i][1])}, w:{int(pred[0][i][2]-pred[0][i][0])}, h:{int(pred[0][i][3]-pred[0][i][1])}, conf:{conf}, class_id:{score}')

        print("----------------------------------------")
    ############# draw rect and save img #############
        if output_dtype != 'float':
            # save rected image to outDir
            im = cv2.imread(test_data)
            im_size = [im.shape[0], im.shape[1]]
            im_s = draw_bboxes(im, im_size, pred)
            cv2.imwrite(os.path.join(savedir, 'yolov5_result.jpg'), im_s)
            print(f'[Info]save processed img {savedir}/yolov5_result.jpg')
            txt_pred = np.array(pred[0],dtype=np.float32)
            np.savetxt(os.path.join(savedir,'result.txt'),txt_pred,fmt='%f')
    return pred

class Identify(torch.nn.Module):
    def __init__(self):
        super(Identify, self).__init__()
    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
        return x

def write_numpy_to_file(data, fileName, is_float=False):
    N, C, H, W = data.shape
    file = open(fileName, 'w+')
    head = 'SHAPE:(N:{0}, C:{1}, H:{2}, W:{3})\n'.format(N, C, H, W)
    file.write(head)
    for i in range(C):
        str_C = ""
        file.write('C[{0}]:\n'.format(i))
        for j in range(H):
            for k in range(W):
                if is_float:
                    str_C = str_C + '{0},'.format(float(data[0][i][j][k]))
                else:
                    str_C = str_C + '{0},'.format(int(data[0][i][j][k]))
            str_C = str_C + '\n'
        file.write(str_C)
    file.close()

def print_red_strings(info):
    print('\033[31m{}\033[0m'.format(info))

def print_green_strings(info):
    print('\033[32m{}\033[0m'.format(info))

def tensor2variable(tensor):
    variable = tensor.to_variable(dtype=np.float32, shape=tensor.shape)
    return variable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make input data scripts')
    parser.add_argument(
        '--input',
        type=str,
        default='',
        required=True,
        help='input image')
    parser.add_argument(
        '--outpath',
        type=str,
        default='/data/examples/yolov5s/output/pre',
        required=True,
        help='the out image.')
    parser.add_argument(
        '--post_processing',
        action='store_true',
        default=False,
        help='post_processing')
    parser.add_argument(
        '--pre_processing',
        action='store_true',
        default=False,
        help='pre_processing')
    FLAGS, unparsed = parser.parse_known_args()

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='image')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.pre_processing:
        dataloader = Dataloader(FLAGS.input, 1,'uint8')
        if not os.path.exists(FLAGS.outpath):
            os.makedirs(FLAGS.outpath)
        data,_ = dataloader.load_image(FLAGS.input,  FLAGS.outpath)

    if FLAGS.post_processing:
        detection_post_process(FLAGS.outpath,FLAGS.input,output_dtype=None,is_dir=False,test_data=FLAGS.data)