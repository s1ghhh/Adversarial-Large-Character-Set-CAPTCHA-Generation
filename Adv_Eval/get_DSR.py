import torchvision.transforms as T
import torch
import torch.nn as nn
import os
from PIL import Image
import json
import jsonpath
import numpy as np
import glob
import xml.etree.ElementTree as ET
import time
import gc
import argparse

from config_DSR import config as conf
import sys

sys.path.append(r"./Dataset_and_Model_Preparation/Model_Library_Building")

from frcnn.utils.utils import (cvtColor, resize_image, preprocess_input, get_anchors_yolo)

from frcnn.nets.frcnn import FasterRCNN
from frcnn.utils.utils_bbox import DecodeBox

from yolov5.nets.yolo import YoloBody
from yolov5.utils.utils_bbox_yolo import DecodeBox_yolo

from ssd.nets.ssd import SSD300
from ssd.utils.utils_bbox import BBoxUtility
from ssd.utils.anchors import get_anchors

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme_list', default='yd,dajie,yy,baidu,dx,shumei,58,renmin,sougou,geetest', help='scheme of target captcha')
    parser.add_argument('--attack_list', default='SR,YV,FA', help='scheme of target captcha')
    parser.add_argument('--CAP_dirname_list', default='test_captcha', help='scheme of target captcha')
    opt = parser.parse_args()  

    return opt


def get_detect_results(schemes, dirnames, attack):
    global model_name, backbone, model_path
    if attack == 'SR':
        model_name = 'ssd'
        backbone = 'vgg'
        model_path = conf.model_path_ssd   
    elif attack == 'FA':
        model_name = 'frcnn'
        backbone = 'resnet50'
        model_path = conf.model_path_frcnn
    elif attack == 'YV':
        model_name = 'yolo'
        backbone = 'cspdarknet'
        model_path = conf.model_path_yolo
    else:
        raise ValueError(f"Error: attack pipline: {attack} does not exist. ")

    if model_name == 'frcnn':

        model = FasterRCNN(conf.num_classes, "predict", anchor_scales = conf.anchors_size_frcnn, backbone = backbone)
        model.load_state_dict(torch.load(model_path, map_location=conf.device))
        std    = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(conf.num_classes + 1)[None]
        std    = std.cuda()
        bbox_util  = DecodeBox(std, conf.num_classes)

    elif model_name == 'yolo':

        anchors, _      = get_anchors_yolo(conf.anchors_path_yolo)
        model = YoloBody(conf.anchors_mask_yolo, conf.num_classes, conf.phi, backbone = backbone, input_shape = conf.input_shape)
        model.load_state_dict(torch.load(model_path, map_location=conf.device))
        bbox_util = DecodeBox_yolo(anchors, conf.num_classes, (conf.input_shape[0], conf.input_shape[1]), conf.anchors_mask_yolo)

    elif model_name == 'ssd':

        model = SSD300(conf.num_classes + 1, backbone)
        model.load_state_dict(torch.load(model_path, map_location=conf.device))
        bbox_util = BBoxUtility(conf.num_classes + 1)
    

    model = model.eval()
    model = nn.DataParallel(model)
    model = model.cuda()

    for scheme in schemes:
        for dirname in dirnames:
            time_list = []
            scheme_path = os.path.join(conf.dataset_path, scheme)
            img_filepath     = os.path.join(scheme_path, dirname)
            img_filenames       = os.listdir(img_filepath)

            time_sum = 0.0
            for i, img_filename in enumerate(img_filenames):
                img_path  = os.path.join(img_filepath, img_filename)
                t = get_detect_result(model, scheme, dirname, bbox_util, img_filename, img_path)

                if t == None:
                    t=0.0
                time_sum+=t
            gc.collect()
            time_list.append(time_sum)
            print("Get results of " + scheme + '/' + dirname)
            print(time_list)
            print(sum(time_list)/len(time_list))


def get_detect_result(model, scheme, dirname, bbox_util, img_filename, img_path):
    image = Image.open(img_path)
    results_save_path = os.path.join(conf.output_path, f"{scheme}_{dirname}_{model_name}_{backbone}")
    if not os.path.exists(results_save_path):
        os.mkdir(results_save_path)
    f = open(os.path.join(results_save_path, img_filename.replace('.png', '.txt')), "w", encoding='utf-8')

    image_shape = np.array(np.shape(image)[0:2])
    image       = cvtColor(image)
    image_data  = resize_image(image, (conf.input_shape[1], conf.input_shape[0]), conf.letterbox_image)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    images = torch.from_numpy(image_data)
    images = images.cuda()

    with torch.no_grad():
        start = time.perf_counter()
        if model_name == 'frcnn':
            roi_cls_locs, roi_scores, rois, _ = model(images)
            results = bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, conf.input_shape, conf.letterbox_image, nms_iou = conf.nms_iou, confidence = conf.confidence)

            if len(results[0]) <= 0:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        elif model_name == 'yolo':
            outputs = model(images)
            outputs = bbox_util.decode_box(outputs)

            results = bbox_util.non_max_suppression(torch.cat(outputs, 1), conf.num_classes, conf.input_shape, 
                        image_shape, conf.letterbox_image, conf_thres = conf.confidence, nms_thres = conf.nms_iou)
            if results[0] is None: 
                return 
            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        elif model_name == 'ssd':
            anchors_ssd = torch.from_numpy(get_anchors(conf.input_shape, conf.anchors_size_ssd, backbone)).type(torch.FloatTensor)
            anchors_ssd = anchors_ssd.cuda()
            outputs = model(images)
            results = bbox_util.decode_box(outputs, anchors_ssd, image_shape, conf.input_shape, conf.letterbox_image,
                                                    nms_iou = conf.nms_iou, confidence = conf.confidence)

            if len(results[0]) <= 0:
                return 
            top_label   = np.array(results[0][:, 4], dtype = 'int32')
            top_conf    = results[0][:, 5]
            top_boxes   = results[0][:, :4]
        end = time.perf_counter()

    for i, c in list(enumerate(top_label)):
        predicted_class = conf.class_names[int(c)]
        box             = top_boxes[i]
        score           = str(top_conf[i])

        top, left, bottom, right = box
        if predicted_class not in conf.class_names:
            continue

        f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
    f.close()

    return end - start


def get_ground_truth(scheme):
    print("Get ground truth result of " + scheme)
    ground_truth_path = os.path.join(conf.output_path, 'ground_truth')
    scheme_ground_truth_path = os.path.join(ground_truth_path, 'scheme')
    if not os.path.exists(scheme_ground_truth_path):
        os.mkdir(scheme_ground_truth_path)

    scheme_path = os.path.join(conf.dataset_path, scheme)
    scheme_label_path = os.path.join(scheme_path, 'test_captcha_label')

    json_ids = os.listdir(scheme_label_path)
    for json_filename in json_ids:
        with open(os.path.join(scheme_label_path, json_filename), 'r') as f:
            label_json = json.load(f)

        for s in label_json['shapes']:
            # left top right bottom
            b0, b1, b2, b3 = int(s['points'][0][0]), int(s['points'][0][1]), int(s['points'][1][0]), int(s['points'][1][1])
            if b3 < b1:
                tmp = b1
                b1 = b3
                b3 = tmp
            if b2 < b0:
                tmp = b0
                b0 = b2
                b2 = tmp
            
            with open(os.path.join(scheme_ground_truth_path, json_filename.replace('.json', '.txt')), "w") as save_f:
                save_f.write("%s %s %s %s %s\n" % ('char', b0, b1, b2, b3))
    print("Done.")


def get_DSR(scheme, dirname, MINOVERLAP=0.5, score_threhold=0.5, path='/root/autodl-tmp/dachuang/数据集'):

    ground_truth_path = os.path.join(conf.output_path, 'ground_truth')
    scheme_ground_truth_path = os.path.join(ground_truth_path, 'scheme')
    detect_results_path = os.path.join(conf.output_path, f"{scheme}_{dirname}_{model_name}_{backbone}")

    detect_results_filename_list = os.listdir(detect_results_path)

    total_num = 0

    success_num = 0

    for dr_filename in detect_results_filename_list:

        total_num += 1
        with open(os.path.join(detect_results_path, dr_filename)) as f:
            content = f.readlines()
            lines_list = [x.strip() for x in content]


        with open(os.path.join(scheme_ground_truth_path, dr_filename), 'r', encoding='utf-8') as file:
            gt_lines = file.readlines()
            
            num_char_no_be_detected = len(gt_lines)

            for line_gt in gt_lines:
                line_gt = line_gt.replace('\n', '')

                char_gt = line_gt.split(' ')
                char_gt = [float(char_gt[1]), float(char_gt[2]), float(char_gt[3]), float(char_gt[4])]
                ovmax = -1

                for line in lines_list:
                    class_name, confidence, left, top, right, bottom = line.split()
                    left = float(left)
                    top = float(top)
                    right = float(right)
                    bottom = float(bottom)

                    if float(confidence) > float(score_threhold):
                        bi = [max(float(left), char_gt[0]), max(float(top), char_gt[1]), min(float(right), char_gt[2]), min(float(bottom), char_gt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (right - left + 1) * (bottom - top + 1) + (char_gt[2] - char_gt[0]
                                                                              + 1) * (char_gt[3] - char_gt[1] + 1) - iw * ih
                            ov = iw * ih / ua

                            if ov > ovmax:
                                ovmax = ov

                if ovmax > MINOVERLAP:
                    num_char_no_be_detected -= 1

            if num_char_no_be_detected == 0:
                success_num += 1

    DSR = success_num/total_num

    return DSR
    

def exp_DSR(scheme_list, CAP_dirname_list, attack_list):
    for attack in attack_list:
        print("+++++++++++ " + attack + " +++++++++++")
        print("Get all detect results.")
        get_detect_results(scheme_list, CAP_dirname_list, attack)

        print("Get DSR.")
        cap_list = []
        for scheme in scheme_list:
            get_ground_truth(scheme)
            print("============ " + scheme + " ============")
            
            for dirname in CAP_dirname_list:
                DSR = get_DSR(scheme, dirname)
                cap_list.append(DSR)
                print("------------ " + dirname + " ------------")
                print('DSR : %.3f'%(DSR))
        print(cap_list)

if __name__ == "__main__":
    args = get_config()
    scheme_list = args.scheme_list.split(',')
    attack_list = args.attack_list.split(',')
    CAP_dirname_list = args.CAP_dirname_list.split(',')

    exp_DSR(scheme_list , CAP_dirname_list, attack_list)
