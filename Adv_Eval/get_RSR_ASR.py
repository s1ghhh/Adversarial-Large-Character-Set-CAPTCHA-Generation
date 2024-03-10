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
import torch
import os
from PIL import Image
from torchvision import transforms

import json
import time
import gc
from get_DSR import get_detect_results
import argparse
from config_RSR_ASR import config as conf

import sys
sys.path.append(r"./Dataset_and_Model_Preparation/Model_Library_Building/attention")
from nets.attention_ocr import OCR
from utils.tokenizer import Tokenizer


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme_list', default='yd,dajie,yy,baidu,dx,shumei,58,renmin,sougou,geetest', help='scheme of target captcha')
    parser.add_argument('--attack_list', default='SR,YV,FA', help='scheme of target captcha')
    parser.add_argument('--CAP_dirname_list', default='test_captcha', help='scheme of target captcha')
    parser.add_argument('--metric_list', default='RSR,ASR', help='scheme of target captcha')
    opt = parser.parse_args()  

    return opt

def crop_chars_detect(img_path, class2idx_path, label_path, detect_result_path, mode='no_attention', ASRorRSR = 'ASR'):
    # :return char_imgs 检测模型给出的box切出的字
    # :return char_ids label里标出的字
    if mode == 'attention':
        cap_img = Image.open(img_path).convert('RGB')
    else:
        cap_img = Image.open(img_path).convert('RGB')
    w, h = cap_img.size
    # captcha_list.append(cap_img)
    # captcha_names.append(img)
    idxs = []
    with open(label_path, 'r', encoding='utf-8') as f:
        json_dic = json.load(f)
        char_names = jsonpath.jsonpath(json_dic, '$..label')
    f.close()

    class2idx = dict()
    with open(class2idx_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    f.close()
    for l in lines:
        char_class, idx = l.strip('\n').split(' ')
        class2idx[char_class] = idx
    
    # for name in char_names:
    #     char_classes.append(name)

    char_ids = []
    for char_class in char_names:
        if char_class == '１':
            char_ids.append(int(class2idx[str(1)]))
        elif char_class == '２':
            char_ids.append(int(class2idx[str(2)]))
        elif char_class == '３':
            char_ids.append(int(class2idx[str(3)]))
        elif char_class == '４':
            char_ids.append(int(class2idx[str(4)]))
        elif char_class == '５':
            char_ids.append(int(class2idx[str(5)]))
        elif char_class == '11':
            char_ids.append(int(class2idx[str(1)]))
        else:
            char_ids.append(int(class2idx[char_class]))
    char_imgs = []

    if ASRorRSR == 'RSR':
        boxes = jsonpath.jsonpath(json_dic, '$..points')
    elif ASRorRSR == 'ASR':
        boxes = []
        with open(detect_result_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for l in lines:
                l = l.replace('\n', '')
                detect_result = l.split(' ')
                boxes.append(detect_result[2:6])
        f.close()

    for box in boxes:
        # left, top, right, bottom
        if ASRorRSR == 'RSR':
            # x0, y0, x1, y1 = max(box[0][0], 1), max(box[0][1], 1), min(box[1][0], w-1), min(box[1][1], h-1)
            x0, y0, x1, y1 = box[0][0],box[0][1],box[1][0],box[1][1]
        elif ASRorRSR == 'ASR':
            x0, y0, x1, y1 = max(float(box[0]), 1), max(float(box[1]), 1), min(float(box[2]), w-1), min(float(box[3]), h-1)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        # b = x0, y0, x1, y1
        b = x0, y0, x1, y1
        char_img = cap_img.crop(b)
        # print(type(char_img))

        if mode != 'attention':
            transform = T.Compose([
                T.Resize(conf.attn_input_shape[1:]),
                T.ToTensor(), # totensor normalization
            ])
            char_imgs.append(torch.unsqueeze(transform(char_img), 0))

        else:
            char_imgs.append(char_img)

    if mode != 'attention':
        if len(char_imgs) != 0:
            char_imgs = torch.cat(char_imgs, 0)
            char_imgs = char_imgs.cuda()
        char_ids = torch.from_numpy(np.array(char_ids))

    return char_imgs, char_ids


def Recognition(scheme, CAP_dirname, mode, metric=''):
    if mode == 'SR':
        reco_model = 'Res50'
        detect_model = 'ssd_vgg'
    elif mode == 'YV':
        reco_model = 'Vgg16'
        detect_model = 'yolo_cspdarknet'
    else:
        raise ValueError(f"Error: attack pipline: {mode} does not exist. ")
    print('reco_model = ' + reco_model + '     detect_model = ' + detect_model +  '     metric = ' + metric)
    reco_model_path = f'{conf.model_weight_path}/{scheme}/{reco_model}/final_model.pth'
    captcha_img_dir = f'{conf.dataset_path}/{scheme}/{CAP_dirname}'
    detect_result_dir = os.path.join(conf.detect_result_path, f"{scheme}_{CAP_dirname}_{detect_model}")
    
    json_dir = f'{conf.dataset_path}/{scheme}/test_captcha_label'
    class_idx_path = f'{conf.dataset_path}/{scheme}/class_to_idx.txt'

    time_rec = 0
    success_num = 0
    total_num = 0

    model = torch.load(reco_model_path)
    model = model.module
    model.eval()

    for captcha_name in os.listdir(captcha_img_dir):
        total_num += 1
        number = captcha_name.replace('.png','')
        detect_result_path = os.path.join(detect_result_dir, number+'.txt')
        captcha_path = os.path.join(captcha_img_dir, number+'.png')
        json_path = os.path.join(json_dir, number+'.json')
        char_imgs, y = crop_chars_detect(captcha_path, class_idx_path, json_path, detect_result_path, ASRorRSR=metric, mode='noattention')

        start = time.perf_counter()
        out = []
        for x in char_imgs:
            x = x.unsqueeze(0)
            tmp_r = model(x)
            out.append(tmp_r.argmax(1).item())
            
        end = time.perf_counter()
        time_rec += end - start
        out = set(out)
        y = set(y.tolist())

        if y.issubset(out):

            if scheme == 'renmin' or scheme == 'sougou':
                if len(y) == len(out):
                    success_num += 1
            else:
                success_num += 1
    
    success_rate = success_num/total_num

    del model
    torch.cuda.empty_cache() 
    time.sleep(2)
    return success_rate, time_rec


def Recognition_attention(scheme, CAP_dirname, metric=''):

    captcha_img_dir = f'{conf.dataset_path}/{scheme}/{CAP_dirname}'
    detect_result_dir = os.path.join(conf.detect_result_path, f"{scheme}_{CAP_dirname}_frcnn_resnet50")
    
    json_dir = f'{conf.dataset_path}/{scheme}/test_captcha_label'
    class_idx_path = f'{conf.dataset_path}/{scheme}/class_to_idx.txt'

    time_rec = 0
    success_num = 0
    total_num = 0

    class2idx = dict()
    with open(class_idx_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    f.close()
    for l in lines:
        char_class, idx = l.strip('\n').split(' ')
        class2idx[char_class] = idx
    model, img_trans, tokenizer = attention_model(scheme)

    for captcha_name in os.listdir(captcha_img_dir):
        total_num += 1
        number = captcha_name.replace('.png','')
        detect_result_path = os.path.join(detect_result_dir, number+'.txt')
        captcha_path = os.path.join(captcha_img_dir, number+'.png')
        # print(number)
        json_path = os.path.join(json_dir, number+'.json')
        char_imgs, y = crop_chars_detect(captcha_path, class_idx_path, json_path, detect_result_path, mode='attention', ASRorRSR=metric)


        start = time.perf_counter()
        if scheme == 'sougou' or scheme == 'renmin':
            cap_img = Image.open(captcha_path).convert('RGB')
            pred = model(img_trans(cap_img).unsqueeze(0).to(device=conf.device))
            
            tmp_r = tokenizer.translate(pred.squeeze(0).argmax(1))
            out = list(tmp_r)
            for i,item in enumerate(out):
                if(len(item) == 1 and item != '?'):
                    out[i] = int(class2idx[item])
        else:
            out = []
            for i, x in enumerate(char_imgs):
                x = Image.fromarray(np.uint8(x))
  
                pred = model(img_trans(x).unsqueeze(0).to(device=conf.device))

                tmp_r = tokenizer.translate(pred.squeeze(0).argmax(1))

                if(len(tmp_r) == 1 and tmp_r != '?'):
                    out.append(int(class2idx[tmp_r]))
            
        end = time.perf_counter()
        time_rec += end - start
        out = set(out)
        y = set(y)

        if y.issubset(out):
            if scheme == 'renmin' or scheme == 'sougou':
                if len(y) == len(out):
                    success_num += 1
            else:
                    success_num += 1

    success_rate = success_num/total_num

    del model
    torch.cuda.empty_cache() 
    time.sleep(2)
    return success_rate, time_rec


def attention_model(scheme):
    json_file=f"{conf.model_weight_path}/{scheme}/attention/config_infer.json"

    model_path = f"{conf.model_weight_path}/{scheme}/attention/best_model.pth"
    f = open(json_file, encoding='utf-8')
    data = json.load(f)

    cnn_option = data["cnn_option"]
    cnn_backbone = data["cnn_backbone_model"][str(cnn_option)]  # list containing model, model_weight

    with open(f'{conf.dataset_path}/{scheme}/class_to_idx.txt', 'r', encoding='utf-8') as f:
        cls_lines = f.readlines()
    cls_list = []
    for l in cls_lines:
        char_class, _ = l.strip('\n').split(' ')
        cls_list.append(char_class)

    tokenizer = Tokenizer(cls_list)

    if scheme == 'sougou':
        model = OCR(122, 47, 512, tokenizer.n_token,
                    4 + 1, tokenizer.SOS_token, tokenizer.EOS_token,cnn_backbone).to(device=data["device"])

        img_trans = transforms.Compose([
                    transforms.Resize((47, 122)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
                ])
    elif scheme == 'renmin':
        model = OCR(150, 40, 512, tokenizer.n_token,
                    2 + 1, tokenizer.SOS_token, tokenizer.EOS_token,cnn_backbone).to(device=data["device"])

        img_trans = transforms.Compose([
                    transforms.Resize((40, 150)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
                ])
    else:
        model = OCR(data["img_width"], data["img_height"], data["nh"], tokenizer.n_token, data["max_len"] + 1, tokenizer.SOS_token, tokenizer.EOS_token,cnn_backbone).to(device=data["device"])
        img_trans = transforms.Compose([
                transforms.Resize((data["img_height"], data["img_width"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
            ])
    

    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt, strict = False)
    model = model.eval()

    return model, img_trans, tokenizer

def exp_RSR_ASR(scheme_list, CAP_dirname_list, attack_list, metric_list):
    for attack in attack_list:
        print("======================== " + attack + " ========================")

        print("Get all detect results.")
        # get_detect_results(CAP_list, ADV_list, attack)
        print('Done.')
        for metric in metric_list:
            print("------------------------ " + metric + " ------------------------")
            for scheme in scheme_list:
                print("============ " + scheme + " ============")
                for CAP_dirname in CAP_dirname_list:
                    print("------------ " + CAP_dirname + " ------------")
                    success_rec_list = []
                    time_rec_list = []
                        
                    if attack == 'SR' or attack == 'YV':
                        success_rate, time_rec = Recognition(scheme, CAP_dirname, mode=attack, metric=metric)
                    elif attack == 'FA':
                        success_rate, time_rec = Recognition_attention(scheme, CAP_dirname, metric=metric)
                    success_rec_list.append(success_rate)
                    time_rec_list.append(time_rec)
                    gc.collect()
                    print('ASR:' + str(sum(success_rec_list)/len(success_rec_list)))
                    print(success_rec_list)
                    print('total time forward:' + str(sum(time_rec_list)/len(time_rec_list)))
                    print(time_rec_list)






if __name__ == "__main__":

    args = get_config()
    scheme_list = args.scheme_list.split(',')
    attack_list = args.attack_list.split(',')
    CAP_dirname_list = args.CAP_dirname_list.split(',')
    metric_list = args.metric_list.split(',')

    exp_RSR_ASR(scheme_list, CAP_dirname_list, attack_list, metric_list)



    
