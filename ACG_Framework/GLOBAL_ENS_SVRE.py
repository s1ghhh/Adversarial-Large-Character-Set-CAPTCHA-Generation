from ctypes.wintypes import MAX_PATH
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st
from torch.nn.functional import conv2d
from torchvision import transforms
import cv2
from PIL import Image
import random
from config_stage2 import config as conf
from torchvision.utils import save_image
import torch.nn.functional as F


def svre_inner_reco(x, x_inner, y_reco, boxes_reco_batch, model_reco):
    # y_reco = torch.cat([y_reco, y_reco], dim=0)
    grad = None
    x_orgin = x.detach().type(torch.FloatTensor).to(conf.device)
    x_orgin.requires_grad = True
    x_inner_orgin = x_inner.detach().type(torch.FloatTensor).to(conf.device)
    x_inner_orgin.requires_grad = True
    char_adv_list = []
    char_adv_inner_list = []
    for i, boxes_reco in enumerate(boxes_reco_batch):
        for j, box in enumerate(boxes_reco):
            # print((box[3]-box[1])* (box[2]- box[0]))
            cropped_img_adv = x_orgin[i, :, box[1]:box[3], box[0]:box[2]].unsqueeze(0)
            # linear | bilinear | bicubic | trilinear
            resized_img_adv = F.interpolate(cropped_img_adv, size=(128, 128), mode='bilinear', align_corners=False)
            char_adv_list.append(resized_img_adv)
            cropped_img_adv_inner = x_inner_orgin[i, :, box[1]:box[3], box[0]:box[2]].unsqueeze(0)
            resized_img_adv_inner = F.interpolate(cropped_img_adv_inner, size=(128, 128), mode='bilinear', align_corners=False)
            char_adv_inner_list.append(resized_img_adv_inner)

    char_adv = torch.cat(char_adv_list, dim=0)

    char_adv_inner = torch.cat(char_adv_inner_list, dim=0)

    loss_fn = nn.CrossEntropyLoss()
    output = model_reco(char_adv)
    loss_reco = loss_fn(output, y_reco)
    loss_reco.backward()
    grad = x_orgin.grad.data
    output_inner = model_reco(char_adv_inner)
    loss_reco_inner = loss_fn(output_inner, y_reco)
    loss_reco_inner.backward()
    grad_inner = x_inner_orgin.grad.data
    # print(torch.count_nonzero(x_orgin.grad[0]))
    # save_image(x_orgin.grad[0], f"chat_cap_grad.jpg")
    return grad, grad_inner


def save_captcha(cap, x_adv_list, box_list, output_dir):
    k = 0

    for idx in range(x_adv_list):
        box = box_list[idx]
        char_image = x_adv_list[idx].cpu()
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0

        char_image = char_image.detach().numpy()
        char_image = np.transpose(char_image, (1, 2, 0))
        char_image = char_image * 255.0
        char_image = Image.fromarray(char_image.astype(np.uint8))
        char_image = char_image.resize((int(width), int(height)))
        cap.paste(char_image, (int(x0), int(y0)))
    # cap_path = os.path.join(output_dir, self.captcha_names[k])
    cap.save('test.png')
    k += 1


def clip_by_value(x, x_min, x_max):
    x = x * (x<=x_max).type(torch.FloatTensor).to(conf.device) + x_max * (x>x_max).type(torch.FloatTensor).to(conf.device)
    x = x * (x>=x_min).type(torch.FloatTensor).to(conf.device) + x_min * (x<x_min).type(torch.FloatTensor).to(conf.device)
    return x

def grad_ensemble(model_frcnn, model_yolo, model_ssd, imgs, y, loss_yolo, loss_ssd, w_frcnn, w_yolo, w_ssd, model_name):
    grad = None
    x_orgin = imgs.detach().type(torch.FloatTensor).to(conf.device)
    x_orgin.requires_grad = True
    if model_name == 'frcnn':
        bboxes_frcnn, labels_frcnn = y[0], y[1]
        rpn_loc, rpn_cls, roi_loc, roi_cls, total = model_frcnn.forward(x_orgin, bboxes_frcnn, labels_frcnn, False)
        # total = rpn_loc + roi_loc
        total.backward()
        grad = x_orgin.grad.data

    elif model_name == 'yolov5':
        bboxes_yolo, labels_yolo = y[2], y[3]
        loss_value_all = 0
        outputs = model_yolo(x_orgin)
        for l in range(len(outputs)):
            loss_item = loss_yolo(l, outputs[l], bboxes_yolo, labels_yolo[l])
            loss_value_all += loss_item
        loss_value = loss_value_all
        loss_value.backward()
        grad = x_orgin.grad.data

    elif model_name == 'ssd':
        bboxes_ssd = y[4]
        out = model_ssd(x_orgin)
        bboxes_ssd = bboxes_ssd.to(conf.device)
        loss = loss_ssd.forward(bboxes_ssd, out)
        loss.backward()
        grad = x_orgin.grad.data

    else:
        ## frcnn loss 
        bboxes_frcnn, labels_frcnn, bboxes_yolo, labels_yolo, bboxes_ssd = y[0], y[1], y[2], y[3], y[4]
        rpn_loc, rpn_cls, roi_loc, roi_cls, frcnn_loss = model_frcnn.forward(x_orgin, bboxes_frcnn, labels_frcnn, False)
        # total = rpn_loc + roi_loc
        # frcnn_loss = rpn_loc + roi_loc
        # print(rpn_loc)
        # print(rpn_cls)
        # print(roi_loc)
        # print(roi_cls)

        ## yolo loss
        yolo_loss_all = 0
        outputs = model_yolo(x_orgin)
        for l in range(len(outputs)):
            loss_item = loss_yolo(l, outputs[l], bboxes_yolo, labels_yolo[l])
            yolo_loss_all += loss_item
        yolo_loss = yolo_loss_all

        ## sdd loss
        out = model_ssd(x_orgin)
        bboxes_ssd = bboxes_ssd.to(conf.device)
        ssd_loss = loss_ssd.forward(bboxes_ssd, out)

        # print(frcnn_loss)
        # print(yolo_loss)
        # print(ssd_loss)
        total_loss = w_yolo*yolo_loss + w_frcnn*frcnn_loss + w_ssd*ssd_loss
        total_loss.backward()
        grad = x_orgin.grad.data
    return grad


def total_loss_ensemble(model_reco, model_frcnn, model_yolo, model_ssd, imgs, y, y_reco, boxes_reco_batch, loss_yolo, loss_ssd, w_frcnn, w_yolo, w_ssd, w_reco, loss_path=None):
    grad = None
    x_orgin = imgs.detach().type(torch.FloatTensor).to(conf.device)
    x_orgin.requires_grad = True
    
    char_adv_list = []

    for i, boxes_reco in enumerate(boxes_reco_batch):
        for j, box in enumerate(boxes_reco):
            cropped_img_clean = x_orgin[i, :, box[1]:box[3], box[0]:box[2]].unsqueeze(0)
            resized_img_clean = F.interpolate(cropped_img_clean, size=(128, 128), mode='bilinear', align_corners=False)
            char_adv_list.append(resized_img_clean)

    char_adv = torch.cat(char_adv_list, dim=0)


    loss_fn = nn.CrossEntropyLoss()
    output = model_reco(char_adv)

    loss_reco = loss_fn(output, y_reco)


    if loss_path:
        import os
        with open(os.path.join('/root/autodl-tmp/dachuang/adv_detect/adv_emsable/clean_loss_check1', loss_path+'.txt'), 'a') as f:
            f.write(f'loss_reco: {loss_reco}\n')
    ## frcnn loss 
    bboxes_frcnn, labels_frcnn, bboxes_yolo, labels_yolo, bboxes_ssd = y[0], y[1], y[2], y[3], y[4]
    rpn_loc, rpn_cls, roi_loc, roi_cls, frcnn_loss = model_frcnn.forward(x_orgin, bboxes_frcnn, labels_frcnn, False)
    # total = rpn_loc + roi_loc
    # frcnn_loss = rpn_loc + roi_loc

    ## yolo loss
    yolo_loss_all = 0
    outputs = model_yolo(x_orgin)
    for l in range(len(outputs)):
        loss_item = loss_yolo(l, outputs[l], bboxes_yolo, labels_yolo[l])
        yolo_loss_all += loss_item
    yolo_loss = yolo_loss_all

    ## sdd loss
    out = model_ssd(x_orgin)
    bboxes_ssd = bboxes_ssd.to(conf.device)
    ssd_loss = loss_ssd.forward(bboxes_ssd, out)

    total_loss = w_yolo*yolo_loss + w_frcnn*frcnn_loss + w_ssd*ssd_loss + w_reco*loss_reco


    total_loss.backward()
    grad = x_orgin.grad.data
    return grad
    

def get_mask(x, bboxes_cover):
    bboxes_cover = bboxes_cover[0]
    height, width = x.size(2), x.size(3)
    n = x.size(0)
    masks = np.ones((n, 3, height, width))

    
    for i in range(n):
        if conf.AA:
            boxes = bboxes_cover[i]
            m = boxes.shape[0]
            for j in range(m):
                x1, y1, x2, y2 = int(boxes[j][0]), int(boxes[j][1]), int(boxes[j][2]), int(boxes[j][3])
                masks[i, :, y1:y2, x1:x2] = 0

        if conf.scheme == 'geetest':
            masks[i, :, int(0.9*height):height, :] = 0
    masks = torch.from_numpy(masks)
    masks = masks.to(conf.device)
    return masks


def global_ens_svre(x, images_clean, y, labels_split, boxes_reco_batch, model_reco, model_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd, max_eps, num_iter, momentum, m_svrg, bboxes_cover=None):

    char_clean_list = []
    labels_list = []
    for i, boxes_reco in enumerate(boxes_reco_batch):
        for j, box in enumerate(boxes_reco):
            cropped_img_clean = images_clean[i, :, box[1]:box[3], box[0]:box[2]].unsqueeze(0)
            resized_img_clean = F.interpolate(cropped_img_clean, size=(128, 128), mode='bilinear', align_corners=False)
            char_clean_list.append(resized_img_clean)
        labels_list.extend(labels_split[i])

    labels = torch.tensor(labels_list).to('cuda')

    if conf.gene_on_char:
        eps = max_eps / 255.0
    else:
        mask  = get_mask(x, y)
        eps = max_eps * mask / 255.0

    x_min = torch.clamp(images_clean - max_eps, 0, 1)
    x_max = torch.clamp(images_clean + max_eps, 0, 1)

    alpha = eps / (num_iter * 1.0)

    grad = torch.zeros_like(x)
    x = x.type(torch.FloatTensor).to(conf.device)


    for i_iter in range(num_iter):
        
        x_adv = x

        
        noise_ensemble = total_loss_ensemble(model_reco, model_frcnn, model_yolo, model_ssd, x_adv, y, labels, boxes_reco_batch, loss_yolo, loss_ssd, conf.w_frcnn, conf.w_yolo, conf.w_ssd, conf.w_reco)

        mask = torch.zeros_like(noise_ensemble, dtype=torch.bool)
        for k, boxes_reco in enumerate(boxes_reco_batch):
            for box in boxes_reco:
                mask[k, :, box[1]:box[3], box[0]:box[2]] = True
        masked_noise_ensemble = noise_ensemble.clone()
        masked_noise_ensemble[~mask] = 0.0

        x_inner = x
        grad_inner = torch.zeros_like(x)

        for j in range(m_svrg):

            # choose model uniformly from model pool
            model_name = random.choice(conf.model_names)
            if model_name != 'reco':
                noise_x = grad_ensemble(model_frcnn, model_yolo, model_ssd, x, y, loss_yolo, loss_ssd, conf.w_frcnn, conf.w_yolo, conf.w_ssd, model_name)
                noise_x_inner = grad_ensemble(model_frcnn, model_yolo, model_ssd, x_inner, y, loss_yolo, loss_ssd, conf.w_frcnn, conf.w_yolo, conf.w_ssd, model_name)

                noise_inner = noise_x_inner - (noise_x - noise_ensemble)
                noise_inner = noise_inner / torch.mean(torch.abs(noise_inner), (1, 2, 3), keepdims=True)

                grad_inner = momentum * grad_inner + noise_inner
            else:

                noise_char, noise_char_inner = svre_inner_reco(x, x_inner, labels, boxes_reco_batch, model_reco)

                noise_inner = noise_char_inner - (noise_char - masked_noise_ensemble)
                noise_inner = noise_inner / torch.mean(torch.abs(noise_inner), (1, 2, 3), keepdims=True)

                grad_inner = momentum * grad_inner + noise_inner

            x_inner = x_inner + alpha * torch.sign(grad_inner)
            x_inner = clip_by_value(x_inner, x_min, x_max)

        noise = grad_inner /  torch.mean(torch.abs(grad_inner), (1, 2, 3), keepdims=True)
        grad = momentum * grad + noise

        x = x + alpha * torch.sign(grad)
        x = clip_by_value(x, x_min, x_max)

    return x

