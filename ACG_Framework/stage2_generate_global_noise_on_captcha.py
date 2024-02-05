import os
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from config_stage2 import config as conf
from dataloader import Dataset, dataset_collate
from SVRE_MI_FGSM import svre_mi_fgsm, ens_mi_fgsm, random_noise
from GLOBAL_ENS_SVRE import global_ens_svre


import sys
sys.path.append(r"./Dataset_and_Model_Preparation/Model_Library_Building")
from frcnn.nets.frcnn import FasterRCNN
from frcnn.nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)

from yolov5.nets.yolo import YoloBody

from yolov5.nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)

from yolov5.utils.utils import get_anchors, get_classes
from ssd.utils.anchors import get_anchors as get_anchors_ssd
from ssd.nets.ssd import SSD300
from ssd.nets.ssd_training import MultiboxLoss

def generate_loop(gen, model_reco, model_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd, noise_mode):
    loop = tqdm(enumerate(gen), total=min(len(gen),conf.break_point))
    for iter, batch in loop:

        if iter >= conf.break_point:
            break
        images, boxes, locations, img_filenames, idxs, boxes_reco, images_clean = batch[0], batch[1], batch[2], batch[3], batch[-3], batch[-2], batch[-1]
        # print(images.shape)
        # # print(boxes.shape)
        # print(locations)
        # print(idxs)
        # print(boxes_reco)
        # print(images_clean.shape)
        images = images.to(conf.device)
        images_clean = images_clean.to(conf.device)

        if noise_mode == 'svre_mi_fgsm':
            X_adv = svre_mi_fgsm(images, boxes, model_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd, conf.max_eps, conf.num_iter, conf.momentum, conf.m_svrg)
        elif noise_mode == 'ens_mi_fgsm':
            X_adv = ens_mi_fgsm(images, boxes, model_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd, conf.max_eps, conf.num_iter, conf.momentum)
        elif noise_mode == 'random_noise':
            X_adv = random_noise(images, boxes, conf.max_eps, mode='global')
        elif noise_mode == 'global_ens_svre':
            X_adv = global_ens_svre(images, images_clean, boxes, idxs, boxes_reco, model_reco, model_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd, conf.max_eps, conf.num_iter, conf.momentum, conf.m_svrg)
        else:
            raise ValueError(f"Wrong noise_mode: {noise_mode}")
        
        n = images.size(0)
        for i in range(n):
            image, location, img_filename = X_adv[i], locations[i], img_filenames[i]
            save_path = os.path.join(conf.save_path, img_filename)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_image(image, save_path, location)

def save_image(x, path, location):
    image = x.cpu()
    img = image.detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * 255.0
    img = Image.fromarray(img.astype(np.uint8))
    img = img.crop(location)
    img = img.resize((conf.width, conf.height))
    img.save(path)



if __name__ == "__main__":

    class_names, num_classes = get_classes(conf.classes_path)

    anchors_yolo, num_anchors_yolo = get_anchors(conf.anchors_path_yolo)
    model_yolo = YoloBody(conf.anchors_mask_yolo, num_classes, conf.phi, conf.backbone_yolo, pretrained=conf.pretrained, input_shape=conf.input_shape)
    model_yolo.load_state_dict(torch.load(conf.model_path_yolo, map_location=conf.device))
    model_yolo = model_yolo.to(conf.device)
    loss_yolo    = YOLOLoss(anchors_yolo, num_classes, conf.input_shape, True, conf.anchors_mask_yolo, 0)
    model_yolo = model_yolo.eval()
    
    model_frcnn = FasterRCNN(num_classes, anchor_scales=conf.anchors_size_frcnn, backbone=conf.backbone_frcnn, pretrained=conf.pretrained)
    model_frcnn.load_state_dict(torch.load(conf.model_path_frcnn, map_location = conf.device))
    model_frcnn = model_frcnn.eval()
    model_frcnn = torch.nn.DataParallel(model_frcnn).cuda()
    optimizer = optim.Adam(model_frcnn.parameters())
    train_util_frcnn = FasterRCNNTrainer(model_frcnn, optimizer)

    anchors_ssd = get_anchors_ssd(conf.input_shape, conf.anchors_size_ssd, conf.backbone_ssd)
    model_ssd = SSD300(num_classes+1, conf.backbone_ssd, conf.pretrained)
    model_ssd.load_state_dict(torch.load(conf.model_path_ssd, map_location = conf.device))
    model_ssd = model_ssd.eval()
    model_ssd = torch.nn.DataParallel(model_ssd)
    cudnn.benchmark = True
    model_ssd = model_ssd.cuda()
    loss_ssd       = MultiboxLoss(num_classes, neg_pos_ratio=3.0)

    model_reco = torch.load(conf.model_path_IncResV2)
    model_reco = model_reco.module
    model_reco.eval()


    origin_dataset = Dataset(conf.class2idx_path, conf.input_img_path, conf.clean_img_path, conf.label_path, \
                             conf.input_shape, num_classes, anchors_yolo, conf.anchors_mask_yolo, \
                             anchors_ssd, epoch_length=300, train=False, special_aug_ratio=0)

    gen = DataLoader(origin_dataset, shuffle=False, batch_size=conf.batch_size, num_workers=1, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=None)

    generate_loop(gen, model_reco, train_util_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd, noise_mode=conf.noise_mode)
