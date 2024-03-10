import os
import xml.etree.ElementTree as ET
import gc
import glob
import json
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import time
import argparse
import sys
from config_DSR import config as conf

sys.path.append(r"./Dataset_and_Model_Preparation/Model_Library_Building")
import mrcnn.nets.config
import mrcnn.nets.model
import mrcnn.nets.visualize


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme_list', default='yd,dajie,yy,baidu,dx,shumei,58,renmin,sougou,geetest', help='scheme of target captcha')
    parser.add_argument('--CAP_dirname_list', default='test_captcha', help='scheme of target captcha')
    opt = parser.parse_args()  

    return opt

def get_mrcnn_results(map_out_path, scheme, dirname):
    classes_path = f'{conf.dataset_path}/{scheme}/class_to_idx.txt'
    weight_path = f'{conf.model_weight_path}/mrcnn_models/'+scheme+'_30.h5'
    CLASS_NAMES = ['0']

    with open(classes_path, 'r', encoding='utf-8') as file:
        class_ids = file.readlines()
        n = 1
        for id in class_ids:
            char_label = id.split(' ')

            CLASS_NAMES.append(str(n))
            n += 1

    class SimpleConfig(mrcnn.nets.config.Config):

        NAME = "coco_inference"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

        NUM_CLASSES = len(CLASS_NAMES)
        DETECTION_MIN_CONFIDENCE = 0.5
        BACKBONE = 'resnet50'
    # image_ids = open(os.path.join(VOCdevkit_path, "VOC" + scheme+"/ImageSets/Main/test.txt")).read().strip().split()
    

    start = time.perf_counter()
    model = mrcnn.nets.model.MaskRCNN(mode="inference",
                                    config=SimpleConfig(),
                                    model_dir=os.getcwd())
    tf.keras.Model.load_weights(model.keras_model, weight_path, by_name=True)
    end = time.perf_counter()
    time_load = end - start



    scheme_path = os.path.join(conf.dataset_path, scheme)
    img_filepath     = os.path.join(scheme_path, dirname)
    
    image_ids = os.listdir(img_filepath)
    
    time_forward = .0

    for image_id in tqdm(image_ids):
        image_path = os.path.join(img_filepath, image_id)
        time_item = model.get_map_txt(image_id.replace('.png', ''), image_path, CLASS_NAMES, map_out_path)
        time_forward += time_item
    
    time_all = time_forward + time_load
    return time_forward, time_all


def get_ASR(scheme, MINOVERLAP, score_threhold=0.5, path='./mrcnn_results'):
    DR_PATH = os.path.join(path, 'detection-results')
    detection_results_files_list = glob.glob(DR_PATH + '/*.txt')

    total_num = 0

    match_num = 0
    for dr_file in detection_results_files_list:
        total_num += 1

        with open(dr_file) as f:
            content = f.readlines()
        lines_list = [x.strip() for x in content]

        label_gt = []
        label_dr = []
        with open(dr_file.replace('detection-results', 'ground-truth'), 'r', encoding='utf-8') as file:
            gt_lines = file.readlines()


            for line_gt in gt_lines:
                line_gt = line_gt.replace('\n', '')
                char_gt = line_gt.split(' ')
                label_gt.append(str(char_gt[0]))
            for line_dr in lines_list:
                name, confi, _, _, _, _ = line_dr.split()
                label_dr.append(str(name))
            y = set(label_gt)
            y_pred = set(label_dr)

            if y.issubset(y_pred):
                
                if scheme == 'renmin' or scheme == 'sougou':
                    if len(y) == len(y_pred):
                        match_num += 1
                else:
                        match_num += 1

    return {'total_num' : total_num, 'match_num' : match_num}

def ASR_EM(CAP_list, ADV_list):

    MINOVERLAP = 0.3
    confidence = 0.5
    score_threhold = 0.5

    for cap in CAP_list:
        for adv in ADV_list:
            print('=============== ' +cap+ '  ' + adv + ' ===============' )
            map_out_path = './DSR_results/mrcnn_results/'+cap+'_'+adv
            if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
                os.makedirs(os.path.join(map_out_path, 'ground-truth'))
            if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
                os.makedirs(os.path.join(map_out_path, 'detection-results'))


            time_all_list = []
            time_forward_list = []
            # for i in range(0,3):
            #     print(i)
            time_forward, time_all = get_mrcnn_results(map_out_path, cap, adv)
            gc.collect()
            time_all_list.append(time_all)
            time_forward_list.append(time_forward)

            get_ground_truth(cap, map_out_path)
            ASR_result = get_ASR(cap, MINOVERLAP, score_threhold=score_threhold, path=map_out_path)
            print('ASR_result: ' + str(ASR_result))

def get_ground_truth(scheme, map_out_path):
    print("Get ground truth result of " + scheme)

    class2idx = dict()
    with open(f'{conf.dataset_path}/{scheme}/class_to_idx.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    f.close()
    for l in lines:
        char_class, idx = l.strip('\n').split(' ')
        class2idx[char_class] = int(idx)

    ground_truth_path = os.path.join(conf.output_path, 'ground_truth')
    scheme_ground_truth_path = os.path.join(ground_truth_path, scheme)
    if not os.path.exists(scheme_ground_truth_path):
        os.makedirs(scheme_ground_truth_path)

    scheme_path = os.path.join(conf.dataset_path, scheme)
    scheme_label_path = os.path.join(scheme_path, 'test_captcha_label')

    json_ids = os.listdir(scheme_label_path)
    for json_filename in json_ids:
        with open(os.path.join(scheme_label_path, json_filename), 'r') as f:
            label_json = json.load(f)
        with open(os.path.join(os.path.join(map_out_path, "ground-truth/"), json_filename.replace('.json', '.txt')), "w") as save_f:
            for s in label_json['shapes']:
                char_name = s['label']
                char_idx = str(class2idx[char_name] + 1)
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
            
                save_f.write("%s %s %s %s %s\n" % (char_idx, b0, b1, b2, b3))
    print("Done.")

if __name__ == "__main__":
    args = get_config()
    scheme_list = args.scheme_list.split(',')
    CAP_dirname_list = args.CAP_dirname_list.split(',')

    ASR_EM(scheme_list , CAP_dirname_list)