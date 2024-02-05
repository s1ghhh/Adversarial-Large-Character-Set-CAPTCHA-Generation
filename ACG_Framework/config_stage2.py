import torch

class config:
   noise_mode = 'global_ens_svre'   # ens_mi_fgsm  svre_mi_fgsm  random_noise global_ens_svre
   scheme = 'geetest'

   CAP_size_dic = {'sougou': [122,47], 'yd': [320,160], 'yy': [340,160], 'geetest':[344,384], 
         'baidu': [878,463], '58': [300, 160], 'shumei':[375, 187], 'dx':[375, 187], 
         'dajie': [362,125], 'renmin': [150,40]}

   width = CAP_size_dic[scheme][0]
   height = CAP_size_dic[scheme][1]

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   input_shape = [544, 544]
   batch_size = 4
   num_workers = 4

   pretrained = False

   dataset_path = ''
   model_weight_path = ''

   class2idx_path = f'{dataset_path}/{scheme}/class_to_idx.txt'
   input_img_path = f'{dataset_path}/{scheme}/JPEGImages_char_noised'
   clean_img_path = f'{dataset_path}/{scheme}/test_captcha'
   label_path = f'{dataset_path}/{scheme}/test_captcha_label'
   save_path = f'{dataset_path}/{scheme}/test'

   classes_path = f"{model_weight_path}/classes.txt"

   anchors_path_yolo = './Dataset_and_Model_Preparation/Model_Library_Building/yolov5/yolo_anchors.txt'
   anchors_mask_yolo = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
   phi = 'l'
   backbone_yolo = 'cspdarknet'
   model_path_yolo = f"{model_weight_path}/yolo-ep100-loss0.041-val_loss0.039.pth"  # "/root/autodl-tmp/dachuang/adv_detect/adv_em/model_data/50_0.9690_ep015.pth"

   model_path_frcnn = f"{model_weight_path}/frcnn101-ep100-loss0.667-val_loss0.617.pth" # "/root/autodl-tmp/dachuang/adv_detect/adv_em/model_data/frcnn_101_0.9658_ep070.pth" # 
   anchors_size_frcnn = [4, 16, 32]
   backbone_frcnn = 'resnet101'
   
   model_path_ssd = f"{model_weight_path}/ssd_mb2_0.9802_ep100.pth"  # "/root/autodl-tmp/dachuang/adv_detect/adv_em/model_data/ssd_vgg_0.9819_ep090.pth"
   anchors_size_ssd    = [30, 60, 111, 162, 213, 264, 315]
   backbone_ssd = 'mb2'

   model_path_IncResV2 = f"{model_weight_path}/{scheme}/IncResV2/final_model.pth"

   AA = True
   max_eps = 25.5
   num_iter = 10
   m_svrg = 24

   momentum = 1.0
   w_yolo = 1
   w_frcnn = 1 
   w_ssd = 1
   w_reco = 0.1

   model_names = ['frcnn', 'yolov5', 'ssd', 'reco']
   gene_on_char = True
   break_point = 9999