import torch

class config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    letterbox_image = True
    input_shape = [544, 544]

    # scheme = 'shumei'
    
    # frcnn-50 , frcnn-101 , frcnn-vgg , yolo , ssd-mb2 , ssd-vgg
    type_model = 'frcnn-vgg'
    # CLEAN , ADV
    num_classes = 1
    confidence = 0.5
    nms_iou = 0.3
    class_names = ['char']
    # reco 128*128        attention 120*120
    reco_input_shape = [3,128,128]

    VOCdevkit_path = '/root/autodl-tmp/dachuang/数据集/test_detect/VOC'
    out_path = '/root/autodl-tmp/dachuang/数据集/detect_results/DR_'
    out_path_ground_truth = '/root/autodl-tmp/dachuang/数据集/ground_truth/'

    
    if type_model == 'frcnn-50':
        model_name = 'frcnn'
        backbone = 'resnet50'
        model_path = '/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/frcnn-50-ep100-loss0.689-val_loss0.633.pth'
    elif type_model == 'frcnn-101':
        model_name = 'frcnn'
        backbone = 'resnet101'
        model_path = '/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/frcnn101-ep100-loss0.667-val_loss0.617.pth'
    elif type_model == 'frcnn-vgg':
        model_name = 'frcnn'
        backbone = 'vgg'
        model_path = '/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/vgg-ep100-loss0.588-val_loss0.563.pth'
    elif type_model == 'yolo':
        model_name = 'yolo'
        backbone = 'cspdarknet'
        model_path = '/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/yolo-ep100-loss0.041-val_loss0.039.pth'
    elif type_model == 'ssd-mb2':
        model_name = 'ssd'
        backbone = 'mobilenetv2'
        model_path = '/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/ssd_mb2_0.9802_ep100.pth'    
    elif type_model == 'ssd-vgg':
        model_name = 'ssd'
        backbone = 'vgg'
        model_path = '/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/ssd_vgg_0.9819_ep100.pth'    

    origin_annotation_path = "/root/autodl-tmp/dachuang/adv_detect/adv_emsable/2007_train.txt"
    batch_size = 4
    num_workers = 4
    classes_path = "/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/classes.txt"
    pretrained = False

    anchors_path_yolo = 'model_data/yolo_anchors.txt'
    anchors_mask_yolo = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    phi = 'l'
    backbone_yolo = 'cspdarknet'
    model_path_yolo = "/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/yolo-ep100-loss0.041-val_loss0.039.pth"  # "/root/autodl-tmp/dachuang/adv_detect/adv_em/model_data/50_0.9690_ep015.pth"

    model_path_frcnn = "/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/frcnn101-ep100-loss0.667-val_loss0.617.pth" # "/root/autodl-tmp/dachuang/adv_detect/adv_em/model_data/frcnn_101_0.9658_ep070.pth" # 
    anchors_size_frcnn = [4, 16, 32]
    backbone_frcnn = 'resnet101'
    
    model_path_ssd = "/root/autodl-tmp/dachuang/adv_detect/adv_emsable/model_data/ssd_mb2_0.9802_ep100.pth"  # "/root/autodl-tmp/dachuang/adv_detect/adv_em/model_data/ssd_vgg_0.9819_ep090.pth"
    anchors_size_ssd    = [30, 60, 111, 162, 213, 264, 315]
    backbone_ssd = 'mb2'

    AA = True
    model_names = ['frcnn', 'yolov5', 'ssd']
