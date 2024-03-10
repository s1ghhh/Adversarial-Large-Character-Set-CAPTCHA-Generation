import torch

class config:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    letterbox_image = True
    input_shape = [544, 544]

    num_classes = 1
    confidence = 0.5
    nms_iou = 0.3
    class_names = ['char']
    reco_input_shape = [3, 128, 128]
    
    model_weight_path = './models/detect_model'
    output_path = './DSR_results'
    dataset_path = './CAPTCHA_examples'
    
    batch_size = 4
    num_workers = 4
    classes_path = f"{model_weight_path}/classes.txt"
    pretrained = False

    anchors_path_yolo = f"{model_weight_path}/yolo_anchors.txt"
    anchors_mask_yolo = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    phi = 'l'
    backbone_yolo = 'cspdarknet'
    model_path_yolo = f"{model_weight_path}/yolo-ep100-loss0.041-val_loss0.039.pth"

    model_path_frcnn = f"{model_weight_path}/frcnn-50-ep100-loss0.689-val_loss0.633.pth"
    anchors_size_frcnn = [4, 16, 32]
    backbone_frcnn = 'resnet50'
    
    model_path_ssd = f"{model_weight_path}/ssd_vgg_0.9819_ep100.pth"
    anchors_size_ssd    = [30, 60, 111, 162, 213, 264, 315]
    backbone_ssd = 'vgg'

