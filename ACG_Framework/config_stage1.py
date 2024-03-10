import torch
from parso import parse

class config:

    scheme = 'geetest'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reco_batch_size = 50
    reco_input_shape = [3, 128, 128]
    reco_height = reco_input_shape[1]
    reco_width = reco_input_shape[2]

    capthca_path = f'./CAPTCHA_examples/{scheme}/test_captcha'
    json_path = f'./CAPTCHA_examples/{scheme}/test_captcha_label'
    class2idx_path = f'./CAPTCHA_examples/{scheme}/class_to_idx.txt'
    reco_output_path = f'./CAPTCHA_examples/{scheme}/JPEGImages_char_noised'

    reco_model_weight_path = './models/reco_model'
    reco_white_box_model_path = f"{reco_model_weight_path}/{scheme}/IncResV2/final_model.pth"
    reco_black_box_models = [(scheme, 'IncV3'), (scheme, 'Res50'), (scheme, 'Vgg16')]
    reco_black_box_model_path = [f"./models/reco_model/{scheme}/{model}/final_model.pth" for scheme, model in reco_black_box_models]


    # parameters of M-VNI-CT-FGSM
    # max perturbation of M-VNI-CT-FGSM
    reco_max_eps = 25.5
    # maximum epsilon in M-VNI-CT-FGSM
    central_eps = 76.5

    # number of iteration in M-VNI-CT-FGSM
    reco_iter = 10

    # number neighbours in M-VNI-CT-FGSM
    reco_N = 10

    # momentum of M-VNI-CT-FGSM
    reco_momentum = 1.0

    # number of iteration in M-VNI-CT-FGSM
    reco_beta = 10

    # transformation probablity in M-VNI-CT-FGSM
    reco_prob = 0.7

    # image resize for M-VNI-CT-FGSM
    reco_diver_image_resize = 160


    
