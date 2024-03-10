import torch

class config:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weight_path = './models/reco_model'
    dataset_path = './CAPTCHA_examples'
    detect_result_path = './DSR_results'
    attn_input_shape = [3,128,128]
