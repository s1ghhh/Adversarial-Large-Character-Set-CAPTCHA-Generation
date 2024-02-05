import torch

class config:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weight_path = ''
    dataset_path = ''
    detect_result_path = '../DSR_results'
    attn_input_shape = [3,128,128]
