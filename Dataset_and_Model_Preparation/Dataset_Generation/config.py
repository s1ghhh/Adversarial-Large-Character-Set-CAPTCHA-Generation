from PIL import ImageFilter

class config:
    type = 'dx'
    color_path = 'colors_' + type + '_rgb.txt'
    background_path = 'background_' + type
    char_every_char = 100
    char_num_range = list(range(4, 6))
    char_set_path = 'char_set_' + type + '.txt'
    char_size_range = list(range(49, 53))
    max_angle = 70
    color_noise = list(range(-10, 10))
    font_dir = 'font_' + type
    # blur = [ImageFilter.BLUR, ImageFilter.DETAIL, ImageFilter.EDGE_ENHANCE, ImageFilter.SMOOTH_MORE, ImageFilter.SMOOTH,
    #         ImageFilter.GaussianBlur(radius=3), ImageFilter.MaxFilter, ImageFilter.ModeFilter(5),
    #         ImageFilter.MedianFilter(5)]

    # blur = [ImageFilter.EDGE_ENHANCE, ImageFilter.SMOOTH_MORE, ImageFilter.SMOOTH, ImageFilter.GaussianBlur(radius=1),
    #         ImageFilter.MaxFilter, ImageFilter.ModeFilter(3), ImageFilter.ModeFilter(4), ImageFilter.ModeFilter(5),
    #         ImageFilter.MedianFilter(3), ImageFilter.MedianFilter(5)]
    # blur = [ImageFilter.BLUR]
    # blur = [ImageFilter.DETAIL]
    # blur = [ImageFilter.EDGE_ENHANCE]
    # blur = [ImageFilter.SMOOTH_MORE]
    # blur = [ImageFilter.GaussianBlur(radius=1),ImageFilter.GaussianBlur(radius=1.15),ImageFilter.GaussianBlur(radius=1.1),
    #         ImageFilter.GaussianBlur(radius=1.2),ImageFilter.GaussianBlur(radius=1.25),ImageFilter.GaussianBlur(radius=1.3),
    #         ImageFilter.GaussianBlur(radius=1.4),ImageFilter.GaussianBlur(radius=1.5),ImageFilter.GaussianBlur(radius=1.55)]

    # blur = [ImageFilter.GaussianBlur(radius=1),ImageFilter.GaussianBlur(radius=1.04),ImageFilter.GaussianBlur(radius=0.82),
    #         ImageFilter.GaussianBlur(radius=1.08),ImageFilter.GaussianBlur(radius=0.85),ImageFilter.GaussianBlur(radius=1.15),
    #         ImageFilter.GaussianBlur(radius=0.97),ImageFilter.GaussianBlur(radius=0.94),ImageFilter.GaussianBlur(radius=0.91)]

    blur = [ImageFilter.GaussianBlur(radius=0.8)]

    # blur =[ImageFilter.MaxFilter]
    # blur = [ImageFilter.ModeFilter(3)]
    captcha_path = 'D:/adv_captcha_0715/dx_dataset_gen/captcha'
    json_path = 'D:/adv_captcha_0715/dx_dataset_gen/json'
    char_path = 'D:/adv_captcha_0715/dx_dataset_gen/char'

    map_save_path = 'map_save.txt'
    up_sampling = 3

    background_width = 375
    background_height = 187

    noise_mean = 0
    noise_sigma = 1/255