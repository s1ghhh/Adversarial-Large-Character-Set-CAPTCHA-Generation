# Adversarial-Large-Character-Set-CAPTCHA-Generation

## Overview

Our code is composed of three parts: 

1. `ACG_Framework Part` can generate adversarial CAPTCHAs.

   * `stage1_generate_finegrained_noise_on_character.py`, the first stage, can use `ATTN_VNI_CT_FGSM` to generate fine-grained adversarial examples in the characters of the CAPTCHA.

   * `stage2_generate_global_noise_on_captcha.py` can use `GLOBAL_ENS_SVRE` to generate global adversarial examples in the CAPTCHA.

   * `config_stage1.py` and `config_stage2.py` contains the parameter setting and path setting of the adversarial CAPTCHAs generation process.

   * To generate adversarial CAPTCHAs, you can run `stage1_generate_finegrained_noise_on_character.py` first and then run `stage2_generate_global_noise_on_captcha.py`.

   * The workflow is shown as follows,
<div  align="center">  
<img src="src/framework.pdf" width="80%"> 
</div>

2. `Adv_Eval Part` includes a toolkit for evaluating various CAPTCHAs, which can measure metrics such as DSR, RSR, ASR, and so on.

   * `get_DSR.py` can evaluate the performance of the CAPTCHAs on the DSR metric.

   * `get_RSR_ASR.py` can evaluate the performance of the CAPTCHAs on the RSR and ASR metrics.

3. `Dataset and Model Preparation Part` contains the synthetic dataset and models.

   * `Model_Library_Building` contains the architecture and training parameters for all character detection and recognition models.

   * `Dataset_Generation` contains scripts to generate the synthetic dataset.

## Dataset

We collected 10 types of CAPTCHA commonly used on the Chinese Internet: Yidun (dun.163.com), Shumei (ishumei.com), Baidu (pan.baidu.com), YY (yy.com), Dingxiang (dingxiang-inc.com), 58 (58.com), Geetest (geetest.com), Dajie (dajie.com), Renmin (people.com.cn), Sougou (sougou.com). Some examples are shown in the figure below.

<div  align="center">  
<img src="src/CAPTCHA_examples.pdf" width="80%"> 
</div>



## Environments
```{bash}
torch==1.13.0
foolbox==3.3.2
h5py==3.1.0
opencv-python==4.6.0.66
Pillow==9.3.0
tensorflow-gpu==2.5.0
```

## Usage
* Generate adversarial CAPTCHAs
```{bash}
bash ./scripts/generate_adversarial_CAPTCHAs.sh
```

* Evaluate CAPTCHAs
```{bash}
bash ./scripts/evaluate_adversarial_CAPTCHAs.sh
```


## For complete dataset and well-trained models
To prevent the misuse of the CAPTCHA dataset and well-trained models, we will not release them here. Please contact `whzh.nc@scu.edu.cn` or `sunguoheng2k@gmail.com` to request the dataset and models.