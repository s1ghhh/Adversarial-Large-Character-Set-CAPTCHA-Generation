# Adversarial-Large-Character-Set-CAPTCHA-Generation

### Our code is composed of three parts.
1. `ACG_Framework Part` can generate adversarial CAPTCHAs.

   * `stage1_generate_finegrained_noise_on_character.py`, the first stage, can use `ATTN_VNI_CT_FGSM` to generate fine-grained adversarial examples in the characters of the CAPTCHA.

   * `stage2_generate_global_noise_on_captcha.py` can use `GLOBAL_ENS_SVRE` to generate global adversarial examples in the CAPTCHA.

   * `config_stage1.py` and `config_stage2.py` contains the parameter setting and path setting of the adversarial CAPTCHAs generation process.

   * To generate adversarial CAPTCHAs, you can run `stage1_generate_finegrained_noise_on_character.py` first and then run `stage2_generate_global_noise_on_captcha.py`.

2. `Adv_Eval Part` includes a toolkit for evaluating various CAPTCHAs, which can measure metrics such as DSR, RSR, ASR, and so on.

   * `get_DSR.py` can evaluate the performance of the CAPTCHAs on the DSR metric.

   * `get_RSR_ASR.py` can evaluate the performance of the CAPTCHAs on the RSR and ASR metrics.

3. `Dataset and Model Preparation Part` contains the synthetic dataset and models.

   * `Model_Library_Building` contains the architecture and training parameters for all character detection and recognition models.

   * `Dataset_Generation` contains scripts to generate the synthetic dataset.

To prevent the misuse of the CAPTCHA dataset and well-trained models, we will not release them here. Please contact `sunguoheng2k@gmail.com` to request the dataset and models.