## DMR: Disentangling Marginal Representations for Out-of-Distribution Detection

<!--[![arXiv](https://img.shields.io/badge/arXiv-2307.00000-b31b1b.svg)](https://vuno.co/)-->

* This repository provides official PyTorch implementations for <b>DMR</b>.
* This work has been accepted to the **CVPR 2024** workshop on [VAND 2.0: Visual Anomaly and Novelty Detection (VAND)](https://sites.google.com/view/vand-2-0-cvpr-2024/home).

<!--<p align="center"><img src="/sources/method_flow.png" width=90%/></p>-->

#### Authors: [Dasol Choi](https://github.com/Dasol-Choi), [Dongbin Na](https://github.com/ndb796)

### Abstract

> Out-of-distribution (OOD) detection is crucial for the reliable deployment of deep-learning applications. When a given input image does not belong to any categories of the deployed classification model, the classification model is expected to alert the user that the predicted outputs might be unreliable.
Recent studies have shown that utilizing a large amount of explicit OOD training data helps improve OOD detection performance. However, collecting explicit real-world OOD data is burdensome, and pre-defining all out-of-distribution labels is fundamentally difficult. In this work, we present a novel method, Disentangling Marginal Representations (DMR), that generates artificial OOD training data by extracting marginal features from images of an In-Distribution (ID) training dataset and manipulating these extracted marginal representations. DMR is intuitive and can be used as a realistic solution that does not require any extra real-world OOD data. Moreover, our method can be simply applied to pre-trained classifier networks without affecting the original classification performance. We demonstrate that a shallow rejection network that is trained on the small subset of synthesized OOD training data generated from our method and attachable to the classifier network achieves superior OOD detection performance. With extensive experiments, we show that our proposed method significantly outperforms the state-of-the-art OOD detection methods on the broadly used CIFAR-10 and CIFAR-100 detection benchmark datasets. We also demonstrate that our proposed method can be further improved when combined with existing methods.

### Dataset Configuration

* We use all the 50,000 $\mathcal{X}^{ID}$ images from CIFAR-10 and CIFAR-100 as the ID dataset.
* The synthesized 50,000 $\mathcal{X}^{OOD}_{train}$ images are used as the OOD training dataset for <b>KIRBY</b> and <b>DMR</b>.
* <b>OOD dataset for evaluation</b>: (1) SVHN, (2) Textures, (3) LSUN-Crop, (4) Tiny-ImageNet, (5) Places-365, (6) Gaussian Noise.

### OOD Detection Methods

#### Post-hoc methods

* MSP (Maximum Softmax Probability) (ICLR 2017) [\[Paper\]](https://arxiv.org/abs/1610.02136)
* ODIN (ICLR 2018) [\[Paper\]](https://arxiv.org/abs/1706.02690)
* Mahalanobis	(NeurIPS 2018) [\[Paper\]](https://arxiv.org/abs/1807.03888)
* Energy-based (NeurIPS 2020) [\[Paper\]](https://arxiv.org/abs/2010.03759)
* Entropy (CVPR 2021) [\[Paper\]](https://arxiv.org/abs/2010.03759)
* Maximum Logit (ICML 2022) [\[Paper\]](https://arxiv.org/abs/1911.11132)
* KL-Matching (ICML 2022) [\[Paper\]](https://arxiv.org/abs/1911.11132)
* ViM (CVPR 2022) [\[Paper\]](https://arxiv.org/abs/2203.10807)

#### Training methods that utilize the OOD training data

* OE (Outlier Exposure) (ICLR 2019) [\[Paper\]](https://arxiv.org/abs/1812.04606)
* KIRBY (AAAI 2023) [\[Paper\]](https://arxiv.org/abs/2301.13012)
  * OOD training images generation codes: [\[CIFAR-10\]](./OOD_generation/KIRBY_for_CIFAR10.ipynb)
  * Generated OOD training images (.zip): [\[CIFAR-10\]](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EZIOP0pq3ZpMnMX5o5lsOK0BDlUbJ6_f-3gJWkCgukzOsA)
* <b>Ours</b>
  * OOD training image generation codes:
    * Multiple Latent Mix-up <b>without DMR</b>: [\[CIFAR-10\]](./OOD_generation/Multiple_without_DMR_for_CIFAR10.ipynb)
    * Single Latent Inversion <b>with DMR</b>: [\[CIFAR-10\]](./OOD_generation/DMR_Single_for_CIFAR10.ipynb)
    * Multiple Latent Mix-up <b>with DMR</b>: [\[CIFAR-10\]](./OOD_generation/DMR_Multiple_for_CIFAR10.ipynb)
  * Generated OOD training images (.zip):
    * Multiple Latent Mix-up <b>without DMR</b>: [\[CIFAR-10\]](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EUIlZ7z0umtCuHGsg3kgCHMBCnhDqW0q373ODcakgrabNw)
    * Single Latent Inversion <b>with DMR</b>: [\[CIFAR-10\]](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EUM9c0EYbUJEube_0rGrl5YB19N7N3Oexjtqk6tvr11YVQ)
    * Multiple Latent Mix-up <b>with DMR</b>: [\[CIFAR-10\]](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EZuQVjyaP6tMgBi-uMfr4-wBeTSzxiUALKR51667nBYQYQ)

### Evaluations

* Baseline Evaluations:  [\[CIFAR-10\]](./Evaluation/OOD_Baseline_Evaluation_CIFAR10.ipynb), [\[CIFAR-100\]](./Evaluation/OOD_Baseline_Evaluation_CIFAR100.ipynb)
* DMR Evaluations: [\[CIFAR-10\]](./OOD_generation/DMR_Evaluation_CIFAR10.ipynb), [\[CIFAR-100\]](./OOD_generation/DMR_Evaluation_CIFAR100.ipynb)


## Citation
<pre>
@inproceedings{choi2024dmr,
  title={DMR: Disentangling Marginal Representations for Out-of-Distribution Detection},
  author={Choi, Dasol and Na, Dongbin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4032--4041},
  year={2024}
}
