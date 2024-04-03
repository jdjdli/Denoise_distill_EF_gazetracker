# Denoising Distillation Makes Event-Frame Transformers as Accurate Gaze Trackers

The source code of the paper "Denoising Distillation Makes Event-Frame Transformers as Accurate Gaze Trackers"


[Jiading Li](https://alexander-kirillov.github.io/), [Zhiyu Zhu](https://scholars.cityu.edu.hk/en/persons/zhiyu-zhu(5432dc82-cfd7-473d-afbb-fa6d4cf25331).html), [Jinhui Hou](https://scholars.cityu.edu.hk/en/persons/jinhui-hou(4a1e6c89-e054-420a-bb67-fce3c89ee7eb).html), [Junhui Hou](https://scholars.cityu.edu.hk/en/persons/junhui-hou(1e5e437a-b84d-471d-af08-5f13a2d0b1c3).html), [Jinjian Wu](https://web.xidian.edu.cn/wjj/)


![Overview](asset/overview.png?raw=true)

We delve into the potential of utilizing pre-trained vision Transformers for cross-modal eye tracking, building upon their proven competence in capturing complex spatial-temporal relationships within multi-modal event-frame dataset. Our framework emphasizes the modeling of gaze shifts from a baseline calibration state, serving as an anchor, to the dynamically acquired state captured during actual use. Specificly, our approach ingests input in the form of a frame coupled with corresponding event data, meticulously aligned to highlight the visible region. The system is designed to scrutinize the discernible ocular zones, subsequently inferring the directional gaze vector as the output. Beyond the confines of static frame-based gaze estimation, the study of dynamic ocular movements constitutes an additional research trajectory within computer vision.

![Framework](asset/twostage.png?raw=true)

We first partition the whole dataset into several sub-regions, wherein each region's data is trained to cultivate a local expert network. Each expert network is simultaneously fed with the anchor state and a search state and utilize the transformers to explicitly model the correlation between the anchor and states. Subsequently, a latent-denoising knowledge distillation method is introduced to amalgamate the expertise of these several local expert networks into a singular, comprehensive student network. Note that the latent denoising and knowledge distillation are utilized in the training phase only.


## Visualization
### Static points
<div align="center">
  <img src="asset/eye_motion.png" width="80%" higth="80%">
</div>


### Free points
<div align="center">
  <img src="asset/visulization.gif">
</div>


## Getting Started

Clone the repository locally:

```
pip install+git https://github.com/jdjdli/Denoise_distill_EF_gazetracker.git
```

Create and activate a conda environment and install the required packages:

```

```




## Training
4 GPUs
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 8849 train_multiGPU.py
```


## Citing
```
```
