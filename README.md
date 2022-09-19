# Global_-Category-wise_Attack
# Getting Started
This document provides tutorials of Global Catergory-wise Attack (GCA).
![](/readme/fig1.png)
Basically it contains two parts:
1) The installation procedures of the target network CenterNet, more details can be found at [Object as Points](http://arxiv.org/abs/1904.07850)
2) The installation procedures of GCA.
3) Make sure you have finished [Installation Procedures](/readme/INSTALL.md) and [Datasets](/readme/DATA.md) before started.

## Networks download 
Download the models you want to evaluate from the [model zoo](/readme/MODEL_ZOO.md) and put them in `Root_File/models/`. 

##Evaluation
### Detection
To evaluate PascalVOC object detection
~~~
> Clean outputs:
> run:
python test.py ctdet --exp_id pascal_dla_1x_clean --dataset pascal --arch dla_34 --not_prefetch_test --load_model ../models/ctdet_pascal_dla_512.pth
~~~
~~~
> Adversarial outputs (GCA):
> run:
python test_universal.py ctdet --exp_id pascal_dla_1x_Uniattack --dataset pascal --arch dla_34 --not_prefetch_test --load_model ../models/ctdet_pascal_dla_512.pth
~~~
### Pose estimation
~~~
> Clean outputs:
> run:
python test.py multi_pose --exp_id coco_resdcn18_clean --dataset coco_hp --arch dla_34 --not_prefetch_test --load_model ../models/multi_pose_dla_1x.pth
~~~
~~~
> Adversarial outputs (GCA):
> run:
python test_universal.py multi_pose --exp_id coco_resdcn18_attack_uni --dataset coco_hp --arch dla_34 --not_prefetch_test --load_model ../models/multi_pose_dla_1x.pth
~~~
