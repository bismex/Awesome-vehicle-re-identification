# Awesome Vehicle Re-identification [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is a repository for organizing articles related to person re-identification. Most papers are linked to the pdf address provided by "arXiv" or "Openaccess". However, some papers require an academic license to browse. For example, IEEE, springer, and elsevier journal, etc.

**People who meet the following criteria are free to request a pull (pull request).**
- Suggestions for new categories
- Changes to categories for some articles
- Corrections to the statistical tables
- Additions of a summary or performance

***I am currently studying as a visiting researcher at Carnegie Mellon University. So please email me if you are interested in collaborative research among those interested in re-identification in Pittsburgh or another city in the United States.***

***My person re-identification paper accepted to CVPR2020! See you in Seattle!*** 
- Feel free to visit my **[homepage](https://sites.google.com/site/seokeonchoi/)**
- Check the preprint at **[arxiv](https://arxiv.org/abs/1912.01230)**

***I plan to receive a doctoral degree in Dec. 2020 or Jun. 2021. I'm currently looking for a full-time job, residency program, or post-doc.***
- Feel free to visit my **[linkedin](https://www.linkedin.com/in/seokeon/)**

### :high_brightness: Updated 2020-03-22
- This repository was created.

---

## 1. Dataset

- [StanfordCars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) [[paper](http://ai.stanford.edu/~jkrause/papers/3drr13.pdf)] ICCV2013
- [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html) [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yang_A_Large-Scale_Car_2015_CVPR_paper.pdf)] CVPR2015
- [VeRi-776](https://github.com/VehicleReId/VeRidataset) [[paper](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_53)] ECCV2016
- [VehicleReId](https://medusa.fit.vutbr.cz/traffic/datasets/) [[paper](http://openaccess.thecvf.com/content_cvpr_2016_workshops/w25/papers/Zapletal_Vehicle_Re-Identification_for_CVPR_2016_paper.pdf)] CVPR2016
- [PKU-VehicleID](https://pkuml.org/resources/pku-vehicleid.html) [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Deep_Relative_Distance_CVPR_2016_paper.pdf)] CVPR2016
- [PKU-VD](https://pkuml.org/resources/pku-vds.html) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yan_Exploiting_Multi-Grain_Ranking_ICCV_2017_paper.pdf)] ICCV2017
- [PKU-Vehicle](https://github.com/PKU-IMRE/PKU-Vehicles) [[paper](https://ieeexplore.ieee.org/abstract/document/8265213)] TM2018
- [Vehicle-1M](http://www.nlpr.ia.ac.cn/iva/homepage/jqwang/Vehicle1M.htm) [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16206/16270)] AAAI2018
- [VRIC](https://qmul-vric.github.io/) [[paper](http://www.eecs.qmul.ac.uk/~xiatian/papers/AytacEtAl_GCPR2018.pdf)] GCPR2018
- [VERI-Wild](https://github.com/PKU-IMRE/VERI-Wild) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lou_VERI-Wild_A_Large_Dataset_and_a_New_Method_for_Vehicle_CVPR_2019_paper.pdf)] CVPR2019
- [CityFlow](https://www.aicitychallenge.org/) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tang_CityFlow_A_City-Scale_Benchmark_for_Multi-Target_Multi-Camera_Vehicle_Tracking_and_CVPR_2019_paper.pdf)] CVPR2019
- [VehicleX](https://github.com/yorkeyao/VehicleX) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Tang_PAMTRI_Pose-Aware_Multi-Task_Learning_for_Vehicle_Re-Identification_Using_Highly_Randomized_ICCV_2019_paper.pdf)] ICCV2019
- VRAI [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Vehicle_Re-Identification_in_Aerial_Imagery_Dataset_and_Approach_ICCV_2019_paper.pdf)]

---
## 1. Conference

- Viewpoint-Aware Attentive Multi-View Inference for Vehicle Re-Identification (CVPR2018) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Viewpoint-Aware_Attentive_Multi-View_CVPR_2018_paper.pdf)]
- Part-Regularized Near-Duplicate Vehicle Re-Identification (CVPR2019) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Part-Regularized_Near-Duplicate_Vehicle_Re-Identification_CVPR_2019_paper.pdf)]
- Orientation Invariant Feature Embedding and Spatial Temporal Regularization for Vehicle Re-Identification (ICCV2017) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Orientation_Invariant_Feature_ICCV_2017_paper.pdf)]
- Exploiting Multi-Grain Ranking Constraints for Precisely Searching Visually-Similar Vehicles (ICCV2017) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yan_Exploiting_Multi-Grain_Ranking_ICCV_2017_paper.pdf)]
- Learning Deep Neural Networks for Vehicle Re-ID With Visual-Spatio-Temporal Path Proposals (ICCV2017) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Shen_Learning_Deep_Neural_ICCV_2017_paper.pdf)]
- A Dual-Path Model With Adaptive Attention for Vehicle Re-Identification (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Khorramshahi_A_Dual-Path_Model_With_Adaptive_Attention_for_Vehicle_Re-Identification_ICCV_2019_paper.pdf)]
- Vehicle Re-Identification With Viewpoint-Aware Metric Learning (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chu_Vehicle_Re-Identification_With_Viewpoint-Aware_Metric_Learning_ICCV_2019_paper.pdf)]
- A Deep Learning-Based Approach to Progressive Vehicle Re-identification for Urban Surveillance (ECCV2016) [[paper](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_53)]
- Learning Coarse-to-Fine Structured Feature Embedding for Vehicle Re-Identification (AAAI2018) [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16206/16270)]
- Vehicle re-identification by adversarial bi-directional LSTM network (WACV2018) [[paper](https://ieeexplore.ieee.org/abstract/document/8354181)]

---
## 2. Journal

- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]

---
## 3. Workshop

- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]

---
## 4. Others (single vehicle)

- BoxCars: 3D Boxes as CNN Input for Improved Fine-Grained Vehicle Recognition (CVPR2016) [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Sochor_BoxCars_3D_Boxes_CVPR_2016_paper.pdf)]
- Background Segmentation for Vehicle Re-Identification [[paper](https://arxiv.org/pdf/1910.06613.pdf)]
- CarFusion: Combining Point Tracking and Part Detection for Dynamic 3D Reconstruction of Vehicles (CVPR2018) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Reddy_CarFusion_Combining_Point_CVPR_2018_paper.pdf)]
- Visualizing the Invisible: Occluded Vehicle Segmentation and Recovery 
(ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Visualizing_the_Invisible_Occluded_Vehicle_Segmentation_and_Recovery_ICCV_2019_paper.pdf)]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]

---
## 5. Others (multiple vehicles)

- Highway Vehicle Counting in Compressed Domain (CVPR2016) [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Highway_Vehicle_Counting_CVPR_2016_paper.pdf)]
- Deep MANTA: A Coarse-To-Fine Many-Task Network for Joint 2D and 3D Vehicle Analysis From Monocular Image (CVPR2017) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chabot_Deep_MANTA_A_CVPR_2017_paper.pdf)]
- FCN-rLSTM: Deep Spatio-Temporal Neural Networks for Vehicle Counting in City Cameras (ICCV2017) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_FCN-rLSTM_Deep_Spatio-Temporal_ICCV_2017_paper.pdf)]
- Delving Into Robust Object Detection From Unmanned Aerial Vehicles: A Deep Nuisance Disentanglement Approach (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Delving_Into_Robust_Object_Detection_From_Unmanned_Aerial_Vehicles_A_ICCV_2019_paper.pdf)]
- Joint Monocular 3D Vehicle Detection and Tracking (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Hu_Joint_Monocular_3D_Vehicle_Detection_and_Tracking_ICCV_2019_paper.pdf)]
- Self-supervised Moving Vehicle Tracking with Stereo Sound (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gan_Self-Supervised_Moving_Vehicle_Tracking_With_Stereo_Sound_ICCV_2019_paper.pdf)]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]
- [[paper]()]


---
