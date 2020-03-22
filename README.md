# Awesome Vehicle Re-identification [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is a repository for organizing articles related to person re-identification. Most papers are linked to the pdf address provided by "arXiv" or "Openaccess". However, some papers require an academic license to browse. For example, IEEE, springer, and elsevier journal, etc.

**Other awesome re-identification**

- [Awesome Person Re-identification](https://github.com/bismex/Awesome-person-re-identification)
- [Awesome Cross-modality Person Re-identification](https://github.com/bismex/Awesome-cross-modality-person-re-identification)

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

## 1. Dataset and benchmark

- [[StanfordCars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)] 3D Object Representations for Fine-Grained Categorization (ICCV2013) [[paper](http://ai.stanford.edu/~jkrause/papers/3drr13.pdf)] 
- [[CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)] A Large-Scale Car Dataset for Fine-Grained Categorization and Verification (CVPR2015) [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yang_A_Large-Scale_Car_2015_CVPR_paper.pdf)] 
- [[VeRi-776](https://github.com/VehicleReId/VeRidataset)] A Deep Learning-Based Approach to Progressive Vehicle Re-identification for Urban Surveillance (ECCV2016) [[paper](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_53)] 
- [[VehicleReId](https://medusa.fit.vutbr.cz/traffic/datasets/)] Vehicle Re-Identification for Automatic Video Traffic Surveillance (CVPR2016) [[paper](http://openaccess.thecvf.com/content_cvpr_2016_workshops/w25/papers/Zapletal_Vehicle_Re-Identification_for_CVPR_2016_paper.pdf)] 
- [[PKU-VehicleID](https://pkuml.org/resources/pku-vehicleid.html)] Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles (CVPR2016) [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Deep_Relative_Distance_CVPR_2016_paper.pdf)] 
- [[PKU-VD](https://pkuml.org/resources/pku-vds.html)] Exploiting Multi-Grain Ranking Constraints for Precisely Searching
Visually-similar Vehicles (ICCV2017) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yan_Exploiting_Multi-Grain_Ranking_ICCV_2017_paper.pdf)] 
- [[PKU-Vehicle](https://github.com/PKU-IMRE/PKU-Vehicles)] Group-Sensitive Triplet Embedding for Vehicle Reidentification (TMM2018) [[paper](https://ieeexplore.ieee.org/abstract/document/8265213)] 
- [[Vehicle-1M](http://www.nlpr.ia.ac.cn/iva/homepage/jqwang/Vehicle1M.htm)] Learning Coarse-to-Fine Structured Feature Embedding for Vehicle Re-Identification (AAAI2018) [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16206/16270)] 
- [[VRIC](https://qmul-vric.github.io/)] Vehicle Re-Identification in Context (GCPR2018) [[paper](http://www.eecs.qmul.ac.uk/~xiatian/papers/AytacEtAl_GCPR2018.pdf)] 
- [[VERI-Wild](https://github.com/PKU-IMRE/VERI-Wild)] A Large Dataset and a New Method for
Vehicle Re-Identification in the Wild (CVPR2019) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lou_VERI-Wild_A_Large_Dataset_and_a_New_Method_for_Vehicle_CVPR_2019_paper.pdf)] 
- [[CityFlow](https://www.aicitychallenge.org/)] A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification (CVPR2019) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tang_CityFlow_A_City-Scale_Benchmark_for_Multi-Target_Multi-Camera_Vehicle_Tracking_and_CVPR_2019_paper.pdf)] 
- [[PAMTRI](https://github.com/yorkeyao/VehicleX)] Pose-Aware Multi-Task Learning for Vehicle Re-Identification Using Highly Randomized Synthetic Data (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Tang_PAMTRI_Pose-Aware_Multi-Task_Learning_for_Vehicle_Re-Identification_Using_Highly_Randomized_ICCV_2019_paper.pdf)] 
- [VRAI] Vehicle Re-identification in Aerial Imagery: Dataset and Approach (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Vehicle_Re-Identification_in_Aerial_Imagery_Dataset_and_Approach_ICCV_2019_paper.pdf)] 

---
## 2. Conference

- Viewpoint-Aware Attentive Multi-View Inference for Vehicle Re-Identification (CVPR2018) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Viewpoint-Aware_Attentive_Multi-View_CVPR_2018_paper.pdf)] [[github](https://github.com/csyizhou/Vehicle-Re-ID)]
- Part-Regularized Near-Duplicate Vehicle Re-Identification (CVPR2019) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Part-Regularized_Near-Duplicate_Vehicle_Re-Identification_CVPR_2019_paper.pdf)]
- Orientation Invariant Feature Embedding and Spatial Temporal Regularization for Vehicle Re-Identification (ICCV2017) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Orientation_Invariant_Feature_ICCV_2017_paper.pdf)] [[github](https://github.com/Zhongdao/VehicleReIDKeyPointData)]
- Exploiting Multi-Grain Ranking Constraints for Precisely Searching Visually-Similar Vehicles (ICCV2017) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yan_Exploiting_Multi-Grain_Ranking_ICCV_2017_paper.pdf)]
- Learning Deep Neural Networks for Vehicle Re-ID With Visual-Spatio-Temporal Path Proposals (ICCV2017) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Shen_Learning_Deep_Neural_ICCV_2017_paper.pdf)]
- A Dual-Path Model With Adaptive Attention for Vehicle Re-Identification (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Khorramshahi_A_Dual-Path_Model_With_Adaptive_Attention_for_Vehicle_Re-Identification_ICCV_2019_paper.pdf)] [[github](https://github.com/Pirazh/Vehicle_Key_Point_Orientation_Estimation)]
- Vehicle Re-Identification With Viewpoint-Aware Metric Learning (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chu_Vehicle_Re-Identification_With_Viewpoint-Aware_Metric_Learning_ICCV_2019_paper.pdf)]
- A Deep Learning-Based Approach to Progressive Vehicle Re-identification for Urban Surveillance (ECCV2016) [[paper](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_53)]
- Cross-View GAN Based Vehicle Generation for Re-identification (BMVC2017) [[paper](http://www.bmva.org/bmvc/2017/papers/paper186/paper186.pdf)]
- Learning Coarse-to-Fine Structured Feature Embedding for Vehicle Re-Identification (AAAI2018) [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16206/16270)]
- Vehicle re-identification by adversarial bi-directional LSTM network (WACV2018) [[paper](https://ieeexplore.ieee.org/abstract/document/8354181)]
- Fast Vehicle Identification via Ranked Semantic Sampling based Embedding (IJCAI2018) [[paper](https://pdfs.semanticscholar.org/8233/5202deecc8d6b8d275cad63287bc3ba032cf.pdf)]
- Large-Scale Vehicle Re-identification in Urban Surveillance Videos (ICME2016) [[paper](https://ieeexplore.ieee.org/abstract/document/7553002)]
- Improving Triplet-wise Training of Convolutional Neural Network for Vehicle Re-identification (ICME2017) [[paper](https://ieeexplore.ieee.org/abstract/document/8019491)]
- RAM: A Region-Aware Deep Model for Vehicle Re-identification (ICME2018) [[paper](https://arxiv.org/pdf/1806.09283.pdf)] [[github](https://github.com/liu-xb/RAM)]
- Multi-modal Metric Learning for Vehicle Re-identification in Traffic Surveillance Environment (ICIP2017) [[paper](https://ieeexplore.ieee.org/abstract/document/8296683)]
- Multi-Attribute Driven Vehicle Re-Identification with Spatial-Temporal Re-Ranking (ICIP2018) [[paper](https://ieeexplore.ieee.org/abstract/document/8451776/)]
- Joint Semi-supervised Learning and Re-ranking for Vehicle Re-identification (ICPR2018) [[paper](https://ieeexplore.ieee.org/abstract/document/8545584)]
- VP-ReID: vehicle and person re-identification system (ICMR2018) [[paper](https://dl.acm.org/doi/abs/10.1145/3206025.3206086?casa_token=TTQysSgzCg8AAAAA:INn7N2om4MXYnKe8f7WzxR5LNKOcw6rrLNwOHS5aPNGHowX0Yydf1FB_tUX46BJedWnzqNlevwY)]
- Vehicle Re-identification by Fusing Multiple Deep Neural Networks (IPTA2017) [[paper](https://ieeexplore.ieee.org/document/8310090)]
- Beyond Human-level License Plate Super-resolution with Progressive Vehicle Search and Domain Priori GAN (ACMMM2017) [[paper](https://dl.acm.org/doi/10.1145/3123266.3123422)]

---
## 3. Journal

- Vehicle Re-Identification by Deep Hidden Multi-View Inference (TIP2018) [[paper](https://ieeexplore.ieee.org/abstract/document/8325486)]
- Embedding Adversarial Learning for Vehicle Re-Identification (TIP2019) [[paper](https://ieeexplore.ieee.org/abstract/document/8653852)]
- VR-PROUD: Vehicle Re-identification using Progressive Unsupervised Deep Architecture (PR2019) [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320319300147)]
- PROVID: Progressive and Multimodal Vehicle Re-identification for Large-Scale Urban Surveillance (TMM2018) [[paper](https://ieeexplore.ieee.org/document/8036238)]
- Vehicle Re-Identification Using Quadruple Directional Deep Learning Features (TITS2019) [[paper](https://ieeexplore.ieee.org/abstract/document/8667847/)]
- Joint Feature and Similarity Deep Learning for Vehicle Re-identification (IEEE ACCESS 2018) [[paper](https://ieeexplore.ieee.org/abstract/document/8424333)]

---
## 4. Workshop

- [[AIC 2018](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Naphade_The_2018_NVIDIA_CVPR_2018_paper.pdf)] (CVPRW2018)
  - Vehicle Re-Identification With the Space-Time Prior [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Wu_Vehicle_Re-Identification_With_CVPR_2018_paper.pdf)] [[github](https://github.com/cw1204772/AIC2018_iamai)]
  - Unsupervised Vehicle Re-Identification Using Triplet Networks [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Marin-Reyes_Unsupervised_Vehicle_Re-Identification_CVPR_2018_paper.pdf)] [[github](https://github.com/NVIDIAAICITYCHALLENGE/2018AICITY_LasPalmas)]
  - Single-Camera and Inter-Camera Vehicle Tracking and 3D Speed Estimation based on Fusion of Visual and Semantic Features [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Tang_Single-Camera_and_Inter-Camera_CVPR_2018_paper.pdf)] [[github](https://github.com/zhengthomastang/2018AICity_TeamUW)]
  - Video Analytics in Smart Transportation for the AIC’18 Challenge [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Chang_Video_Analytics_in_CVPR_2018_paper.pdf)]
  - Graph@FIT Submission to the NVIDIA AI City Challenge 2018 [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Sochor_GraphFIT_Submission_to_CVPR_2018_paper.pdf)]
  - Dual-Mode Vehicle Motion Pattern Learning for High Performance Road Traffic Anomaly Detection [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Xu_Dual-Mode_Vehicle_Motion_CVPR_2018_paper.pdf)]
- [[AIC 2019](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Naphade_The_2019_AI_City_Challenge_CVPRW_2019_paper.pdf)] (CVPRW2019)
  - (Rank-1) Multi-camera vehicle tracking and re-identification based on visual and spatial-temporal features [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Tan_Multi-camera_vehicle_tracking_and_re-identification_based_on_visual_and_spatial-temporal_CVPRW_2019_paper.pdf)] [[github](https://github.com/wzgwzg/AICity)]
  - (Rank-2) Multi-View Vehicle Re-Identification using
Temporal Attention Model and Metadata Re-ranking [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Huang_Multi-View_Vehicle_Re-Identification_using_Temporal_Attention_Model_and_Metadata_Re-ranking_CVPRW_2019_paper.pdf)] [[github](https://github.com/ipl-uw/2019-CVPR-AIC-Track-2-UWIPL)]
  - (Rank-3) Vehicle Re-identification with Location and Time Stamps [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Lv_Vehicle_Re-Identification_with_Location_and_Time_Stamps_CVPRW_2019_paper.pdf)]
  - (Rank-4) VehicleNet: Learning Robust Feature Representation for Vehicle Re-identification [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Zheng_VehicleNet_Learning_Robust_Feature_Representation_for_Vehicle_Re-identification_CVPRW_2019_paper.pdf)]
  - (Rank-5) Multi-Camera Vehicle Tracking with Powerful Visual Features and Spatial-Temporal Cue [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/He_Multi-Camera_Vehicle_Tracking_with_Powerful_Visual_Features_and_Spatial-Temporal_Cue_CVPRW_2019_paper.pdf)] [[github](https://github.com/he010103/Traffic-Brain)]
  - (Rank-8) Attention Driven Vehicle Re-identification and Unsupervised Anomaly Detection for Traffic Understanding [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Khorramshahi_Attention_Driven_Vehicle_Re-identification_and_Unsupervised_Anomaly_Detection_for_Traffic_CVPRW_2019_paper.pdf)]
  - (Rank-13) Partition and Reunion: A Two-Branch Neural Network for Vehicle Re-identification [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Chen_Partition_and_Reunion_A_Two-Branch_Neural_Network_for_Vehicle_Re-identification_CVPRW_2019_paper.pdf)]
  - (Rank-18) Supervised Joint Domain Learning for Vehicle Re-Identification [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Liu_Supervised_Joint_Domain_Learning_for_Vehicle_Re-Identification_CVPRW_2019_paper.pdf)]
  - (Rank-19) Vehicle Re-Identification: Pushing the limits of re-identification [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Ayala-Acevedo_Vehicle_Re-Identification_Pushing_the_limits_of_re-identification_CVPRW_2019_paper.pdf)]
  - (Rank-23) Multi-camera Vehicle Tracking and Re-identification on AI City Challenge 2019 [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Chen_Multi-camera_Vehicle_Tracking_and_Re-identification_on_AI_City_Challenge_2019_CVPRW_2019_paper.pdf)]
  - (Rank-25) Vehicle Re-identification with Learned Representation and Spatial Verification and Abnormality Detection with Multi-Adaptive Vehicle Detectors for Traffic Video Analysis [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Nguyen_Vehicle_Re-identification_with_Learned_Representation_and_Spatial_Verification_and_Abnormality_CVPRW_2019_paper.pdf)] [[github](https://github.com/ngocminhbui/ai19_track2_hcmus)]
  - (Rank-36) Deep Feature Fusion with Multiple Granularity for Vehicle Re-identification [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Huang_Deep_Feature_Fusion_with_Multiple_Granularity_for_Vehicle_Re-identification_CVPRW_2019_paper.pdf)]
  - (Rank-45) Vehicle Re-Identification and Multi-Camera Tracking in Challenging City-Scale Environment [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Spanhel_Vehicle_Re-Identifiation_and_Multi-Camera_Tracking_in_Challenging_City-Scale_Environment_CVPRW_2019_paper.pdf)]
  - (Rank-50) AI City Challenge 2019 – City-Scale Video Analytics for Smart Transportation [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Chang_AI_City_Challenge_2019_--_City-Scale_Video_Analytics_for_Smart_CVPRW_2019_paper.pdf)] [[github](https://github.com/yrims/AIC19)]
  - (Rank-51) Multi-Task Mutual Learning for Vehicle Re-Identification [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Kanaci_Multi-Task_Mutual_Learning_for_Vehicle_Re-Identification_CVPRW_2019_paper.pdf)]
  - (Rank-54) Comparative Study of Various Losses for Vehicle Re-identification [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Shankar_Comparative_Study_on_various_Losses_for_Vehicle_Re-identification_CVPRW_2019_paper.pdf)]
- Others
  - Vehicle Re-Identification by Fine-Grained Cross-Level Deep Learning (BMVCW 2017) [[paper](http://www.eecs.qmul.ac.uk/~xiatian/papers/Kanac%C4%B1EtAl_BMVC2017WS.pdf)]
  - Deep Hashing with Multi-task Learning for Large-Scale Instance-Level Vehicle Search (ICMEW 2017) [[paper](https://ieeexplore.ieee.org/abstract/document/8026274)]

---
## 5. Others (single vehicle)

- BoxCars: 3D Boxes as CNN Input for Improved Fine-Grained Vehicle Recognition (CVPR2016) [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Sochor_BoxCars_3D_Boxes_CVPR_2016_paper.pdf)]
- Background Segmentation for Vehicle Re-Identification [[paper](https://arxiv.org/pdf/1910.06613.pdf)]
- CarFusion: Combining Point Tracking and Part Detection for Dynamic 3D Reconstruction of Vehicles (CVPR2018) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Reddy_CarFusion_Combining_Point_CVPR_2018_paper.pdf)]
- Visualizing the Invisible: Occluded Vehicle Segmentation and Recovery 
(ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Visualizing_the_Invisible_Occluded_Vehicle_Segmentation_and_Recovery_ICCV_2019_paper.pdf)]

---
## 6. Others (multiple vehicles)

- Highway Vehicle Counting in Compressed Domain (CVPR2016) [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Highway_Vehicle_Counting_CVPR_2016_paper.pdf)]
- Deep MANTA: A Coarse-To-Fine Many-Task Network for Joint 2D and 3D Vehicle Analysis From Monocular Image (CVPR2017) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chabot_Deep_MANTA_A_CVPR_2017_paper.pdf)]
- FCN-rLSTM: Deep Spatio-Temporal Neural Networks for Vehicle Counting in City Cameras (ICCV2017) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_FCN-rLSTM_Deep_Spatio-Temporal_ICCV_2017_paper.pdf)]
- Delving Into Robust Object Detection From Unmanned Aerial Vehicles: A Deep Nuisance Disentanglement Approach (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Delving_Into_Robust_Object_Detection_From_Unmanned_Aerial_Vehicles_A_ICCV_2019_paper.pdf)]
- Joint Monocular 3D Vehicle Detection and Tracking (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Hu_Joint_Monocular_3D_Vehicle_Detection_and_Tracking_ICCV_2019_paper.pdf)]
- Self-supervised Moving Vehicle Tracking with Stereo Sound (ICCV2019) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gan_Self-Supervised_Moving_Vehicle_Tracking_With_Stereo_Sound_ICCV_2019_paper.pdf)]

## 7. Others (code)
- https://github.com/Jakel21/vehicle-ReID-baseline
- https://github.com/icarofua/vehicle-ReId
- https://github.com/NVIDIAAICITYCHALLENGE/2018AICITY_Beihang
- https://github.com/Simon4Yan/feature_learning

---
## Reference

- https://github.com/bismex/Awesome-person-re-identification
- https://github.com/layumi/Vehicle_reID-Collection
- https://github.com/knwng/awesome-vehicle-re-identification
- https://github.com/VehicleReId/VeRidataset

