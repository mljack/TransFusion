# TransFusion repository

PyTorch implementation of TransFusion for CVPR'2022 paper ["TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers"](https://arxiv.org/abs/2203.11496), by Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu and Chiew-Lan Tai.

This paper focus on LiDAR-camera fusion for 3D object detection. If you find this project useful, please cite:

```bash
@article{bai2021pointdsc,
  title={{TransFusion}: {R}obust {L}iDAR-{C}amera {F}usion for {3}D {O}bject {D}etection with {T}ransformers},
  author={Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu and Chiew-Lan Tai},
  journal={CVPR},
  year={2022}
}
```

## Introduction

LiDAR and camera are two important sensors for 3D object detection in autonomous driving. Despite the increasing popularity of sensor fusion in this field, the robustness against inferior image conditions, e.g., bad illumination and sensor misalignment, is under-explored. Existing fusion methods are easily affected by such conditions, mainly due to a hard association of LiDAR points and image pixels, established by calibration matrices.
We propose TransFusion, a robust solution to LiDAR-camera fusion with a soft-association mechanism to handle inferior image conditions. Specifically, our TransFusion consists of convolutional backbones and a detection head based on a transformer decoder. The first layer of the decoder predicts initial bounding boxes from a LiDAR point cloud using a sparse set of object queries, and its second decoder layer adaptively fuses the object queries with useful image features, leveraging both spatial and contextual relationships. The attention mechanism of the transformer enables our model to adaptively determine where and what information should be taken from the image, leading to a robust and effective fusion strategy. We additionally design an image-guided query initialization strategy to deal with objects that are difficult to detect in point clouds. TransFusion achieves state-of-the-art performance on large-scale datasets. We provide extensive experiments to demonstrate its robustness against degenerated image quality and calibration errors. We also extend the proposed method to the 3D tracking task and achieve the 1st place in the leaderboard of nuScenes tracking, showing its effectiveness and generalization capability.

![pipeline](resources/pipeline.png)

**updates**
- March 23, 2022: paper link added
- March 15, 2022: initial release

## Main Results

Detailed results can be found in [nuscenes.md](configs/nuscenes.md) and [waymo.md](configs/waymo.md). Configuration files and guidance to reproduce these results are all included in [configs](configs), we are not going to release the pretrained models due to the policy of Huawei IAS BU. 

### nuScenes detection test 

| Model   | Backbone | mAP | NDS  | Link  |
|---------|--------|--------|---------|---------|
| [TransFusion-L](configs/transfusion_nusc_voxel_L.py) | VoxelNet | 65.52 | 70.23 | [Detection](https://drive.google.com/file/d/1Wk8p2LJEhwfKfhsKzlU9vDBOd0zn38dN/view?usp=sharing)
| [TransFusion](configs/transfusion_nusc_voxel_LC.py) | VoxelNet | 68.90 | 71.68 | [Detection](https://drive.google.com/file/d/1X7_ig4v5A2vKsiHtUGtgeMN-0RJKsM6W/view?usp=sharing)

### nuScenes tracking test

| Model | Backbone | AMOTA |  AMOTP   | Link  |
|---------|--------|--------|---------|---------|
| [TransFusion-L](configs/transfusion_nusc_voxel_L.py) | VoxelNet | 0.686 | 0.529 | [Detection](https://drive.google.com/file/d/1Wk8p2LJEhwfKfhsKzlU9vDBOd0zn38dN/view?usp=sharing) / [Tracking](https://drive.google.com/file/d/1pKvRBUsM9h1Xgturd0Ae_bnGt0m_j3hk/view?usp=sharing)| 
| [TransFusion](configs/transfusion_nusc_voxel_LC.py)| VoxelNet | 0.718 | 0.551 | [Detection](https://drive.google.com/file/d/1X7_ig4v5A2vKsiHtUGtgeMN-0RJKsM6W/view?usp=sharing) / [Tracking](https://drive.google.com/file/d/1EVuS-MAg_HSXUVqMrXEs4-RpZp0p5cfv/view?usp=sharing)| 

### waymo detection validation

| Model   | Backbone | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   |
|---------|--------|---------|---------|---------|---------|
| [TransFusion-L](configs/transfusion_waymo_voxel_L.py) | VoxelNet | 65.07 | 63.70 | 65.97 | 64.91
| [TransFusion](configs/transfusion_waymo_voxel_LC.py) | VoxelNet | 65.11 | 64.02 | 67.40 | 65.51

## Use TransFusion

**Installation**

Please refer to [getting_started.md](docs/getting_started.md) for installation of mmdet3d. We use mmdet 2.10.0 and mmcv 1.2.4 for this project.

**Benchmark Evaluation and Training**

Please refer to [data_preparation.md](docs/getting_started.md) to prepare the data. Then follow the instruction there to train our model. All detection configurations are included in [configs](configs/).

## Acknowlegement

We sincerely thank the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [CenterPoint](https://github.com/tianweiy/CenterPoint), [GroupFree3D](https://github.com/zeliu98/Group-Free-3D) for open sourcing their methods.






<div align="center">
  <img src="resources/mmdet3d-logo.png" width="600"/>
</div>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection3d.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection3d/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection3d)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/blob/master/LICENSE)


**News**: We released the codebase v0.11.0.

In the recent [nuScenes 3D detection challenge](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) of the 5th AI Driving Olympics in NeurIPS 2020, we obtained the best PKL award and the second runner-up by multi-modality entry, and the best vision-only results. Code and models will be released soon!

Documentation: https://mmdetection3d.readthedocs.io/

## Introduction

English | [��������](README_zh-CN.md)

The master branch works with **PyTorch 1.3+**.

MMDetection3D is an open source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D detection. It is
a part of the OpenMMLab project developed by [MMLab](http://mmlab.ie.cuhk.edu.hk/).

![demo image](resources/mmdet3d_outdoor_demo.gif)

### Major features

- **Support multi-modality/single-modality detectors out of box**

  It directly supports multi-modality/single-modality detectors including MVXNet, VoteNet, PointPillars, etc.

- **Support indoor/outdoor 3D detection out of box**

  It directly supports popular indoor and outdoor 3D detection datasets, including ScanNet, SUNRGB-D, Waymo, nuScenes, Lyft, and KITTI.
  For nuScenes dataset, we also support [nuImages dataset](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/nuimages).

- **Natural integration with 2D detection**

  All the about **300+ models, methods of 40+ papers**, and modules supported in [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) can be trained or used in this codebase.

- **High efficiency**

  It trains faster than other codebases. The main results are as below. Details can be found in [benchmark.md](./docs/benchmarks.md). We compare the number of samples trained per second (the higher, the better). The models that are not supported by other codebases are marked by `��`.

  | Methods | MMDetection3D | [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) |[votenet](https://github.com/facebookresearch/votenet)| [Det3D](https://github.com/poodarchu/Det3D) |
  |:-------:|:-------------:|:---------:|:-----:|:-----:|
  | VoteNet | 358           | ��         |   77  | ��     |
  | PointPillars-car| 141           | ��         |   ��  | 140     |
  | PointPillars-3class| 107           |44     |   ��      | ��    |
  | SECOND| 40           |30     |   ��      | ��    |
  | Part-A2| 17           |14     |   ��      | ��    |

Like [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMCV](https://github.com/open-mmlab/mmcv), MMDetection3D can also be used as a library to support different projects on top of it.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.11.0 was released in 1/3/2021.
Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Benchmark and model zoo

Supported methods and backbones are shown in the below table.
Results and models are available in the [model zoo](docs/model_zoo.md).

Support backbones:

- [x] PointNet (CVPR'2017)
- [x] PointNet++ (NeurIPS'2017)
- [x] RegNet (CVPR'2020)

Support methods

- [x] [SECOND (Sensor'2018)](configs/second/README.md)
- [x] [PointPillars (CVPR'2019)](configs/pointpillars/README.md)
- [x] [FreeAnchor (NeurIPS'2019)](configs/free_anchor/README.md)
- [x] [VoteNet (ICCV'2019)](configs/votenet/README.md)
- [x] [H3DNet (ECCV'2020)](configs/h3dnet/README.md)
- [x] [3DSSD (CVPR'2020)](configs/3dssd/README.md)
- [x] [Part-A2 (TPAMI'2020)](configs/parta2/README.md)
- [x] [MVXNet (ICRA'2019)](configs/mvxnet/README.md)
- [x] [CenterPoint (CVPR'2021)](configs/centerpoint/README.md)
- [x] [SSN (ECCV'2020)](configs/ssn/README.md)
- [x] [ImVoteNet (CVPR'2020)](configs/imvotenet/README.md)

|                    | ResNet   | ResNeXt  | SENet    |PointNet++ | HRNet | RegNetX | Res2Net |
|--------------------|:--------:|:--------:|:--------:|:---------:|:-----:|:--------:|:-----:|
| SECOND             | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| PointPillars       | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| FreeAnchor         | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| VoteNet            | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| H3DNet            | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| 3DSSD            | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| Part-A2            | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| MVXNet             | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| CenterPoint        | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| SSN                | ?        | ?        | ?        | ?         | ?     | ?        | ?     |
| ImVoteNet            | ?        | ?        | ?        | ?         | ?     | ?        | ?     |

Other features
- [x] [Dynamic Voxelization](configs/dynamic_voxelization/README.md)

**Note:** All the about **300+ models, methods of 40+ papers** in 2D detection supported by [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) can be trained or used in this codebase.

## Installation

Please refer to [getting_started.md](docs/getting_started.md) for installation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMDetection3D. We provide guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with customized dataset](docs/2_new_data_model.md) for beginners. There are also tutorials for [learning configuration systems](docs/tutorials/config.md), [adding new dataset](docs/tutorials/customize_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [Waymo dataset](docs/tutorials/waymo.md).

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```

## Contributing

We appreciate all contributions to improve MMDetection3D. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection3D is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new 3D detectors.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
