from mmdet3d.apis import init_model, inference_detector, inference_multi_modality_detector, inference_segmentor



model = init_model(config_file, checkpoint_file, device='cuda:0')
result, data = inference_segmentor(model, pcd)
# visualize the results and save the results in 'results' folder
model.show_results(data, result, out_dir='results', show=True)


# config_file = 'configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py'
# checkpoint_file = 'checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'

# config_file = 'configs/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py'
# checkpoint_file = 'checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20200925_230905-358fbe3b.pth'

config_file = 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'
checkpoint_file = 'checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth'

config_file = 'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py'
checkpoint_file = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'

config_file = 'configs/point_rcnn/point_rcnn_2x8_kitti-3d-3classes.py'
checkpoint_file = 'checkpoints/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth'

config_file = 'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
checkpoint_file = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'


config_file = 'configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py'
checkpoint_file = 'checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth'


# test a single image and show the results
pcd = 'tests/data/kitti/training/velodyne/000000.bin'
img = 'tests/data/kitti/training/image_2/000000.png'
ann_file = 'tests/data/kitti/kitti_infos_train.pkl'

pcd = 'demo/data/kitti/kitti_000008.bin'
img = 'demo/data/kitti/kitti_000008.png'
ann_file = 'demo/data/kitti/kitti_000008_infos.pkl'

pcd = 'demo/data/yc/yc_000000.bin'
pcd = 'demo/data/yc/yc_000000_aligned2.bin'
#pcd = 'demo/data/yc/yc_000000_aligned2_reduce_height0.5.bin'
#pcd = 'demo/data/yc/yc_000000_aligned2_reduce_height0.6.bin'
#pcd = 'demo/data/yc/yc_000000_aligned2_reduce_height0.7.bin'
#pcd = 'demo/data/yc/yc_000000_cloned.bin'
#pcd = 'demo/data/yc/yc_000000_cloned2.bin'
pcd = 'demo/data/yc/yc_000000_cloned3.bin'

img = 'demo/data/yc/yc_000000.jpg'
ann_file = 'demo/data/yc/yc_000000_infos.pkl'

if 0:
    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    result, data = inference_detector(model, pcd)
    # visualize the results and save the results in 'results' folder
    model.show_results(data, result, out_dir='results', show=True)
else:
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    result, data = inference_multi_modality_detector(model, pcd, img, ann_file)
    bboxes_3d = result[0]['pts_bbox']['boxes_3d']
    scores_3d = result[0]['pts_bbox']['scores_3d']
    labels_3d = result[0]['pts_bbox']['labels_3d']
    model.show_results(data, result, out_dir='results', show=True)

