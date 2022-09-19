#Universal attack
#ctdet
python test_universal.py ctdet --exp_id coco_resdcn18_Uniattack --dataset coco --arch resdcn_18 --not_prefetch_test --load_model ../models/ctdet_coco_resdcn18.pth
#multi_pose
python test_universal.py multi_pose --exp_id coco_resdcn18_attack_uni --dataset coco_hp --arch dla_34 --not_prefetch_test --load_model ../models/multi_pose_dla_1x.pth

pascal_class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
  "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
  "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

coco_class_name = [
     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush']
      'ctdet': {'default_resolution': [512, 512], 'num_classes': 80,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'exdet': {'default_resolution': [512, 512], 'num_classes': 80,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'multi_pose': {
        'default_resolution': [512, 512], 'num_classes': 1,
        'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
        'dataset': 'coco_hp', 'num_joints': 17,
        'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                     [11, 12], [13, 14], [15, 16]]},

