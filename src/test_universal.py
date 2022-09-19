from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)


  iter_round = 1
  num_iters_make = 10

  num_iter_test = len(dataset)
  noise_tot = torch.zeros([1,3,512,512]).cuda()

  for iter in range(iter_round):
    for ind in range(num_iters_make):
      img_id = dataset.images[ind]
      img_info = dataset.coco.loadImgs(ids=[img_id])[0]
      img_path = os.path.join(dataset.img_dir, img_info['file_name'])
      ret = detector.run(img_path,1,noise_tot)
      noise = ret['noise']
      noise_tot += noise
      #noise_total_ = noise_tot[0].cpu().numpy()
      #'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
      p_mean = np.array([0.408, 0.447, 0.470],
                        dtype=np.float32).reshape(3, 1, 1)
      p_std = np.array([0.289, 0.274, 0.278],
                       dtype=np.float32).reshape(3, 1, 1)
      #noise_total_ = (noise_total_ * p_std + p_mean)*255
      # print('L2 norm of pertubation:{} '.format(p2))
      print('---processing (pertubation iterating)--the {} th in {} th iter_round---'.format(ind, iter))
    p_mean = np.array([0.408, 0.447, 0.470],
                    dtype=np.float32).reshape(3, 1, 1)
    p_std = np.array([0.289, 0.274, 0.278],
                   dtype=np.float32).reshape(3, 1, 1)
  noise_total_ = noise_tot[0].cpu().numpy()
      #noise_total_ = (noise_total_ * p_std + p_mean) * 255
  norm = (np.linalg.norm(noise_total_[0]) + np.linalg.norm(noise_total_[1]) + np.linalg.norm(noise_total_[2])) / (
          512 * 512)
  print('L2 norm of pertubation:{} '.format(norm))
  noise_tot = noise_tot / norm * 0.001

  noise_tot_ = noise_tot.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
  p_mean = np.array([0.408, 0.447, 0.470],
                    dtype=np.float32).reshape(1, 1, 3)
  p_std = np.array([0.289, 0.274, 0.278],
                   dtype=np.float32).reshape(1, 1, 3)
  noise_tot_show = ((noise_tot_ * 10 * p_std + p_mean) * 255).astype(np.uint8)
  cv2.imwrite('./total_{}percent_Num{}_per.png'.format(num_iters_make/len(dataset),num_iters_make), noise_tot_show)

  results = {}
  bar = Bar('{}'.format(opt.exp_id), max=num_iter_test)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}

# test uni
  for ind in range(num_iter_test):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    ret = detector.run(img_path, 0, noise_tot)
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
      ind, num_iter_test, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  test(opt)
