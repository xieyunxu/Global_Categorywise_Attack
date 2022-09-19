from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import matplotlib.pyplot as plt

from models.losses import FocalLoss
from external.nms import soft_nms
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector_uniattack_release  import BaseDetector


class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, per_iters, per, return_time=False):
        ################################################################################
        #Global Category-wise Attack
        images.data += per.data
        max_pixel = torch.max(images.data.abs())
        eps = 0.05 * max_pixel

        iters = per_iters# 10 #循环的次数，取值为1是FGSM
        cates = 80  # 80 for COCO, 20 for PascalVOC
        attack_thres = 0.1 # 0.1 很关键的参数
        eps_each_iter = eps/iters
        x_adv = torch.autograd.Variable(images.data, requires_grad=True)
        crit = torch.nn.CrossEntropyLoss()
        hm_ori = self.model(images)[-1]['hm']
        hm_ori_sig = hm_ori.sigmoid()

        noise_tot = torch.zeros_like(x_adv.data)
        for iter in range(iters):
            x_adv_temp = torch.autograd.Variable(x_adv.data, requires_grad=True)
            hm_adv = self.model(x_adv_temp)[-1]['hm']
            hm_adv_sig = hm_adv.sigmoid()

            noise = torch.zeros_like(x_adv_temp)
            for cate in range(cates):
                label_loca_adv = (hm_adv_sig[:, cate, :, :] > attack_thres).nonzero()
                label_temp = torch.LongTensor([cate]).to(torch.device('cuda'))
                if len(label_loca_adv) == 0:
                    continue
                loss_count = 0
                for index, item in enumerate(label_loca_adv):
                    if hm_ori_sig[:, cate, item[1], item[2]] > attack_thres:
                        if loss_count == 0:
                            loss = crit(hm_adv[:, :, item[1], item[2]], label_temp)
                        else:
                            loss += crit(hm_adv[:, :, item[1], item[2]], label_temp)
                        loss_count += 1

                if loss_count == 0:
                    continue

                self.model.zero_grad()
                if x_adv_temp.grad is not None:
                    x_adv_temp.grad.data.fill_(0)
                loss.backward(retain_graph=True)
                noise_now = x_adv_temp.grad / x_adv_temp.grad.abs().max()
                noise += noise_now
            noise_tot = noise_tot + (noise.data.sign().mul(eps_each_iter)).data

            x_adv.data += noise.data.sign().mul(eps_each_iter)
            # x_adv.data[:, 0, :, :].clamp_(min_0, max_0)
            # x_adv.data[:, 1, :, :].clamp_(min_1, max_1)
            # x_adv.data[:, 2, :, :].clamp_(min_2, max_2)
            x_adv = torch.autograd.Variable(x_adv.data, requires_grad=True)

        ################################################################################
        output = self.model(x_adv)[-1]

        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg'] if self.opt.reg_offset else None

        if self.opt.flip_test:
            hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
            wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
            reg = reg[0:1] if reg is not None else None
        torch.cuda.synchronize()
        forward_time = time.time()
        dets,sc = ctdet_decode(hm, wh, reg=reg, K=self.opt.K)

        if return_time:
            return output, dets, forward_time, x_adv, noise_tot
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, img_name, noise, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            '''for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))'''

    def debug_noise(self, debugger, images, dets, output, img_name, noise, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)

            p_mean = np.array([0.485, 0.456, 0.406],
                              dtype=np.float32).reshape(1, 1, 3)
            p_std = np.array([0.229, 0.224, 0.225],
                             dtype=np.float32).reshape(1, 1, 3)

            noise = noise[i].detach().cpu().numpy().transpose(1, 2, 0)
            noise_ = ((noise * 30 * p_std + p_mean) * 255).astype(np.uint8)

            debugger.add_img(noise_, img_id='noise_img_{:.1f}'.format(scale))

            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results, img_name=''):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(
                        bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)
        debugger.save_all_imgs(path='D:/python/object_detection/CenterNet/outputs/attack/', genID=True)
