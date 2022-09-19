from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms_39
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector_uniattack_release import BaseDetector


class MultiPoseDetector(BaseDetector):
    def __init__(self, opt):
        super(MultiPoseDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx

    def process(self, images, per_iters, per, return_time=False):
 ################################################################################
 # Global Category-wise Attack
        images.data += per.data

        max_pixel = torch.max(images.data.abs())
        eps = 0.05 * max_pixel# 0.05


       # print('___maxpixel___', eps)
        iters = per_iters# 10
        cates = 17
        attack_thres = 0.1 # 0.1
        eps_each_iter = eps/iters
        x_adv = torch.autograd.Variable(images.data, requires_grad=True)

        crit = torch.nn.CrossEntropyLoss()

        out = self.model(images)[-1]
        #hm_ori = out['hm'] # ['hm'] ['hm_hp']
        str_name = 'hm_hp'
        hmhp_ori = out[str_name]
        hm_ori_sig = hmhp_ori.sigmoid()


        noise_tot = torch.zeros_like(x_adv.data)

        for iter in range(iters):
            x_adv_temp = torch.autograd.Variable(x_adv.data, requires_grad=True)

            hm_adv = self.model(x_adv_temp)[-1][str_name]
            hm_adv_sig = hm_adv.sigmoid()

            noise = torch.zeros_like(x_adv_temp)
            #for cate in range(cates-1,cates): # range 左闭又开 if cates=1 cate=0 only
            for cate in range(cates):  # range 左闭又开 if cates=1 cate=0 only

                label_loca_adv = (hm_adv_sig[:, cate, :, :] > attack_thres).nonzero()
                # 用于寻找每层中的位置,生成一个横坐标和纵坐标集合

                label_temp = torch.LongTensor([cate]).to(torch.device('cuda'))

                if len(label_loca_adv) == 0:
                    continue

                loss_count = 0
                for index, item in enumerate(label_loca_adv):
                    # 对每层中的每个位置叠加梯度
                    if hm_ori_sig[:, cate, item[1], item[2]] > attack_thres:
                        # item[1]和item[2]为非零元素横纵坐标 item[0]

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

                noise += x_adv_temp.grad / x_adv_temp.grad.abs().max()

            noise_tot = noise_tot + (noise.data.mul(eps_each_iter)).data

            x_adv.data += noise.data.sign().mul(eps_each_iter)
            # x_adv.data[:, 0, :, :].clamp_(min_0, max_0)
            # x_adv.data[:, 1, :, :].clamp_(min_1, max_1)
            # x_adv.data[:, 2, :, :].clamp_(min_2, max_2)
            x_adv = torch.autograd.Variable(x_adv.data, requires_grad=True)

        ################################################################################
        output = self.model(x_adv)[-1]

        output['hm'] = output['hm'].sigmoid_()
        if self.opt.hm_hp and not self.opt.mse_loss:
            output['hm_hp'] = output['hm_hp'].sigmoid_()

        reg = output['reg'] if self.opt.reg_offset else None
        hm_hp = output['hm_hp'] if self.opt.hm_hp else None
        hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
        torch.cuda.synchronize()
        forward_time = time.time()


        dets = multi_pose_decode(
            output['hm'], output['wh'], output['hps'],
            reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

        if return_time:
            return output, dets, forward_time, x_adv, noise_tot
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = multi_pose_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
            # import pdb; pdb.set_trace()
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')
    def debug_noise(self, debugger, images, dets, output, img_name, noise, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)

            p_mean = np.array([0.408, 0.447, 0.470],
                              dtype=np.float32).reshape(1, 1, 3)
            p_std = np.array([0.289, 0.274, 0.278],
                             dtype=np.float32).reshape(1, 1, 3)

            noise = noise[i].detach().cpu().numpy().transpose(1, 2, 0)
            noise_ = ((noise * 1 * p_std + p_mean)*255).astype(np.uint8)

            debugger.add_img(noise_, img_id='noise_img_{:.1f}'.format(scale))

            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger,image, results):
        debugger.add_img(image, img_id='multi_pose')#在这里加入的
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                # debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
                debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
       # debugger.show_all_imgs(pause=self.pause)
        debugger.save_all_imgs(path='D:/python/object_detection/CenterNet/outputs/uniattack/', genID=True)



