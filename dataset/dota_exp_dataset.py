"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
import json
import numpy as np
import csv
from collections import defaultdict
import math
import multiprocessing
import pickle
from random import shuffle

import torch
from torch.utils.data import Dataset
from pandarallel import pandarallel


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Device >>> ",torch.cuda.current_device())

# os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
# pandarallel.initialize(nb_workers=30,progress_bar=True,verbose=2,use_memory_fs = False )

ROOT_DIR = os.path.abspath('./')
sys.path.append(ROOT_DIR)

from utils import segment_iou

# dataloader for training
class DoTADataset(Dataset):
    def __init__(self, meta_data, frames_path, 
                 slide_window_size, kernel_list, classes,
                 pos_thresh, neg_thresh, stride_factor, save_samplelist=False,
                 load_samplelist=False, sample_listpath=None):
        super(DoTADataset, self).__init__()

        segment_data = generate_segment_exp(meta_data, classes, frames_path) 

        '''
            'v_id' : [num_frames, start, end, class]
            {'0RJPQ_97dcs_000199': [[100, 40, 70, 3]], '0RJPQ_97dcs_000307': [[110, 23, 53, 9]]
        '''

        self.slide_window_size = slide_window_size
        # all the anchors
        anc_len_lst = []
        anc_cen_lst = []
        
        for i in range(0, len(kernel_list)):
            kernel_len = kernel_list[i]
            anc_cen = np.arange(float((kernel_len) / 2.), float(
                slide_window_size + 1 - (kernel_len) / 2.), math.ceil(kernel_len/stride_factor))
            anc_len = np.full(anc_cen.shape, kernel_len)
            anc_len_lst.append(anc_len)
            anc_cen_lst.append(anc_cen)
        anc_len_all = np.hstack(anc_len_lst)
        anc_cen_all = np.hstack(anc_cen_lst)

        pos_anchor_stats = []
        neg_anchor_stats = []
        # load annotation per video and construct training set
        missing_prop = 0
        print("Here *******************88")
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = [None]*len(meta_data)
            vid_idx = 0
            
            for vid, val in meta_data.items():
                segement_video = np.array(segment_data[vid])
                # results[vid_idx] = pool.apply_async(_get_pos_neg_exp,(meta_data, vid, segement_video, slide_window_size, anc_len_all, anc_cen_all, pos_thresh, neg_thresh))
                print("vid_idx >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ", vid_idx)
                vid_idx += 1
        #     results = results[:vid_idx]
        #     for i, r in enumerate(results):
        #         results[i] = r.get()

        # vid_counter = 0
        # for r in results:
        #     if r is not None:
        #         vid_counter += 1
        #         video_id, total_frame, pos_seg, neg_seg = r
        #         # missing_prop += is_missing
        #         npos_seg = 0
        #         for k in pos_seg:
        #             # all neg_segs are the same, since they need to be negative
        #             # for all samples
        #             all_segs = pos_seg[k]
        #             sent = all_segs[0][-1] #[s[-1] for s in all_segs]
        #             other = [s[:-1] for s in all_segs]
        #             self.sample_list.append(
        #                 (video_id, other, sent, neg_seg, total_frame))
        #             npos_seg += len(pos_seg[k])

        #         pos_anchor_stats.append(npos_seg)
        #         neg_anchor_stats.append(len(neg_seg))

        # print('total number of {} videos: {}'.format(split, vid_counter))
        # print('total number of {} samples (unique segments): {}'.format(
        #     split, len(self.sample_list)))
        # print('avg pos anc: {:.2f} avg neg anc: {:.2f}'.format(
        #     np.mean(pos_anchor_stats), np.mean(neg_anchor_stats)
        # ))

        # if save_samplelist:
        #     with open(sample_listpath, 'wb') as f:
        #         pickle.dump(self.sample_list, f)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        video_id, pos_seg, ann_class_id, neg_seg, total_frame = self.sample_list[index]
        resnet_feat = torch.from_numpy(
            np.load(video_prefix + '_resnet.npy')).float()
        bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()
        img_feat = torch.FloatTensor(np.zeros((self.slide_window_size,
                                     resnet_feat.size(1)+bn_feat.size(1))))
        torch.cat((resnet_feat, bn_feat), dim=1,
                  out=img_feat[:min(total_frame, self.slide_window_size)])

        return (pos_seg, ann_class_id, neg_seg, img_feat)


def _get_pos_neg_exp(vid, segement_video, slide_window_size, anc_len_all, anc_cen_all, pos_thresh, neg_thresh):
    
    
    vgg_feats = torch.from_numpy(np.load(vid + '.npy')).float()
    total_frame = vgg_feats.size(0)
    if total_frame != segment_video[:, 0]:
        print("Here**** \n feat_size {} \t seg_frame {}".format(total_frame, segment_video[:, 0]))
        
    window_start = 0
    window_end = slide_window_size
    window_start_t = window_start
    window_end_t = window_end
    pos_seg = defaultdict(list)
    neg_overlap = [0] * anc_len_all.shape[0]
    pos_collected = [False] * anc_len_all.shape[0]
    for j in range(anc_len_all.shape[0]):
        potential_match = []
        gt_start = segement_video[0][1]
        gt_end = segement_video[0][2]
        print("GT start: {} \t end {}".format(gt_start, gt_end))
        if gt_start > gt_end:
            gt_start, gt_end = gt_end, gt_start
        if anc_cen_all[j] + anc_len_all[j] / 2. <= total_frame:
            if window_start_t <= seg[
                0] and window_end_t * 2 >= seg[1]:
                overlap = segment_iou(np.array([gt_start, gt_end]), np.array([[
                    anc_cen_all[j] - anc_len_all[j] / 2.,
                    anc_cen_all[j] + anc_len_all[j] / 2.]]))

                neg_overlap[j] = max(overlap, neg_overlap[j])

                if not pos_collected[j] and overlap >= pos_thresh:
                    len_offset = math.log(
                        (gt_end - gt_start) / anc_len_all[j])
                    cen_offset = ((gt_end + gt_start) / 2. -
                                    anc_cen_all[j]) / anc_len_all[j]
                    potential_match.append(
                        (ann_idx, j, overlap, len_offset, cen_offset,
                            segment_video[0][3]))
                    pos_collected[j] = True

        filled = False
        for item in potential_match:
            if item[0] not in pos_seg:
                filled = True
                pos_seg[item[0]].append(tuple(item[1:]))
                break

        if not filled and len(potential_match)>0:
            # randomly choose one
            shuffle(potential_match)
            item = potential_match[0]
            pos_seg[item[0]].append(tuple(item[1:]))

    # missing_prop = 0
    # if len(pos_seg.keys()) != len(annotations):
    #     print('Some annotations in video {} does not have '
    #             'any matching proposal'.format(video_prefix))
    #     missing_prop = len(annotations) - len(pos_seg.keys())

    neg_seg = []
    for oi, overlap in enumerate(neg_overlap):
        if overlap < neg_thresh:
            neg_seg.append((oi, overlap))

    npos_seg = 0
    for k in pos_seg:
        npos_seg += len(pos_seg[k])

    print(
        'pos anc: {}, neg anc: {}'.format(npos_seg,
                                            len(neg_seg)))

    return vid, total_frame, pos_seg, neg_seg
    
    # ================================================================================================

def _get_pos_neg(split_path, annotations, vid,
                 slide_window_size, anc_len_all,
                 anc_cen_all, pos_thresh, neg_thresh):
    if os.path.isfile(os.path.join(split_path, vid + '_bn.npy')):
        print('video: {}'.format(vid))

        video_prefix = os.path.join(split_path, vid)

        # load feature
        # T x H
        vgg_feats = torch.from_numpy(
            np.load(video_prefix + '_resnet.npy')).float()
        bn_feat = torch.from_numpy(
            np.load(video_prefix + '_bn.npy')).float()

        if resnet_feat.size(0) != bn_feat.size(0):
            raise Exception(
                'number of frames does not match in feature!')
        total_frame = bn_feat.size(0)

        window_start = 0
        window_end = slide_window_size
        window_start_t = window_start
        window_end_t = window_end
        pos_seg = defaultdict(list)
        neg_overlap = [0] * anc_len_all.shape[0]
        pos_collected = [False] * anc_len_all.shape[0]
        for j in range(anc_len_all.shape[0]):
            potential_match = []
            for ann_idx, ann in enumerate(annotations):
                seg = ann['segment']
                gt_start = seg[0]
                gt_end = seg[1]
                if gt_start > gt_end:
                    gt_start, gt_end = gt_end, gt_start
                if anc_cen_all[j] + anc_len_all[j] / 2. <= total_frame:
                    if window_start_t <= seg[
                        0] and window_end_t * 2 >= \
                            seg[1]:
                        overlap = segment_iou(np.array([gt_start, gt_end]), np.array([[
                            anc_cen_all[j] - anc_len_all[j] / 2.,
                            anc_cen_all[j] + anc_len_all[j] / 2.]]))

                        neg_overlap[j] = max(overlap, neg_overlap[j])

                        if not pos_collected[j] and overlap >= pos_thresh:
                            len_offset = math.log(
                                (gt_end - gt_start) / anc_len_all[j])
                            cen_offset = ((gt_end + gt_start) / 2. -
                                          anc_cen_all[j]) / anc_len_all[j]
                            potential_match.append(
                                (ann_idx, j, overlap, len_offset, cen_offset,
                                 ann['sentence_idx']))
                            pos_collected[j] = True

            filled = False
            for item in potential_match:
                if item[0] not in pos_seg:
                    filled = True
                    pos_seg[item[0]].append(tuple(item[1:]))
                    break

            if not filled and len(potential_match)>0:
                # randomly choose one
                shuffle(potential_match)
                item = potential_match[0]
                pos_seg[item[0]].append(tuple(item[1:]))

        missing_prop = 0
        if len(pos_seg.keys()) != len(annotations):
            print('Some annotations in video {} does not have '
                  'any matching proposal'.format(video_prefix))
            missing_prop = len(annotations) - len(pos_seg.keys())

        neg_seg = []
        for oi, overlap in enumerate(neg_overlap):
            if overlap < neg_thresh:
                neg_seg.append((oi, overlap))

        npos_seg = 0
        for k in pos_seg:
            npos_seg += len(pos_seg[k])

        print(
            'pos anc: {}, neg anc: {}'.format(npos_seg,
                                              len(neg_seg)))

        return video_prefix, total_frame, pos_seg, neg_seg, missing_prop
    else:
        return None

def generate_classes(data):
    class_list = []
    for vid, vinfo in data.items():
        class_list.append(vinfo['anomaly_class'])
    class_list = list(set(class_list))
    class_list = sorted(class_list)
    classes = {'natural': 0}
    for i,cls in enumerate(class_list):
        classes[cls] = i + 1
    return classes

def generate_segment_exp(data, classes, frames_dir):
    segment = {}
    video_list = set(os.listdir(frames_dir))
    for vid, vinfo in data.items():
        if vid in video_list:
            segment[vid] = []
            total_frames = vinfo['num_frames']
            start_time = vinfo['anomaly_start']
            end_time = vinfo['anomaly_end']
            label = classes[vinfo['anomaly_class']]
            segment[vid].append([total_frames, start_time, end_time, label])
    # sort segments by start_time
    for vid in segment:
        segment[vid].sort(key=lambda x: x[0])

    return segment
            
'''
    "video_id" : 1
    "video_start": 502,
    "video_end": 611,
    "anomaly_start": 33,
    "anomaly_end": 63,
    "anomaly_class": "other: turning",
    "num_frames": 110,
    "subset": "train"
'''