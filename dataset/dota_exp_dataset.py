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

# os.s
# os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
# pandarallel.initialize(nb_workers=30,progress_bar=True,verbose=2,use_memory_fs = False )

ROOT_DIR = os.path.abspath('./')
sys.path.append(ROOT_DIR)

from utils import segment_iou

# dataloader for training
class DoTADataset(Dataset):
    def __init__(self, meta_data, frames_path, feats_path,
                 slide_window_size, kernel_list, classes,
                 pos_thresh, neg_thresh, stride_factor, save_samplelist=False,
                 load_samplelist=False, sample_listpath=None):
        super(DoTADataset, self).__init__()
        video_list = set(os.listdir(frames_path))
        segment_data = generate_segment_exp(meta_data, classes, frames_path) 

        '''
            'v_id' : [num_frames, start, end, class]
            {'0RJPQ_97dcs_000199': [[100, 40, 70, 3]], '0RJPQ_97dcs_000307': [[110, 23, 53, 9]]
        '''

        self.slide_window_size = slide_window_size
        # all the anchors
        anc_len_lst = []
        anc_cen_lst = []
        self.sample_list = []
        
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
        
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = [None]*len(meta_data)
            vid_idx = 0
            
            for vid, val in meta_data.items():
                if vid in video_list:
                    segement_video = np.array(segment_data[vid])
                    vid = os.path.join(feats_path, vid)
                    results[vid_idx] = pool.apply_async(_get_pos_neg_exp,(vid, segement_video, slide_window_size, anc_len_all, anc_cen_all, pos_thresh, neg_thresh))
                    vid_idx += 1
            results = results[:vid_idx]
            for i, r in enumerate(results):
                results[i] = r.get()

        vid_counter = 0
        for r in results:
            if r is not None:
                vid_counter += 1
                video_id, total_frame, pos_seg, neg_seg = r
                print("Postivit segment > ", np.array(pos_seg).shape)

                anno_class = pos_seg[0][-1]
                positive_offsets = [[*off[:-1]] for off in pos_seg]
                self.sample_list.append((video_id, positive_offsets, anno_class, neg_seg, total_frame))
                
                pos_anchor_stats.append(len(pos_seg))
                neg_anchor_stats.append(len(neg_seg))

        print('total number of {} videos: '.format(vid_counter))
        print('total number of {} samples '.format(len(self.sample_list)))
        print('avg pos anc: {:.2f} avg neg anc: {:.2f}'.format(
            np.mean(pos_anchor_stats), np.mean(neg_anchor_stats)
        ))

        if save_samplelist:
            with open(sample_listpath, 'wb') as f:
                pickle.dump(self.sample_list, f)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        video_id, pos_seg, ann_class_id, neg_seg, total_frame = self.sample_list[index]
        vgg_feats = torch.from_numpy(np.load(video_id + '.npy')).float()
        return (pos_seg, ann_class_id, neg_seg, vgg_feats)


def _get_pos_neg_exp(vid, segment_video, slide_window_size, anc_len_all, anc_cen_all, pos_thresh, neg_thresh):
    
    
    neg_overlap = [0] * anc_len_all.shape[0]
    pos_collected = [False] * anc_len_all.shape[0]
    pos_seg = []

    vgg_feats = torch.from_numpy(np.load(vid + '.npy')).float()
    total_frame = vgg_feats.size(0)
    # if total_frame != segment_video[:, 0]:
        # print("Here**** \n feat_size {} \t seg_frame {}".format(total_frame, segment_video[:, 0]))
    window_start = 0
    window_end = slide_window_size
    window_start_t = window_start
    window_end_t = window_end
    
    for j in range(anc_len_all.shape[0]):
        potential_match = []
        gt_start = segment_video[0][1]
        gt_end = segment_video[0][2]
        # print("J > {} >>> GT start: {} \t end {} anc_center {} anc_lenght {}".format(j,gt_start, gt_end, anc_cen_all[j], anc_len_all[j]))
        if gt_start > gt_end:
            gt_start, gt_end = gt_end, gt_start
        if anc_cen_all[j] + anc_len_all[j] / 2. <= total_frame:
            if window_start_t <= gt_start and window_end_t >= gt_end:
                
                overlap = segment_iou(np.array([gt_start, gt_end]), np.array([[
                    anc_cen_all[j] - anc_len_all[j] / 2.,
                    anc_cen_all[j] + anc_len_all[j] / 2.]]))
                    
                neg_overlap[j] = max(overlap, neg_overlap[j])

                if not pos_collected[j] and overlap >= pos_thresh:
                    len_offset = math.log((gt_end - gt_start) / anc_len_all[j])
                    cen_offset = ((gt_end + gt_start) / 2. - anc_cen_all[j]) / anc_len_all[j]
                    potential_match.append((j, overlap, len_offset, cen_offset, segment_video[0][3]))
                    pos_collected[j] = True

        # print("potential match > ", np.array(potential_match).shape)
        filled = False
        for item in potential_match:
            pos_seg.append(tuple(item))
            break

        if not filled and len(potential_match)>0:
            # randomly choose one
            shuffle(potential_match)
            item = potential_match[0]
            pos_seg.append(tuple(item))

    neg_seg = []
    for oi, overlap in enumerate(neg_overlap):
        if overlap < neg_thresh:
            neg_seg.append((oi, overlap))

    print(
        'pos anc: {}, neg anc: {}'.format(len(pos_seg), len(neg_seg)))

    return vid, total_frame, pos_seg, neg_seg
    
    # ================================================================================================
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

def dota_collate_fn(batch_lst):
    sample_each = 10  # TODO, hard coded
    pos_seg, anno_class, neg_seg, img_feat = batch_lst[0]

    batch_size = len(batch_lst)

    anno_class_batch = torch.LongTensor(np.ones((batch_size, 1),dtype='int64'))

    tempo_seg_pos = torch.FloatTensor(np.zeros((batch_size, sample_each, 4)))
    tempo_seg_neg = torch.FloatTensor(np.zeros((batch_size, sample_each, 2)))
    for batch_idx in range(batch_size):
 
        pos_seg, anno_class, neg_seg, img_feat = batch_lst[batch_idx]
        img_batch = torch.FloatTensor(np.zeros((batch_size,
                                img_feat.size(0),
                                img_feat.size(1))))

        img_batch[batch_idx,:] = img_feat

        pos_seg_tensor = torch.FloatTensor(pos_seg)
        neg_seg_tensor = torch.FloatTensor(neg_seg)
        anno_class_batch[batch_idx] = anno_class
        
        # sample positive anchors
        perm_idx = torch.randperm(len(pos_seg))
        
        if len(pos_seg) >= sample_each:
            tempo_seg_pos[batch_idx,:,:] = pos_seg_tensor[perm_idx[:sample_each]]

        else:
            tempo_seg_pos[batch_idx,:len(pos_seg),:] = pos_seg_tensor
            idx = torch.multinomial(torch.ones(len(pos_seg)), sample_each-len(pos_seg), True)
            tempo_seg_pos[batch_idx,len(pos_seg):,:] = pos_seg_tensor[idx]

        # sample negative anchors
        # neg_seg_tensor = torch.FloatTensor(neg_seg)
        perm_idx = torch.randperm(len(neg_seg))
        if len(neg_seg) >= sample_each:
            tempo_seg_neg[batch_idx, :, :] = neg_seg_tensor[perm_idx[:sample_each]]
        else:
            tempo_seg_neg[batch_idx, :len(neg_seg), :] = neg_seg_tensor
            idx = torch.multinomial(torch.ones(len(neg_seg)),
                                    sample_each - len(neg_seg),True)
            tempo_seg_neg[batch_idx, len(neg_seg):, :] = neg_seg_tensor[idx]

    return (img_batch, tempo_seg_pos, tempo_seg_neg, anno_class_batch)
     
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