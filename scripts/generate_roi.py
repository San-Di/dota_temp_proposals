# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
import copy
import json
import pickle
import subprocess
import numpy as np
import cv2
from utils import *
FPS = 10
# LENGTH = 768
# WINS = [LENGTH * 8]
#LENGTH = 192
#WINS = [LENGTH * 32]
min_length = 3
overlap_thresh = 0.7
# STEP = LENGTH / 4
META_FILE = '../../Detection-of-Traffic-Anomaly/dataset/metadata_train.json'
data = json.load(open(META_FILE))
FRAME_DIR = '../../Detection-of-Traffic-Anomaly/dataset/frames/'

classes = generate_classes(data) 

segments = generate_segment_exp(data, classes, FRAME_DIR) 


def generate_roi(rois, video, start, end, stride):
    tmp = {}
    tmp['wins'] = ( rois[:,:2] - start ) / stride
    tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0]
    tmp['gt_classes'] = rois[:,2]
    tmp['max_classes'] = rois[:,2]
    tmp['max_overlaps'] = np.ones(len(rois))
    tmp['flipped'] = False
    tmp['frames'] = np.array([[0, start, end, stride]])
    tmp['bg_name'] = os.path.join(FRAME_DIR, video)
    tmp['fg_name'] = os.path.join(FRAME_DIR, video)
    # if not os.path.isfile(os.path.join(FRAME_DIR, video, 'image_' + str(end-1).zfill(5) + '.jpg')):
    #     print (os.path.join(FRAME_DIR, video, 'image_' + str(end-1).zfill(5) + '.jpg'))
        # raise
    return tmp

def generate_roidb(segment):
    video_list = set(os.listdir(FRAME_DIR))
    duration = []
    roidb = []
    for vid in segment:
        if vid in video_list:
            # length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))
            
            db = np.array(segment[vid])
            length = db[:,0]
            if len(db) == 0:
                continue
            # db[:,:2] = db[:,:2] * FPS
            LENGTH = length
            WINS = [LENGTH * 8]
            STEP = LENGTH / 4
            # print("Len {} Wins {}".format(length, WINS))
            for win in WINS:
                stride = int(win / LENGTH)
                step = int(stride * STEP)
                
                # Forward Direction
                for start in range(0, max(1, length - win + 1), step):
                    end = min(start + win, length)
                    assert end <= length
                    # No overlap between gt and dt
                    rois = db[np.logical_not(np.logical_or(db[:,1] >= end, db[:,2] <= start))]
                    # print("Len: {} start: {} end: {} win: {} rois: {}".format(length, start, end, win, rois))
                    # Remove duration less than min_length
                    if len(rois) > 0:
                        duration = rois[:,2] - rois[:,1]
                        rois = rois[duration >= min_length]
                        # print("Len: {} start: {} end: {} win: {} rois: {}".format(length, start, end, win, rois))
                    # Remove overlap(for gt) less than overlap_thresh
                    if len(rois) > 0:
                        time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
                        overlap = time_in_wins / (rois[:,1] - rois[:,0])
                        assert min(overlap) >= 0
                        assert max(overlap) <= 1
                        rois = rois[overlap >= overlap_thresh]
                        # print("Len: {} start: {} end: {} win: {} step: {}".format(length, start, end, win, step))
                    # Append data
                    if len(rois) > 0:
                        rois[:,0] = np.maximum(start, rois[:,0])
                        rois[:,1] = np.minimum(end, rois[:,1])
                        tmp = generate_roi(rois, vid, start, end, stride)
                        roidb.append(tmp)
                        if USE_FLIPPED:
                               flipped_tmp = copy.deepcopy(tmp)
                               flipped_tmp['flipped'] = True
                               roidb.append(flipped_tmp)

                # # Backward Direction
                win = win[0]
                for end in range(length[0], win-1, step): 
                    start = end - win
                    # print("End > {} \t win > {} \t start {} ".format(end, win, start))
                    # assert start > 0
                    rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] <= start))]

                    # Remove duration less than min_length
                    if len(rois) > 0:
                        duration = rois[:,1] - rois[:,0]
                        rois = rois[duration > min_length]

                    # Remove overlap less than overlap_thresh
                    if len(rois) > 0:
                        time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
                        overlap = time_in_wins / (rois[:,1] - rois[:,0])
                        assert min(overlap) >= 0
                        assert max(overlap) <= 1
                        rois = rois[overlap > overlap_thresh]

                    # Append data
                    if len(rois) > 0:
                        rois[:,0] = np.maximum(start, rois[:,0])
                        rois[:,1] = np.minimum(end, rois[:,1])
                        tmp = generate_roi(rois, vid, start, end, stride)
                        roidb.append(tmp)
                    if USE_FLIPPED:
                           flipped_tmp = copy.deepcopy(tmp)
                           flipped_tmp['flipped'] = True
                           roidb.append(flipped_tmp)

    return roidb


USE_FLIPPED = True      
train_roidb = generate_roidb( segments)
# print (train_roidb)
print ("Save dictionary")
myfile = open('test.txt', 'w')
myfile.write("%s"%train_roidb)
myfile.close()
pickle.dump(train_roidb, open('train_data_{}fps_flipped.pkl'.format(FPS),'wb'), pickle.HIGHEST_PROTOCOL)
