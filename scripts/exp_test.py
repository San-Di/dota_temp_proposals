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

'''
    'vid' : [num_frames, start, end, class]
    {'0RJPQ_97dcs_000199': [[100, 40, 70, 3]], '0RJPQ_97dcs_000307': [[110, 23, 53, 9]]
'''

def generate_roidb(segment):
    video_list = set(os.listdir(FRAME_DIR))
    duration = []
    roidb = []
    for vid in segment:
        if vid in video_list:
            length = len(os.listdir(os.path.join(FRAME_DIR, vid)))
            ground_truth = np.array(segment[vid])
            if len(ground_truth) == 0:
                continue
            #  db[:,:2] = db[:,:2] * FPS  ******* No need to multiply with FPS becoz DoTA dataset annotate segement in frames unit ( Not second unit )
            LENGTH = ground_truth[0][0]
            WINS = [LENGTH * 8]
            STEP = LENGTH / 4
            for win in WINS:
                stride = int(win / LENGTH)
                step = int(stride * STEP)
                print("num_frames > {} \t windows > {} \t, stride > {} \t, step > {}".format(LENGTH, win, stride, step))
                print("len > {} \t Length > {}".format(length, LENGTH))
                # Forward Direction
                # for start in range(0, max(1, length - win + 1), step):
                #     end = min(start + win, length)
                #     assert end <= length
                #     # No overlap between gt and dt
                #     rois = ground_truth[np.logical_not(np.logical_or(ground_truth[:,0] >= end, ground_truth[:,1] <= start))]

                #     # Remove duration less than min_length
                #     if len(rois) > 0:
                #         duration = rois[:,1] - rois[:,0]
                #         rois = rois[duration >= min_length]

                #     # Remove overlap(for gt) less than overlap_thresh
                #     if len(rois) > 0:
                #         time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
                #         overlap = time_in_wins / (rois[:,1] - rois[:,0])
                #         assert min(overlap) >= 0
                #         assert max(overlap) <= 1
                #         rois = rois[overlap >= overlap_thresh]

                #     # Append data
                #     if len(rois) > 0:
                #         rois[:,0] = np.maximum(start, rois[:,0])
                #         rois[:,1] = np.minimum(end, rois[:,1])
                #         tmp = generate_roi(rois, vid, start, end, stride, split)
                #         roidb.append(tmp)
                #         if USE_FLIPPED:
                #                flipped_tmp = copy.deepcopy(tmp)
                #                flipped_tmp['flipped'] = True
                #                roidb.append(flipped_tmp)

                # # Backward Direction
                # for end in range(length, win-1, - step):
                #     start = end - win
                #     assert start >= 0
                #     rois = ground_truth[np.logical_not(np.logical_or(ground_truth[:,0] >= end, ground_truth[:,1] <= start))]

                #     # Remove duration less than min_length
                #     if len(rois) > 0:
                #         duration = rois[:,1] - rois[:,0]
                #         rois = rois[duration > min_length]

                #     # Remove overlap less than overlap_thresh
                #     if len(rois) > 0:
                #         time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
                #         overlap = time_in_wins / (rois[:,1] - rois[:,0])
                #         assert min(overlap) >= 0
                #         assert max(overlap) <= 1
                #         rois = rois[overlap > overlap_thresh]

                #     # Append data
                #     if len(rois) > 0:
                #         rois[:,0] = np.maximum(start, rois[:,0])
                #         rois[:,1] = np.minimum(end, rois[:,1])
                #         tmp = generate_roi(rois, vid, start, end, stride, split)
                #         roidb.append(tmp)
                #     if USE_FLIPPED:
                #            flipped_tmp = copy.deepcopy(tmp)
                #            flipped_tmp['flipped'] = True
                #            roidb.append(flipped_tmp)

    return roidb

generate_roidb(segments)
