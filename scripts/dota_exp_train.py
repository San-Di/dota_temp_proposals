"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# general packages
import os
import sys
import errno
import numpy as np
import random
import time

import json
# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.utils.data.distributed

# misc

BASE_DIR = os.path.abspath('../')
ROOT_DIR = os.path.abspath('./')
sys.path.append(ROOT_DIR)
print(os.path.abspath(os.getcwd()))

from dataset.dota_exp_dataset import DoTADataset, generate_classes, dota_collate_fn

seed = 213

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.cuda.manual_seed_all(seed)


meta_data_path = os.path.join(BASE_DIR, 'Detection-of-Traffic-Anomaly/dataset/metadata_train.json')
frames_path = os.path.join(BASE_DIR, 'Detection-of-Traffic-Anomaly/dataset/frames/')
feats_path = os.path.join(BASE_DIR,'Detection-of-Traffic-Anomaly/dataset/features_self/') 

slide_window_size = 280
kernel_list = [1, 2, 3, 4, 5, 7, 9, 11, 15, 21, 29, 41, 57, 71, 111, 161, 211, 251]
pos_thresh = 0.7
neg_thresh = 0.3
stride_factor = 50
save_samplelist = False
load_samplelist = False
sample_listpath = None

# ------ x ------ x ------

batch_size = 3
num_workers = 8
world_size = 2
dist_url = '../dataset/dist_file' # non exist url for distributed url
dist_backend = 'gloo'


def get_dataset():

    # Process classes
    meta_data = json.load(open(meta_data_path))
    # classes = generate_classes(meta_data)
    classes = json.load(open(os.path.join(ROOT_DIR, 'scripts/classes.json')))
    print(classes)
    '''
    (self, meta_data_path, frames_path, 
    slide_window_size, kernel_list, classes,
    pos_thresh, neg_thresh, stride_factor, save_samplelist=False,
    load_samplelist=False, sample_listpath=None)
    '''
    # Create the dataset and data loader instance
    train_dataset = DoTADataset(meta_data, frames_path,
                        feats_path,
                        slide_window_size,
                        kernel_list,
                        classes,
                        pos_thresh,
                        neg_thresh,
                        stride_factor,
                        save_samplelist,
                        load_samplelist,
                        sample_listpath)

    # train_dataset = train_dataset.to('cuda')
    # dist.init_process_group(backend=dist_backend, init_method=dist_url,
    #                         world_size=world_size)
    train_sampler = None
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=dota_collate_fn
                              )

    return train_loader

if __name__ == "__main__":
    # train_loader, sampler, classes  = get_dataset()
    get_dataset()
