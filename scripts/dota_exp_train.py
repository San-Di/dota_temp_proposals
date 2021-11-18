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
from torch.autograd import Variable

# misc

BASE_DIR = os.path.abspath('../')
ROOT_DIR = os.path.abspath('./')
sys.path.append(ROOT_DIR)
print(os.path.abspath(os.getcwd()))

from dataset.dota_exp_dataset import DoTADataset, generate_classes, dota_collate_fn
from models.dota_proposal import DotaProposalModel

seed = 213

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.cuda.manual_seed_all(seed)


meta_train_path = os.path.join(BASE_DIR, 'Detection-of-Traffic-Anomaly/dataset/metadata_train.json')

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

mask_weight = 0.0
gated_mask = False
sample_prob = 0
distributed = True
enable_visdom = False
scst_weight = 0.0
cls_weight =1.0
reg_weight =10
sent_weight =0.25
grad_norm = 1
# ------ x ------ x ------

batch_size = 3
num_workers = 8
world_size = 3
learning_rate = 0.1
alpha = 0.95
beta = 0.999
epsilon = 1e-8
max_epochs = 20
reduce_factor = 0.5
patience_epoch = 1
save_checkpoint_every = 1
checkpoint_path = os.path.join(ROOT_DIR, 'checkpoints')

dist_url = '../dataset/dist_file' # non exist url for distributed url
dist_backend = 'gloo'

def get_dataset():

    # Process classes
    meta_data = json.load(open(meta_train_path))
    classes = generate_classes(meta_data)
    # classes = json.load(open(os.path.join(ROOT_DIR, 'scripts/classes.json')))
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

    dist.init_process_group(backend=dist_backend, init_method='env://',
                            world_size=world_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True, sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=dota_collate_fn
                              )

    return train_loader, train_sampler, classes

def get_model(classes):
    
    model = DotaProposalModel(d_model=args.d_model,
                               d_hidden=args.d_hidden,
                               n_layers=args.n_layers,
                               n_heads=args.n_heads,
                               vocab=sent_vocab,
                               in_emb_dropout=args.in_emb_dropout,
                               attn_dropout=args.attn_dropout,
                               vis_emb_dropout=args.vis_emb_dropout,
                               cap_dropout=args.cap_dropout,
                               nsamples=args.train_sample,
                               kernel_list = kernel_list,
                               stride_factor = stride_factor,
                               learn_mask = mask_weight>0)

    # Ship the model to GPU, maybe
    if distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[3,4], output_device=4)
    else:
        model = torch.nn.DataParallel(model).cuda()

    return model


### Training the network ###
def train(epoch, model, optimizer, train_loader, vis, vis_window):
    model.train() # training mode
    train_loss = []
    nbatches = len(train_loader)
    t_iter_start = time.time()

    sample_prob = min(sample_prob, int(epoch/5)*0.05)
    for train_iter, data in enumerate(train_loader):
        (img_batch, tempo_seg_pos, tempo_seg_neg, sentence_batch) = data
        img_batch = Variable(img_batch)
        tempo_seg_pos = Variable(tempo_seg_pos)
        tempo_seg_neg = Variable(tempo_seg_neg)
        sentence_batch = Variable(sentence_batch)
    
        img_batch = img_batch.cuda()
        tempo_seg_neg = tempo_seg_neg.cuda()
        tempo_seg_pos = tempo_seg_pos.cuda()
        sentence_batch = sentence_batch.cuda()

        t_model_start = time.time()
        (pred_score, gt_score,
        pred_offsets, gt_offsets,
        pred_sentence, gt_sent,
         scst_loss, mask_loss) = model(img_batch, tempo_seg_pos,
                                       tempo_seg_neg, sentence_batch,
                                       sample_prob, stride_factor,
                                       scst = scst_weight > 0,
                                       gated_mask= gated_mask)

        cls_loss = model.bce_loss(pred_score, gt_score) * cls_weight
        reg_loss = model.reg_loss(pred_offsets, gt_offsets) * reg_weight
        sent_loss = F.cross_entropy(pred_sentence, gt_sent) * sent_weight

        total_loss = cls_loss + reg_loss + sent_loss

        if scst_loss is not None:
            scst_loss *= scst_weight
            total_loss += scst_loss

        if mask_loss is not None:
            mask_loss = mask_weight * mask_loss
            total_loss += mask_loss
        else:
            mask_loss = cls_loss.new(1).fill_(0)

        optimizer.zero_grad()
        total_loss.backward()

        # enable the clipping for zero mask loss training
        total_grad_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                         grad_norm)

        optimizer.step()

        train_loss.append(total_loss.data.item())

        if enable_visdom:
            if vis_window['iter'] is None:
                if not distributed or (
                    distributed and dist.get_rank() == 0):
                    vis_window['iter'] = vis.line(
                        X=np.arange(epoch*nbatches+train_iter, epoch*nbatches+train_iter+1),
                        Y=np.asarray(train_loss),
                        opts=dict(title='Training Loss',
                                  xlabel='Training Iteration',
                                  ylabel='Loss')
                    )
            else:
                if not distributed or (
                    distributed and dist.get_rank() == 0):
                    vis.line(
                        X=np.arange(epoch*nbatches+train_iter, epoch*nbatches+train_iter+1),
                        Y=np.asarray([np.mean(train_loss)]),
                        win=vis_window['iter'],
                        opts=dict(title='Training Loss',
                                  xlabel='Training Iteration',
                                  ylabel='Loss'),
                        update='append'
                    )

        t_model_end = time.time()
        print('iter: [{}/{}], training loss: {:.4f}, '
              'class: {:.4f}, '
              'reg: {:.4f}, sentence: {:.4f}, '
              'mask: {:.4f}, '
              'grad norm: {:.4f} '
              'data time: {:.4f}s, total time: {:.4f}s'.format(
            train_iter, nbatches, total_loss.data.item(), cls_loss.data.item(),
            reg_loss.data.item(), sent_loss.data.item(), mask_loss.data.item(),
            total_grad_norm,
            t_model_start - t_iter_start,
            t_model_end - t_iter_start
        ), end='\r')

        t_iter_start = time.time()

    return np.mean(train_loss)


### Validation ##
def valid(model, loader):
    model.eval()
    valid_loss = []
    val_cls_loss = []
    val_reg_loss = []
    val_sent_loss = []
    val_mask_loss = []
    for iter, data in enumerate(loader):
        (img_batch, tempo_seg_pos, tempo_seg_neg, sentence_batch) = data
        with torch.no_grad():
            img_batch = Variable(img_batch)
            tempo_seg_pos = Variable(tempo_seg_pos)
            tempo_seg_neg = Variable(tempo_seg_neg)
            sentence_batch = Variable(sentence_batch)

            img_batch = img_batch.cuda()
            tempo_seg_neg = tempo_seg_neg.cuda()
            tempo_seg_pos = tempo_seg_pos.cuda()
            sentence_batch = sentence_batch.cuda()

            (pred_score, gt_score,
             pred_offsets, gt_offsets,
             pred_sentence, gt_sent,
             _, mask_loss) = model(img_batch, tempo_seg_pos,
                                    tempo_seg_neg, sentence_batch,
                                    stride_factor=stride_factor,
                                    gated_mask=gated_mask)

            cls_loss = model.bce_loss(pred_score, gt_score) * cls_weight
            reg_loss = model.reg_loss(pred_offsets, gt_offsets) * reg_weight
            sent_loss = F.cross_entropy(pred_sentence, gt_sent) * sent_weight

            total_loss = cls_loss + reg_loss + sent_loss

            if mask_loss is not None:
                mask_loss = mask_weight * mask_loss
                total_loss += mask_loss
            else:
                mask_loss = cls_loss.new(1).fill_(0)

            valid_loss.append(total_loss.data.item())
            val_cls_loss.append(cls_loss.data.item())
            val_reg_loss.append(reg_loss.data.item())
            val_sent_loss.append(sent_loss.data.item())
            val_mask_loss.append(mask_loss.data.item())

    return (np.mean(valid_loss), np.mean(val_cls_loss),
            np.mean(val_reg_loss), np.mean(val_sent_loss), np.mean(val_mask_loss))


if __name__ == "__main__":

    train_loader, train_sampler, classes  = get_dataset()
    
    # model = get_model(classes)

    # # filter params that don't require gradient (credit: PyTorch Forum issue 679)
    # # smaller learning rate for the decoder

    # optimizer = optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     learning_rate, betas=(alpha, beta), eps=epsilon)

    # # learning rate decay every 1 epoch
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=reduce_factor,
    #                                            patience=patience_epoch,
    #                                            verbose=True)

    # # Number of parameter blocks in the network
    # print("# of param blocks: {}".format(str(len(list(model.parameters())))))

    # best_loss = float('inf')

    # if enable_visdom:
    #     import visdom
    #     vis = visdom.Visdom()
    #     vis_window={'iter': None,
    #                 'loss': None}
    # else:
    #     vis, vis_window = None, None

    # all_eval_losses = []
    # all_cls_losses = []
    # all_reg_losses = []
    # all_sent_losses = []
    # all_mask_losses = []
    # all_training_losses = []

    # for train_epoch in range(max_epochs):
    #     t_epoch_start = time.time()
    #     print('Epoch: {}'.format(train_epoch))
    #     # distributed is assumed to be true
    #     train_sampler.set_epoch(train_epoch)

    #     epoch_loss = train(train_epoch, model, optimizer, train_loader,
    #                        vis, vis_window)
    #     all_training_losses.append(epoch_loss)

    #     (valid_loss, val_cls_loss, val_reg_loss, val_sent_loss, val_mask_loss) = valid(model, valid_loader)

    #     all_eval_losses.append(valid_loss)
    #     all_cls_losses.append(val_cls_loss)
    #     all_reg_losses.append(val_reg_loss)
    #     all_sent_losses.append(val_sent_loss)
    #     all_mask_losses.append(val_mask_loss)

    #     if enable_visdom:
    #         if vis_window['loss'] is None:
    #             if not distributed or (distributed and dist.get_rank() == 0):
    #                 vis_window['loss'] = vis.line(
    #                 X=np.tile(np.arange(len(all_eval_losses)),
    #                           (6,1)).T,
    #                 Y=np.column_stack((np.asarray(all_training_losses),
    #                                    np.asarray(all_eval_losses),
    #                                    np.asarray(all_cls_losses),
    #                                    np.asarray(all_reg_losses),
    #                                    np.asarray(all_sent_losses),
    #                                    np.asarray(all_mask_losses))),
    #                 opts=dict(title='Loss',
    #                           xlabel='Validation Iter',
    #                           ylabel='Loss',
    #                           legend=['train',
    #                                   'dev',
    #                                   'dev_cls',
    #                                   'dev_reg',
    #                                   'dev_sentence',
    #                                   'dev_mask']))
    #         else:
    #             if not distributed or (
    #                 distributed and dist.get_rank() == 0):
    #                 vis.line(
    #                 X=np.tile(np.arange(len(all_eval_losses)),
    #                           (6, 1)).T,
    #                 Y=np.column_stack((np.asarray(all_training_losses),
    #                                    np.asarray(all_eval_losses),
    #                                    np.asarray(all_cls_losses),
    #                                    np.asarray(all_reg_losses),
    #                                    np.asarray(all_sent_losses),
    #                                    np.asarray(all_mask_losses))),
    #                 win=vis_window['loss'],
    #                 opts=dict(title='Loss',
    #                           xlabel='Validation Iter',
    #                           ylabel='Loss',
    #                           legend=['train',
    #                                   'dev',
    #                                   'dev_cls',
    #                                   'dev_reg',
    #                                   'dev_sentence',
    #                                   'dev_mask']))

    #     if valid_loss < best_loss:
    #         best_loss = valid_loss
    #         if (distributed and dist.get_rank() == 0) or not distributed:
    #             torch.save(model.module.state_dict(), os.path.join(checkpoint_path, 'best_model.t7'))
    #         print('*'*5)
    #         print('Better validation loss {:.4f} found, save model'.format(valid_loss))

    #     # save eval and train losses
    #     if (distributed and dist.get_rank() == 0) or not distributed:
    #         torch.save({'train_loss':all_training_losses,
    #                     'eval_loss':all_eval_losses,
    #                     'eval_cls_loss':all_cls_losses,
    #                     'eval_reg_loss':all_reg_losses,
    #                     'eval_sent_loss':all_sent_losses,
    #                     'eval_mask_loss':all_mask_losses,
    #                     }, os.path.join(checkpoint_path, 'model_losses.t7'))

    #     # learning rate decay
    #     scheduler.step(valid_loss)

    #     # validation/save checkpoint every a few epochs
    #     if train_epoch % save_checkpoint_every == 0 or train_epoch == max_epochs:
    #         if (distributed and dist.get_rank() == 0) or not distributed:
    #             torch.save(model.module.state_dict(), os.path.join(checkpoint_path, 'model_epoch_{}.t7'.format(train_epoch)))

    #     print('-'*80)
    #     print('Epoch {} summary'.format(train_epoch))
    #     print('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
    #         epoch_loss, valid_loss, time.time()-t_epoch_start
    #     ))
    #     print('val_cls: {:.4f}, '
    #           'val_reg: {:.4f}, val_sentence: {:.4f}, '
    #           'val mask: {:.4f}'.format(
    #         val_cls_loss, val_reg_loss, val_sent_loss, val_mask_loss
    #     ))
    #     print('-'*80)
    
