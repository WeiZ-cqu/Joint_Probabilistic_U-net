#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import shutil
from scipy.optimize import linear_sum_assignment
from torch.distributions import Normal, Independent, kl

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def save_checkpoint(state, is_test_best, is_val_best, filename, besttestname, bestvalname):
    torch.save(state, filename)
    if is_test_best:
        shutil.copyfile(filename, besttestname)
    if is_val_best:
        shutil.copyfile(filename, bestvalname)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def organize_image(predictions, groundtruths, image, size):
    # first row : image
    # second row : groundtruths
    # other row : predictions
    IM = np.zeros((128*size[0], 128*size[1]))
    IM[0:128, 0:128] = image.cpu().numpy()
    row=2
    for i in range(len(groundtruths)):
        IM[128*(row-1):128*row, 128*i:128*(i+1)] = groundtruths[i].cpu().numpy()
    row=row+1
    t=0
    for i in range(len(predictions)):
        if i % size[1] == 0 and i != 0:
            row=row+1
            t=0
        IM[128*(row-1):128*row, 128*t:128*(t+1)] = predictions[i].cpu().numpy()
        t=t+1
    return IM

def organize_color_image(predictions, groundtruths, image, size, imsize=256):
    # first row : image
    # second row : groundtruths
    # other row : predictions
    IM = np.zeros((imsize*size[0], imsize*size[1], 3))
    IM[0:imsize, 0:imsize, :] = image.cpu().numpy()
    row=2
    for i in range(len(groundtruths)):
        IM[imsize*(row-1):imsize*row, imsize*i:imsize*(i+1), :] = groundtruths[i].cpu().numpy()
    row=row+1
    t=0
    for i in range(len(predictions)):
        if i % size[1] == 0 and i != 0:
            row=row+1
            t=0
        IM[imsize*(row-1):imsize*row, imsize*t:imsize*(t+1), :] = predictions[i].cpu().numpy()
        t=t+1
    return IM

def organize_style_image(predictions, groundtruths, size, imsize=256):
    # first row : image
    # second row : groundtruths
    # other row : predictions
    IM = np.zeros((imsize*size[0], imsize*size[1], 3))
    row=1
    for i in range(len(groundtruths)):
        IM[imsize*(row-1):imsize*row, imsize*i:imsize*(i+1), :] = groundtruths[i].cpu().numpy()
    row=row+1
    t=0
    for i in range(len(predictions)):
        if i % size[1] == 0 and i != 0:
            row=row+1
            t=0
        IM[imsize*(row-1):imsize*row, imsize*t:imsize*(t+1), :] = predictions[i].cpu().numpy()
        t=t+1
    return IM

def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.sigmoid(output)
    if weights is not None:
        assert len(weights) == 2, "length of weights is invalid!"
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(loss)

def tile(device, a, dim, n_tile):
    """
    This function is taken form PyTorch forum and mimics the behavior of tf.tile.
    Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) 
                    + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)

def sample_latent(device, dist, latent_dim, n_sample, size, mode):
    if mode == 'rsample':
        samples = dist.rsample([n_sample]).permute(1,0,2,3,4)
    elif mode == 'sample':
        samples = dist.sample([n_sample]).permute(1,0,2,3,4)
    samples = samples.reshape(-1, n_sample * latent_dim, samples.size(3), samples.size(4))
    
    samples = tile(device, samples, 2, size)
    samples = tile(device, samples, 3, size)
    
    return samples

def mask_IoU(prediction, groundtruth):
    prediction = prediction.detach().cpu().numpy()
    groundtruth = groundtruth.detach().cpu().numpy()
    intersection = np.logical_and(groundtruth, prediction)
    union = np.logical_or(groundtruth, prediction)
    if np.sum(union) == 0:
        return 1
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def generalized_energy_distance(samples, groundtruths):
    D1 = 0
    for i in range(len(samples)):
        for j in range(len(groundtruths)):
            D1 += 1 - mask_IoU(samples[i], groundtruths[j])
    D2 = 0
    for i in range(len(samples)):
        for j in range(len(samples)):
            D2 += 1 - mask_IoU(samples[i], samples[j])
    D3 = 0
    for i in range(len(groundtruths)):
        for j in range(len(groundtruths)):
            D3 += 1 - mask_IoU(groundtruths[i], groundtruths[j])
            
    n, m = len(samples), len(groundtruths)
    D = 2/(n*m)*D1 - 1/(n*n)*D2 - 1/(m*m)*D3
    return D, 1/(n*n)*D2

def HM_IoU(samples, groundtruths):
    groundtruths = groundtruths * (len(samples) // len(groundtruths))
    assert len(groundtruths) == len(samples)
    cost = np.zeros((len(samples), len(groundtruths)))
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            cost[i, j] = mask_IoU(samples[i], groundtruths[j])
    row_ind, col_ind = linear_sum_assignment(-cost)
    return np.mean(cost[row_ind, col_ind])

def mask_Dice(prediction, groundtruth):
    prediction = prediction.detach().cpu().numpy()
    groundtruth = groundtruth.detach().cpu().numpy()
    intersection = np.logical_and(groundtruth, prediction)
    union = np.logical_or(groundtruth, prediction)
    if np.sum(union) + np.sum(intersection) == 0:
        return 1
    dice_score = 2*np.sum(intersection) / (np.sum(union) + np.sum(intersection))
    return dice_score

def mean_dice(samples, groundtruths):
    dice = 0
    for i in range(len(samples)):
        for j in range(len(groundtruths)):
            dice += mask_Dice(samples[i], groundtruths[j])
    return dice / (len(samples) * len(groundtruths))

def ncc(a,v, zero_norm=True):

    a = a.flatten().detach().cpu().numpy()
    v = v.flatten().detach().cpu().numpy()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a, v)

def variance_ncc_dist(samples, groundtruths):
    
    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):
        
        log_samples_p = torch.log(m_samp + eps)
        log_samples_n = torch.log((1-m_samp) + eps)
        
        return -1.0*(m_gt*log_samples_p + (1-m_gt)*log_samples_n)
    
    tes_samples = torch.stack(samples, dim=0) # N x H x W
    tes_groundtruths = torch.stack(groundtruths, dim=0) # M x H x W
    mean_seg = torch.mean(tes_samples, dim=0)
    
    N = tes_samples.size(0)
    M = tes_groundtruths.size(0)
    
    sX = tes_samples.size(1)
    sY = tes_samples.size(2)
    
    E_ss_arr = torch.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(tes_samples[i,...], mean_seg)
        # print('pixel wise xent')
        # plt.imshow( E_ss_arr[i,...])
        # plt.show()
    E_ss = torch.mean(E_ss_arr, dim=0)

    E_sy_arr = torch.zeros((M, N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(tes_samples[i,...], tes_groundtruths[j,...])
    E_sy = torch.mean(E_sy_arr, dim=1)

    ncc_list = []
    for j in range(M):

        ncc_list.append(ncc(E_ss, E_sy[j,...]))

    return (1/M)*sum(ncc_list)
    
    
    
    
    
    
        