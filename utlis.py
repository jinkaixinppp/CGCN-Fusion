import torch
import torch.nn as nn
import numpy as np
from skimage import feature
import random

from skimage.metrics import structural_similarity as ssim
from torch.nn import functional as F
from math import exp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    # 检查softmax得分是否包含NaN
    if torch.any(torch.isnan(ssim_map)):
        print('Warning: ssim_map contains NaN. Check the forward pass.')
        ssim_map = torch.where(torch.isnan(ssim_map), torch.zeros_like(ssim_map), ssim_map)

    ret = ssim_map.mean()

    return 1-ret


def new_fusion_rule1(feature_maps1,feature_maps2)->torch.FloatTensor:
    m1=(feature_maps1-feature_maps1.min())/(feature_maps1.max()-feature_maps1.min())
    m2=(feature_maps2-feature_maps2.min())/(feature_maps2.max()-feature_maps2.min())
    Feature_maps1=torch.sum(m1,dim=1,keepdim=True)
    Feature_maps2=torch.sum(m2,dim=1,keepdim=True)
    total_map=torch.cat([Feature_maps1,Feature_maps2],dim=1)
    total_map=torch.exp(total_map)/(torch.exp(total_map).sum(dim=1,keepdim=True))
    Map=feature_maps1*total_map[:,0:1]+feature_maps2*total_map[:,1:2]
    return Map


def l1_norm(matrix):
    """
    Calculate the L1 norm for some fusion strategies
    """
    return torch.abs(matrix).sum()


def fusion_strategy(f1, f2, device, strategy="average"):
    """
    f1: the extracted features of images 1
    f2: the extracted features of images 2
    strategy: 6 fusion strategy, including:
    "addition", "average", "FER", "L1NW", "AL1NW", "FL1N"
    addition strategy
    average strategy
    FER strategy: Feature Energy Ratio strategy
    L1NW strategy: L1-Norm Weight Strategy
    AL1NW strategy: Average L1-Norm Weight Strategy
    FL1N strategy: Feature L1-Norm Strategy

    Note:
    If the original image is PET or SPECT modal,
    it should be converted into YCbCr data, including Y1, Cb and Cr.
    """

    # The fused feature
    fused = torch.zeros_like(f1, device=device)
    if strategy == "addition":
        fused = f1 + f2
    elif strategy == "average":
        fused = (f1 + f2) / 2
    elif strategy == "FER":
        f_sum = (f1 ** 2 + f2 ** 2).clone()
        f_sum[f_sum == 0] = 1
        k1 = f1 ** 2 / f_sum
        k2 = f2 ** 2 / f_sum
        fused = k1 * f1 + k2 * f2
    elif strategy == "L1NW":
        l1 = l1_norm(f1)
        print(l1)
        l2 = l1_norm(f2)
        print(l2)
        fused = l1 * f1 + l2 * f2
    elif strategy == "AL1NW":
        p1 = l1_norm(f1) / 2
        p2 = l1_norm(f2) / 2
        fused = p1 * f1 + p2 * f2
    elif strategy == "FL1N":
        l1 = l1_norm(f1)
        l2 = l1_norm(f2)
        w1 = l1 / (l1 + l2)
        w2 = l2 / (l1 + l2)
        fused = w1 * f1 + w2 * f2
    elif strategy == "SFNN":
        def process_for_nuc(f):
            f = f.squeeze(0)
            total = []
            for i in range(f.shape[0]):
                temp = torch.norm(f[i], "nuc")
                # total = np.append(total, temp)
                total.append(temp.item())
            return total

        f1_soft = nn.functional.softmax(f1)
        f2_soft = nn.functional.softmax(f2)
        l1 = process_for_nuc(f1_soft)
        # print(l1)
        l2 = process_for_nuc(f2_soft)
        l1 = np.array(l1)
        l2 = np.array(l2)
        # w1 = np.mean(l1) / (np.mean(l1) + np.mean(l2))
        # w2 = np.mean(l2) / (np.mean(l1) + np.mean(l2))
        w1 = sum(l1) / (sum(l1) + sum(l2))
        w2 = sum(l2) / (sum(l1) + sum(l2))
        # w1 = max(l1) ** 2 / (max(l1) ** 2 + max(l2) ** 2)
        # w2 = max(l2) ** 2 / (max(l1) ** 2 + max(l2) ** 2)
        # f_sum = (f1 ** 2 + f2 ** 2).clone()
        # f_sum[f_sum == 0] = 1
        # k1 = f1 ** 2 / f_sum
        # k2 = f2 ** 2 / f_sum

        fused = w1 * f1 + w2 * f2
    # Need to do reconstruction on "fused"
        conv = nn.Conv2d(fused.shape[1], 256, kernel_size=1)
        fused = conv(fused)
    return fused
