import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def split_into_patches2d(x, patch_size = 64):
    patches = torch.empty([1,x.shape[1],patch_size,patch_size], device = x.device)
    for xx in range(0,x.shape[-2]//patch_size):
        for yy in range(0,x.shape[-1]//patch_size):
            patches = torch.cat([patches, x[...,xx*patch_size:(xx+1)*patch_size, yy*patch_size:(yy+1)*patch_size]], 0)
    patches = patches[1:,...]
    return patches

def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers

def cal_kld(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    bw = 0.2 / 64
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
    bin_edges = None
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    return kl_sym #kl_fwd #, kl_inv, kl_sym


# For pytorch loss
def kl_divergence(p, q, eps=1e-10):
    p = torch.clamp(p, eps, 1)
    q = torch.clamp(q, eps, 1)
    return torch.sum(p * torch.log(p / q), dim=-1)  # sum over bins

def patch_histogram(patch, bins=256):
    # patch: (B, C, H, W) in [0,1]
    B, C, H, W = patch.shape
    patch_vals = (patch * (bins - 1)).long()  # scale to [0, bins-1]
    one_hot = F.one_hot(patch_vals, num_classes=bins).float()  # (B, C, H, W, bins)
    hist = one_hot.sum(dim=(2, 3))  # sum over H,W → (B, C, bins)
    hist /= hist.sum(dim=-1, keepdim=True)  # normalize
    return hist

def kld_loss(real, fake, bins=256):
    # Calculate D(real || fake) using KL divergence on histograms (how much fake differs from real)
    hist1 = patch_histogram(real, bins=bins)
    hist2 = patch_histogram(fake, bins=bins)

    kl_per_channel = kl_divergence(hist1, hist2)  # (B, C)
    kl_per_patch = kl_per_channel.mean(dim=1)     # mean over RGB channels
    mean_kl = kl_per_patch.mean()                 # mean over batch

    return mean_kl
