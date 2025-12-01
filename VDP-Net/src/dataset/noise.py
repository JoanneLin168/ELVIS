import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import kornia

blur_ksize = 21 # fixed
actual_labels = {
    'exposure_value': [0, -3.5], # NEGATIVE exposure value in stops (so actually 0 to -1.5 ev stops)
    'shot_noise_log': [np.log(1e-1), np.log(30)], # from their code
    'read_noise': [0, 0.1], # read noise
    'quant_noise': [0, 0.1], # change to [0, 4] if using custom quant noise
    'band_noise': [0, 0.03], # banding noise
    'blur_sigma1': [0.5, 10], # to create covariance matrix for multivariate gaussian (0.5 min to avoid weird kernels)
    'blur_sigma2': [0.5, 10], # to create covariance matrix for multivariate gaussian
    'blur_angle': [0, 0.25],  # in radians (0-45deg) # to avoid repeat kernels
}

def apply_exposure(x, ev, device='cuda'):
    x = kornia.color.rgb_to_xyz(x) # sRGB -> XYZ
    x = x * (2 ** ev)
    x = kornia.color.xyz_to_rgb(x) # XYZ -> sRGB
    return x

def shot_noise(x, k, device='cuda'):
    if x.max() <= 1.0:
        x = (x * 255).int()
        noisy = torch.poisson(x / k) * k
        noisy = noisy.float() / 255.0
    else:
        noisy = torch.poisson(x / k) * k
    return noisy.to(device)

def gaussian_noise(x, scale, loc=0, device='cuda'):
    return torch.randn_like(x) * scale + loc

def quantization_noise(x, vmax, device='cuda'):
    n_quant = vmax * torch.rand(x.shape, device=device)
    return n_quant

def banding_noise(x, band_params, band_angles, num_frames, device='cuda'):
    x = x.view(-1, num_frames, x.shape[1], x.shape[2], x.shape[3])
    B, N, C, H, W = x.shape
    band_all = []
    
    for i in range(x.shape[0]):
        band_angle = torch.round(band_angles[i][0]).item()
        if band_angle == 0: # horizontal banding
            band_temp = band_params[i][0] * torch.randn((N, C, H), device=device).unsqueeze(-1) # (NxCxHx1)
            band_temp = band_temp.repeat(1, 1, 1, W).view(N, C, H, W)
            band_all.append(band_temp)
        elif band_angle == 1: # vertical banding
            band_temp = band_params[i][0] * torch.randn((N, C, W), device=device).unsqueeze(-2) # (NxCx1xW)
            band_temp = band_temp.repeat(1, 1, H, 1).view(N, C, H, W)
            band_all.append(band_temp)
        else:
            raise ValueError("band_angle should be 0 or 1 but got:", band_angle)
    n_band = torch.stack(band_all, dim=0)
    n_band = n_band.view(B*N, C, H, W)
    x = x.view(B*N, C, H, W)

    return n_band

def get_covariance_matrix(sig1, sig2, theta, device='cuda'):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.tensor([[c, -s],[s, c]], dtype=torch.float32, device=device)
    D = torch.diag(torch.tensor([sig1**2, sig2**2], dtype=torch.float32, device=device))
    return R @ D @ R.T

def get_multivariate_gaussian_kernel(kernel_size=15, cov=None, device='cuda'):
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).float()  # (ksize, ksize, 2)

    cov_inv = torch.linalg.inv(cov)
    exponent = -0.5 * torch.einsum('...i,ij,...j->...', coords, cov_inv, coords)

    kernel = torch.exp(exponent)
    kernel = kernel / kernel.sum()

    return kernel.view(1, 1, kernel_size, kernel_size)  # (1, 1, ksize, ksize)

def apply_blur(x, kernel):
    C = x.shape[1]
    ksize = kernel.shape[-1]
    kernel = kernel.expand(C, 1, ksize, ksize).to(x.device)  # (C, 1, ksize, ksize)
    return F.conv2d(x, kernel, padding=ksize // 2, groups=C) 


def reshape_noise_dict(in_noise_dict, batch_size=1, num_frames=1):
    """For when the input noise parameters are already in a dictionary"""

    out_noise_dict = {}
    # bs = batch_size * num_frames
    b = batch_size
    n = num_frames

    out_noise_dict['exposure_value'] = in_noise_dict['exposure_value'].view(b, n, 1, 1, 1)
    out_noise_dict['shot_noise_log'] = in_noise_dict['shot_noise_log'].view(b*n, 1, 1, 1)
    out_noise_dict['read_noise'] = in_noise_dict['read_noise'].view(b*n, 1, 1, 1)
    out_noise_dict['quant_noise'] = in_noise_dict['quant_noise'].view(b*n, 1, 1, 1)
    out_noise_dict['band_noise'] = in_noise_dict['band_noise'].view(b, n, 1, 1, 1)
    out_noise_dict['band_noise_angle'] = in_noise_dict['band_noise_angle'].view(b, n, 1)
    out_noise_dict['blur_sigma1'] = in_noise_dict['blur_sigma1'].view(b, n, 1, 1)
    out_noise_dict['blur_sigma2'] = in_noise_dict['blur_sigma2'].view(b, n, 1, 1)
    out_noise_dict['blur_angle'] = in_noise_dict['blur_angle'].view(b, n, 1, 1)
    return out_noise_dict

def reshape_noise_params(noise_params, num_frames=1):
    noise_dict = {
        'exposure_value': noise_params[:, :, 0],
        'shot_noise_log': noise_params[:, :, 1],
        'read_noise': noise_params[:, :, 2],
        'quant_noise': noise_params[:, :, 3],
        'band_noise': noise_params[:, :, 4],
        'band_noise_angle': noise_params[:, :, 5],
        'blur_sigma1': noise_params[:, :, 6],
        'blur_sigma2': noise_params[:, :, 7],
        'blur_angle': noise_params[:, :, 8],
    }

    noise_dict = reshape_noise_dict(noise_dict, batch_size=noise_params.shape[0], num_frames=num_frames)

    return noise_dict


def ProposedNoise(x, noise_dict, num_frames=1, device='cuda'):
    # Apply blur
    x = x.view(-1, num_frames, x.shape[1], x.shape[2], x.shape[3])
    B, N, C, H, W = x.shape
    x_m = []
    for b in range(B):
        x_b = []
        for n in range(N):
            x_frame = x[b, n]  # (C, H, W)
            covar_sigma1 = noise_dict['blur_sigma1'][b, n]
            covar_sigma2 = noise_dict['blur_sigma2'][b, n]
            covar_angle = noise_dict['blur_angle'][b, n] * torch.pi # convert to radians
            cov = get_covariance_matrix(covar_sigma1, covar_sigma2, covar_angle, device=device)  # (2, 2)
            blur_kernel = get_multivariate_gaussian_kernel(kernel_size=blur_ksize, cov=cov, device=device)
            x_blur = apply_blur(x_frame.unsqueeze(0), blur_kernel).squeeze(0)  # (C, H, W)
            x_b.append(x_blur)
        x_b = torch.stack(x_b, dim=0)  # (N, C, H, W)
        x_m.append(x_b)  # (N, C, H, W)
    x = torch.stack(x_m, dim=0).view(B*N, C, H, W)  # (B*N, C, H, W)

    # Apply noise
    x = x.view(-1, num_frames, x.shape[1], x.shape[2], x.shape[3])
    B, N, C, H, W = x.shape
    x = x.view(B*N, C, H, W)
    x = shot_noise(x, noise_dict['shot_noise'], device=device)
    x += gaussian_noise(x, noise_dict['read_noise'], device=device)
    x += quantization_noise(x, noise_dict['quant_noise'], device=device)
    x += banding_noise(x, noise_dict['band_noise'], noise_dict['band_noise_angle'], num_frames, device=device)

    return x


def generate_noise(x, noise_dict_not_scaled, num_frames=1, return_dark=False, device='cuda'):
    assert x.min() >= 0 and x.max() <= 1, "Input tensor should be in [0, 1] range"
    squeeze = False
    if x.ndim == 4:
        x = x.unsqueeze(0)
        squeeze = True
    assert x.ndim == 5, f"Input tensor should have shape [B, N, C, H, W] but got: {x.shape} for N={num_frames}"

    # Scale noise dict back to actual values: (vmax-vmin)*label + vmin
    noise_dict = {}
    for key in actual_labels:
        if key == 'shot_noise_log': # scale log_K to be between log(1e-1) and log(30)
            scale = actual_labels[key][1] - actual_labels[key][0]
            log_K = scale*noise_dict_not_scaled[key] + actual_labels[key][0]
            noise_dict['shot_noise'] = torch.exp(log_K)
        elif key == 'read_noise_tlambda': # scale log_K to be between log(1e-1) and log(30)
            scale = actual_labels[key][1] - actual_labels[key][0]
            log_lmbda = scale*noise_dict_not_scaled[key] + actual_labels[key][0]
            noise_dict['read_noise_tlambda'] = torch.exp(log_lmbda)
        else:
            scale = actual_labels[key][1] - actual_labels[key][0]
            noise_dict[key] = scale*noise_dict_not_scaled[key] + actual_labels[key][0]
    noise_dict['band_noise_angle'] = torch.round(noise_dict_not_scaled['band_noise_angle']) # 0 for horizontal, 1 for vertical

    # Put labels onto device
    for key in noise_dict:
        noise_dict[key] = noise_dict[key].to(device)

    # Apply degradations
    B, N, C, H, W = x.shape
    x_dark = apply_exposure(x, noise_dict['exposure_value'], device=device)
    x = x_dark.view(B*N, C, H, W)  # (B*N, C, H, W)
    noisy = ProposedNoise(x, noise_dict, num_frames=num_frames, device=device)
    noisy = noisy.view(B, N, C, H, W)
    noisy = torch.clip(noisy, 0, 1)
    
    if squeeze:
        x = x.squeeze(0)
        noisy = noisy.squeeze(0)
        x_dark = x_dark.squeeze(0)

    if return_dark:
        return noisy, x_dark
    return noisy
