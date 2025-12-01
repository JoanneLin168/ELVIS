import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import kornia

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# for sampling, use actual value of: (vmax-vmin)*label + vmin, as label is scaled to be between 0 and 1
# NOTE: for brightness it is already in the right range, as it isn't a value to be predicted
# and for banding_noise_angle it is either 0 or 1 so no need for scaling
actual_labels_starlight = {
    'alpha_brightness': [0.1, 0.5],  # alpha for brightness
    'gamma_brightness': [0.1, 1],  # gamma for brightness
    'shot_noise': [0, 0.5],
    'read_noise': [0, 0.1],
    'quant_noise': [0, 0.1],
    'band_noise': [0, 0.005],
    'band_noise_temp': [0, 0.005],
    'periodic0': [0, 0.5],
    'periodic1': [0, 0.5],
    'periodic2': [0, 0.5],
}

actual_labels_eld = {
    'alpha_brightness': [0, 1],  # alpha for brightness
    'gamma_brightness': [0, 1],  # gamma for brightness
    'shot_noise_log': [np.log(1e-1), np.log(30)], # from their code https://github.com/Vandermode/ELD/blob/master/noise.py#L215
    'read_noise_scale': [0, 30],
    'read_noise_tlambda': [np.log(1e-1), np.log(30)], # lmbda=0.14 for a gaussian (refer to paper)
    'quant_noise': [1, 1], # just use 1: https://github.com/Srameo/LED/blob/main/led/data/noise_utils/noise_generator.py#L149
    'band_noise': [0, 0.005]
}

actual_labels_simple = {
    'alpha_brightness': [0, 1],  # alpha for brightness
    'gamma_brightness': [0, 1],  # gamma for brightness
    'shot_noise_log': [np.log(1e-1), np.log(30)],
    'read_noise': [0, 0.1],
    'quant_noise': [0, 0.1],
    'band_noise': [0, 0.01],
}

blur_ksize = 21 # fixed

actual_labels_proposed = {
    'exposure_value': [0, -3.5], # NEGATIVE exposure value in stops (so actually 0 to -1.5 ev stops)
    'shot_noise_log': [np.log(1e-1), np.log(30)], # from their code
    'read_noise': [0, 0.1], # read noise
    'quant_noise': [0, 0.1], # change to [0, 4] if using custom quant noise
    'band_noise': [0, 0.03], # banding noise
    'blur_sigma1': [0.5, 10], # to create covariance matrix for multivariate gaussian (0.5 min to avoid weird kernels)
    'blur_sigma2': [0.5, 10], # to create covariance matrix for multivariate gaussian
    'blur_angle': [0, 0.25],  # in radians (0-45deg) # to avoid repeat kernels
}


actual_labels_dict = {
    "starlight": actual_labels_starlight,
    "eld": actual_labels_eld,
    "simple": actual_labels_simple,
    "proposed": actual_labels_proposed,
}

def apply_exposure(x, ev, device='cuda'):
    x = kornia.color.rgb_to_xyz(x) # sRGB -> XYZ
    x = x * (2 ** ev)
    x = kornia.color.xyz_to_rgb(x) # XYZ -> sRGB
    return x

def heteroscedastic_noise(x, var_r, var_s, device='cuda'):
    var = x*var_s + var_r
    n_h = torch.randn(x.shape, device=device)*var
    return n_h

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

# REFERENCE: https://github.com/Srameo/LED/blob/main/led/data/noise_utils/common.py
def tukey_lambda_noise(x, scale, t_lambda=1.4, device='cuda'):
    def tukey_lambda_ppf(p, t_lambda):
        assert not torch.any(t_lambda == 0.0)
        return 1 / t_lambda * (p ** t_lambda - (1 - p) ** t_lambda)
    
    tmp_scale = False
    if x.max() <= 1.0:
        x = x * 255
        tmp_scale = True

    epsilon = 1e-10
    U = torch.rand_like(x) * (1 - 2 * epsilon) + epsilon
    Y = tukey_lambda_ppf(U, t_lambda) * scale

    if tmp_scale:
        Y = (Y / 255.0).float()

    return Y

def quant_noise(x, q, device='cuda'):
    return (torch.rand_like(x) - 0.5) * q

def quantization_noise(x, vmax, device='cuda'):
    n_quant = vmax * torch.rand(x.shape, device=device)
    return n_quant

def get_quant_step(q):
    level = (2**torch.round(q)).float()
    step = 1/level
    return step

def quant_custom_noise(x, q, device='cuda'):
    step = get_quant_step(q)
    n_quant = (torch.rand_like(x) - 0.5) * step
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

def banding_temp_noise(x, bandt_params, bandt_angles, num_frames, device='cuda'):
    # Banding temp noise
    # NOTE: must do this for individual videos to ensure different angles per video in batch
    x = x.view(-1, num_frames, x.shape[1], x.shape[2], x.shape[3])
    B, N, C, H, W = x.shape
    bandt_all = []
    for i in range(x.shape[0]):
        bandt_angle = torch.round(bandt_angles[i][0]).item()
        if bandt_angle == 0: # horizontal banding
            bandt_temp = bandt_params[i] * torch.randn((C, H), device=device).unsqueeze(-1).unsqueeze(0) # 1xCxHx1
            bandt_temp = bandt_temp.repeat(N, 1, 1, W).view(N, C, H, W)
            bandt_all.append(bandt_temp)
        elif bandt_angle == 1: # vertical banding
            bandt_temp = bandt_params[i] * torch.randn((C, W), device=device).unsqueeze(-2).unsqueeze(0)  # 1xCx1xW
            bandt_temp = bandt_temp.repeat(N, 1, H, 1).view(N, C, H, W)
            bandt_all.append(bandt_temp)
        else:
            raise ValueError("bandt_angles should be 0 or 1 but got:", bandt_angles)
    n_bandt = torch.stack(bandt_all, dim=0)
    n_bandt = n_bandt.view(B*N, C, H, W)
    return n_bandt

def periodic_noise(x, band_angles, param0, param1, param2, num_frames, device='cuda'):
    x = x.view(-1, num_frames, x.shape[1], x.shape[2], x.shape[3])
    B, N, C, H, W = x.shape
    # Create periodic noise separately for each frame based on band_noise_angle
    n_periodic = torch.zeros_like(x, device=device)
    for i in range(x.shape[0]):
        c_periodic = torch.zeros(*x[i].shape,  dtype=torch.cfloat, device=device)
        band_angle = torch.round(band_angles[i][0]).item()
        if band_angle == 0:
            c_periodic[...,0,0] = param0[i][0]*torch.randn((x.shape[1:-2]), device=device)
            periodic0 = param1[i][0]*torch.randn((x.shape[1:-2]), device=device)
            periodic1 = param2[i][0]*torch.randn((x.shape[1:-2]), device=device) 
            c_periodic[...,x.shape[-2]//4,0] = torch.complex(periodic0, periodic1)
            c_periodic[...,3*x.shape[-2]//4,0] = torch.complex(periodic0, -periodic1)
        elif band_angle == 1:
            c_periodic[...,0,0] = param0[i][0]*torch.randn((x.shape[1:-2]), device=device)
            periodic0 = param1[i][0]*torch.randn((x.shape[1:-2]), device=device)
            periodic1 = param2[i][0]*torch.randn((x.shape[1:-2]), device=device) 
            c_periodic[...,0,x.shape[-1]//4] = torch.complex(periodic0, periodic1)
            c_periodic[...,0,3*x.shape[-1]//4] = torch.complex(periodic0, -periodic1)
        n_periodic[i] = torch.abs(torch.fft.ifft2(c_periodic, norm="ortho"))
    x = x.view(B*N, C, H, W)
    n_periodic = n_periodic.view(B*N, C, H, W)

    return n_periodic

def get_motion_blur_kernel(kernel_size=15, angle=0, device='cuda'):
    kernel = torch.zeros((kernel_size, kernel_size)).to(device)
    center = kernel_size // 2
    angle_rad = (angle * torch.pi / 180)

    dx = torch.cos(angle_rad)
    dy = torch.sin(angle_rad)

    for i in range(kernel_size):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1

    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size) # (1, 1, ksize, ksize)

def get_gaussian_blur_kernel(kernel_size=15, sigma=1.0, device='cuda'):
    ax = torch.arange(kernel_size) - kernel_size // 2
    ax = ax.to(device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)  # (1, 1, ksize, ksize)

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

def calculate_ksize(value):
    assert 0.0 <= value <= 1.0, "value representing kernel size must be between 0 and 1"
    index = int(value * 10 + 1e-8)
    return 2 * index + 1

def calculate_numframes(value):
    assert 0.0 <= value <= 1.0, "value representing number of frames must be between 0 and 1"
    index = int(value * 4 + 1e-8)
    return 2 * index # 0, 2, 4, 6, 8 frames


def reshape_noise_dict(in_noise_dict, noise_model, batch_size=1, num_frames=1):
    """For when the input noise parameters are already in a dictionary"""

    out_noise_dict = {}
    # bs = batch_size * num_frames
    b = batch_size
    n = num_frames

    if noise_model == 'proposed':
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

    # Add brightness params
    out_noise_dict['alpha_brightness'] = in_noise_dict['alpha_brightness'].view(b, n, 1, 1, 1)
    out_noise_dict['gamma_brightness'] = in_noise_dict['gamma_brightness'].view(b, n, 1, 1, 1)
    out_noise_dict['quant_noise'] = in_noise_dict['quant_noise'].view(b*n, 1, 1, 1)
    out_noise_dict['band_noise'] = in_noise_dict['band_noise'].view(b, n, 1, 1, 1)
    out_noise_dict['band_noise_angle'] = in_noise_dict['band_noise_angle'].view(b, n, 1)

    # Reshape noise parameters for batch processing
    if noise_model == "starlight":
        out_noise_dict['shot_noise'] = in_noise_dict['shot_noise'].view(b*n, 1, 1, 1)
        out_noise_dict['read_noise'] = in_noise_dict['read_noise'].view(b*n, 1, 1, 1)
        out_noise_dict['periodic0'] = in_noise_dict['periodic0'].view(b*n, 1)
        out_noise_dict['periodic1'] = in_noise_dict['periodic1'].view(b*n, 1)
        out_noise_dict['periodic2'] = in_noise_dict['periodic2'].view(b*n, 1)
        out_noise_dict['band_noise_temp'] = in_noise_dict['band_noise_temp'].view(b, n, 1, 1, 1)
    elif noise_model == "eld":
        out_noise_dict['shot_noise_log'] = in_noise_dict['shot_noise_log'].view(b*n, 1, 1, 1)
        out_noise_dict['read_noise_scale'] = in_noise_dict['read_noise_scale'].view(b*n, 1, 1, 1)
        out_noise_dict['read_noise_tlambda'] = in_noise_dict['read_noise_tlambda'].view(b*n, 1, 1, 1)
    elif noise_model == "simple":
        out_noise_dict['shot_noise_log'] = in_noise_dict['shot_noise_log'].view(b*n, 1, 1, 1)
        out_noise_dict['read_noise'] = in_noise_dict['read_noise'].view(b*n, 1, 1, 1)
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    return out_noise_dict


def reshape_noise_params(noise_params, noise_model, num_frames=1):
    # Reshape noise parameters for batch processing
    if noise_model == "starlight":
        noise_dict = {
            'alpha_brightness': noise_params[:, :, 0],
            'gamma_brightness': noise_params[:, :, 1],
            'shot_noise': noise_params[:, :, 2],
            'read_noise': noise_params[:, :, 3],
            'quant_noise': noise_params[:, :, 4],
            'band_noise': noise_params[:, :, 5],
            'band_noise_temp': noise_params[:, :, 6],
            'band_noise_angle': noise_params[:, :, 7],
            'periodic0': noise_params[:, :, 8],
            'periodic1': noise_params[:, :, 9],
            'periodic2': noise_params[:, :, 10],
        }
    elif noise_model == "eld":
        noise_dict = {
            'alpha_brightness': noise_params[:, :, 0],
            'gamma_brightness': noise_params[:, :, 1],
            'shot_noise_log': noise_params[:, :, 2],
            'read_noise_scale': noise_params[:, :, 3],
            'read_noise_tlambda': noise_params[:, :, 4],
            'quant_noise': noise_params[:, :, 5],
            'band_noise': noise_params[:, :, 6],
            'band_noise_angle': noise_params[:, :, 7],
        }
    elif noise_model == "simple":
        noise_dict = {
            'alpha_brightness': noise_params[:, :, 0],
            'gamma_brightness': noise_params[:, :, 1],
            'shot_noise_log': noise_params[:, :, 2],
            'read_noise': noise_params[:, :, 3],
            'quant_noise': noise_params[:, :, 4],
            'band_noise': noise_params[:, :, 5],
            'band_noise_angle': noise_params[:, :, 6],
        }
    elif noise_model == "proposed":
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
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")
    
    noise_dict = reshape_noise_dict(noise_dict, noise_model, batch_size=noise_params.shape[0], num_frames=num_frames)

    return noise_dict

def StarlightNoise(x, noise_dict, num_frames=1, device='cuda'):
    noise = torch.zeros_like(x, device=device)
    noise += heteroscedastic_noise(x, noise_dict['shot_noise'], noise_dict['read_noise'], device=device)
    noise += quantization_noise(x, noise_dict['quant_noise'], device=device)
    noise += banding_noise(x, noise_dict['band_noise'], noise_dict['band_noise_angle'], num_frames, device=device)
    noise += banding_temp_noise(x, noise_dict['band_noise_temp'], noise_dict['band_noise_angle'], num_frames, device=device)
    noise += periodic_noise(x, noise_dict['band_noise_angle'], noise_dict['periodic0'], noise_dict['periodic1'], noise_dict['periodic2'], num_frames, device=device)
    
    return x + noise

def ELDNoise(x, noise_dict, num_frames=1, device='cuda'):
    noisy = shot_noise(x, noise_dict['shot_noise'], device=device)
    noisy += tukey_lambda_noise(x, noise_dict['read_noise_scale'], noise_dict['read_noise_tlambda'], device=device)
    noisy += quant_noise(x, noise_dict['quant_noise'], device=device)
    noisy += banding_noise(x, noise_dict['band_noise'], noise_dict['band_noise_angle'], num_frames, device=device)
    return noisy

def SimpleNoise(x, noise_dict, num_frames=1, device='cuda'):
    noisy = shot_noise(x, noise_dict['shot_noise'], device=device)
    noisy += gaussian_noise(x, noise_dict['read_noise'], device=device)
    noisy += quantization_noise(x, noise_dict['quant_noise'], device=device)
    noisy += banding_noise(x, noise_dict['band_noise'], noise_dict['band_noise_angle'], num_frames, device=device)
    return noisy

def ProposedNoise(x, noise_dict, num_frames=1, device='cuda'):
    """Adds gaussian and motion blur to ELD noise"""

    # Convert to (B,N,C,H,W) format for motion+gaussian blur
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


def generate_noise(x, noise_dict_not_scaled, noise_model, num_frames=1, return_dark=False, device='cuda'):
    assert x.min() >= 0 and x.max() <= 1, "Input tensor should be in [0, 1] range"
    squeeze = False
    if x.ndim == 4:
        x = x.unsqueeze(0)
        squeeze = True
    # assert x.ndim == 4 and x.shape[0] % num_frames == 0, f"Input tensor should have shape [B*N, C, H, W] but got: {x.shape} for N={num_frames}"
    assert x.ndim == 5, f"Input tensor should have shape [B, N, C, H, W] but got: {x.shape} for N={num_frames}"

    # Scale noise dict back to actual values: (vmax-vmin)*label + vmin
    noise_dict = {}
    actual_labels = actual_labels_dict[noise_model]
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
    
    # Add other noise parameters
    noise_dict['band_noise_angle'] = torch.round(noise_dict_not_scaled['band_noise_angle'])

    # Put labels onto device
    for key in noise_dict:
        noise_dict[key] = noise_dict[key].to(device)

    
    # Apply degradations
    B, N, C, H, W = x.shape
    if noise_model == 'proposed':
        x_dark = apply_exposure(x, noise_dict['exposure_value'], device=device)
        x = x_dark.view(B*N, C, H, W)  # (B*N, C, H, W)
        noisy = ProposedNoise(x, noise_dict, num_frames=num_frames, device=device)
        noisy = noisy.view(B, N, C, H, W)

    else:
        alpha = noise_dict['alpha_brightness']
        gamma = noise_dict['gamma_brightness']
        x_dark = alpha*(torch.pow(x, 1/gamma))

        if noise_model == "starlight":
            x = x_dark.view(B*N, C, H, W)  # (B*N, C, H, W)
            noisy = StarlightNoise(x, noise_dict, num_frames=num_frames, device=device)
            noisy = noisy.view(B, N, C, H, W)  # (B, N, C, H, W)
        elif noise_model == "eld":
            x = x_dark.view(B*N, C, H, W)  # (B*N, C, H, W)
            noisy = ELDNoise(x, noise_dict, num_frames=num_frames, device=device)
            noisy = noisy.view(B, N, C, H, W)  # (B, N, C, H, W)
        elif noise_model == "simple":
            x = x_dark.view(B*N, C, H, W)  # (B*N, C, H, W)
            noisy = SimpleNoise(x, noise_dict, num_frames=num_frames, device=device)
            noisy = noisy.view(B, N, C, H, W)  # (B, N, C, H, W)
        else:
            raise ValueError(f"Unknown noise model: {noise_model}")

    noisy = torch.clip(noisy, 0, 1)
    
    if squeeze:
        x = x.squeeze(0)
        noisy = noisy.squeeze(0)
        x_dark = x_dark.squeeze(0)

    if return_dark:
        return noisy, x_dark
    return noisy
    