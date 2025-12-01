import os
import random
import numpy as np
import torch
import torchvision
import cv2
import scipy
import logging

from .proposed import reshape_noise_params, generate_noise
from .lv.synthesis import generate_lowlight
from .cui.synthesis import Dark_ISP, random_noise_levels
from .lin.noise import den_noise

logger = logging.getLogger(__name__)

def apply_degradations(images, model="proposed", kwargs=None, return_params=False):

    """
    Apply degradations to a list of images using the specified model.

    NOTE: input images are in the range [0, 255] and of type torch.Tensor.
    """

    if model == "lv":
        camera_idx = np.random.choice(np.arange(0, 201, 1))
        alpha = np.random.uniform(0.9, 1.0)
        beta = np.random.uniform(0.5, 1.0)
        gamma = np.random.uniform(1.5, 5)
        sig_s = np.random.uniform(0, 0.16)# [0,0.16]
        sig_c = np.random.uniform(0, 0.06) # [0,0.06]
        noise_dict = {
            'camera_idx': camera_idx,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'sig_s': sig_s,
            'sig_c': sig_c
        }

        # convert to numpy
        noisy_images = []
        for img in images:
            img = img.permute(1, 2, 0).cpu().numpy()
            noisy_images.append(img.astype('uint8'))
        noisy_images = generate_lowlight(noisy_images, noise_dict=noise_dict)
        noisy_images = [torch.from_numpy(img).permute(2, 0, 1) for img in noisy_images]

    elif model == "cui":
        # Source: https://github.com/cuiziteng/ICCV_MAET/blob/master/configs/MAET_yolo/maet_yolo_coco_ort.py#L42
        config = dict(darkness_range=(0.01, 1.0),
                                gamma_range=(2.0, 3.5),
                                rgb_range=(0.8, 0.1),
                                red_range=(1.9, 2.4),
                                blue_range=(1.5, 1.9),
                                quantisation=[12, 14, 16])
        dark_isp = Dark_ISP(config=config)

        # Degradation to sample from
        xyz2cams = [[[1.0234, -0.2969, -0.2266],
                        [-0.5625, 1.6328, -0.0469],
                        [-0.0703, 0.2188, 0.6406]],
                    [[0.4913, -0.0541, -0.0202],
                        [-0.613, 1.3513, 0.2906],
                        [-0.1564, 0.2151, 0.7183]],
                    [[0.838, -0.263, -0.0639],
                        [-0.2887, 1.0725, 0.2496],
                        [-0.0627, 0.1427, 0.5438]],
                    [[0.6596, -0.2079, -0.0562],
                        [-0.4782, 1.3016, 0.1933],
                        [-0.097, 0.1581, 0.5181]]]


        # Ensure the degradation is the same for all frames in a video
        # camera colour matrix
        xyz2cam = random.choice(xyz2cams)
        gamma = random.uniform(config['gamma_range'][0], config['gamma_range'][1])
        rgb_gain = random.normalvariate(config['rgb_range'][0], config['rgb_range'][1])
        red_gain = random.uniform(config['red_range'][0], config['red_range'][1])
        blue_gain = random.uniform(config['blue_range'][0], config['blue_range'][1])
        shot_noise, read_noise = random_noise_levels()
        bits = random.choice(config['quantisation'])
        lower, upper = config['darkness_range'][0], config['darkness_range'][1]
        mu, sigma = 0.1, 0.08
        darkness = scipy.stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        darkness = darkness.rvs()
        degradation_info = {
            'xyz2cam': xyz2cam,
            'gamma': gamma,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
            'shot_noise': shot_noise,
            'read_noise': read_noise,
            'bits': bits,
            'darkness': darkness
        }

        # Process each video
        noisy_images = []
        for img in images:
            img = img.float() / 255.0
            dark_img, _ = dark_isp(img, degradation_info)
            noisy_images.append((dark_img * 255).int())

    elif model == "lin":
        if kwargs['noise_param_dir'] is not None:
            noise_params_folders = [os.path.join(kwargs['noise_param_dir'], f) for f in os.listdir(kwargs['noise_param_dir'])
                                    if os.path.isdir(os.path.join(kwargs['noise_param_dir'], f))]
            
            # put all noise params from all folders into one list
            noise_params = []
            for folder in noise_params_folders:
                noise_params += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
            noise_param_file = np.load(random.choice(noise_params))
            noise_param = torch.tensor(noise_param_file)
        else:
            # randomly generate noise_param in range [0,1] with shape (1, 8)
            noise_param = torch.rand(1, 8)

        alpha = torch.tensor(np.round(np.random.uniform(0.1, 1))).view(1, 1)
        gamma = torch.tensor(np.round(np.random.uniform(0.1, 1))).view(1, 1)
        noise_param = torch.cat((alpha, gamma, noise_param), dim=1).repeat(len(images), 1)

        images = torch.stack(images, dim=0).float() / 255.0
        noisy_images = den_noise(images, noise_param, device="cpu")
        noisy_images = (noisy_images * 255).int()
        noisy_images = [img.detach().cpu() for img in noisy_images]

    elif model == "proposed":
        if kwargs['noise_param_dir'] is not None:
            # put all noise params from all folders into one list
            noise_params_folders = [os.path.join(kwargs['noise_param_dir'], f) for f in os.listdir(kwargs['noise_param_dir'])
                                    if os.path.isdir(os.path.join(kwargs['noise_param_dir'], f))]

            # put all noise params from all folders into one list
            noise_params = []
            for folder in noise_params_folders:
                noise_params += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
                
            # order by videos in dict
            video_noise_params_dict = {os.path.basename(f).split('_')[1]: [] for f in noise_params}
            for f in noise_params:
                video_name = os.path.basename(f).split('_')[1]
                video_noise_params_dict[video_name].append(f)

            random_video = random.choice(list(video_noise_params_dict.keys()))
            video_noise_params = video_noise_params_dict[random_video]
            noise_param_idx = random.randint(0, len(video_noise_params)-1)
            noise_param_file = video_noise_params[noise_param_idx]
            noise_param = np.load(noise_param_file)
            noise_param = torch.tensor(noise_param)

            while noise_param.shape[1] < len(images):
                noise_param_idx += 1
                if noise_param_idx >= len(video_noise_params):
                    noise_param_idx = 0
                noise_param_video_name = os.path.basename(noise_param_file).split('_')[1]
                tmp = video_noise_params_dict[noise_param_video_name][noise_param_idx]
                tmp = np.load(tmp)
                tmp = torch.tensor(tmp)
                noise_param = torch.cat((noise_param, tmp), dim=1)
            noise_param = noise_param[:, :len(images), :]
        else:
            # randomly generate noise_param in range [0,1] with shape (1, len(images), 9)
            noise_param = torch.rand(1, len(images), 9)

        images = torch.stack(images, dim=0).float() / 255.0
        noise_dict = reshape_noise_params(noise_param, noise_model="proposed", num_frames=len(images))
        noisy_images = generate_noise(images, noise_dict, noise_model="proposed", num_frames=len(images), device="cpu")

        noisy_images = (noisy_images * 255).int()
        noisy_images = [img.detach().cpu() for img in noisy_images]
    
    else:
        raise ValueError(f"Unknown degradation model: {model}")

    noisy_images = [torch.clamp(img.detach().cpu(), 0, 255) for img in noisy_images]
    
    if return_params:
        return noisy_images, noise_param
    else:
        return noisy_images