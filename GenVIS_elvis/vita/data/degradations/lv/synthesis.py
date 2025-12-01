import numpy as np
import cv2
import os
import argparse
import tqdm
from . import bayer_util
import csv
import shutil
import torch

np.random.seed(42)

# Multiply image by scale factor
def dim(image, scale_factor=0.5):
    return (image * scale_factor).astype('uint8')

# REFERENCE: https://arxiv.org/abs/1908.00682
def adjust_gamma(image, gamma=1.0):
    # invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

def darken_image(image, alpha, beta, gamma):
    out = alpha * image
    out = adjust_gamma(out.astype('uint8'), gamma=gamma)
    out = beta * out
    return out.astype('uint8')

# Poissonian-Gaussian noise model:
# z(x) = y(x) + n_p(y(x)) + n_g(x)
# where chi(y(x) + n_p(y(x))) ~ Poisson(chi*y(x)) and n_g ~ N(0, b^2)
# so: z(x) = Poisson(y(x))/chi + N(0, b^2)

# The paper then gets an overall noise approximation: z(x) ~= y(x) + n_h(y(x)) where n_h(x) ~ N(0, ay(x)+b)
# but I didn't bother here
def apply_noise(image, chi, b):
    image_float = image / 255
    poisson_component = (np.random.poisson(chi * image_float) / chi) # remember: poisson component has both the image and poisson noise
    gaussian_component = np.random.normal(0, b, image_float.shape)

    noisy = np.clip((poisson_component + gaussian_component) * 255, 0, 255).astype('uint8')

    return noisy

# Here you bothered:
def get_approx_noise(image, chi, b):
    a = 1/chi
    # due to heteroskedastic normal approximation: z(x) = y(x) + N(0, ay(x)+b)
    # refer to notes for working out
    signal_dependent_noise = np.random.normal(0, a * image + b, image.shape)
    return signal_dependent_noise

# From paper: https://arxiv.org/pdf/1807.04686.pdf, (https://github.com/GuoShi28/CBDNet/blob/master/utils/AddNoiseMosai.m)
# which is what https://arxiv.org/pdf/1908.00682.pdf bases their's off of
def ccd_camera_noise(irradiance, sig_s, sig_c):
    # sig_s = np.random.choice(np.arange(0, 0.16, step=0.02))
    # sig_c = np.random.choice(np.arange(0, 0.06, step=0.01))
    var_s = irradiance * sig_s**2
    var_c = sig_c**2
    # noise_s = np.random.normal(0, np.sqrt(var_s), irradiance.shape)
    # noise_c = np.random.normal(0, np.sqrt(var_c), irradiance.shape)
    # signal_dependent_noise = noise_s + noise_c
    noise = np.random.normal(0, np.sqrt(var_s + var_c), irradiance.shape)
    return noise

def display(image):
    img = (255 * image).astype('uint8')
    cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

# Referring to paper: https://arxiv.org/pdf/1908.00682.pdf
# and https://people.csail.mit.edu/celiu/pdfs/denoise_TPAMI.pdf
def convert_to_low_light(image, noise_dict):
    # Get noise parameters
    camera_idx = noise_dict['camera_idx']
    alpha = noise_dict['alpha']
    beta = noise_dict['beta']
    gamma = noise_dict['gamma']
    sig_s = noise_dict['sig_s']
    sig_c = noise_dict['sig_c']

    # Get CRF
    crf, icrf = bayer_util.get_crf_icrf(camera_idx)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_norm = image.copy() / 255

    # Part 1: Darken image
    img_dark = darken_image(image, alpha, beta, gamma) / 255

    # Part 2: Add noise through simulated camera processing pipeline
    # Part 2.1: Get irradiance, which is Inverse CRF image, then mosaic into Bayer format
    norm_irradiance = bayer_util.mosaic(icrf(img_norm))
    dark_irradiance = bayer_util.mosaic(icrf(img_dark))

    # Part 2.2: Add noise to irradiance
    noise = ccd_camera_noise(dark_irradiance, sig_s, sig_c)
    norm_irradiance_noisy = norm_irradiance + noise
    dark_irradiance_noisy = dark_irradiance + noise

    norm_irradiance_noisy = np.clip(norm_irradiance_noisy, 0, 1)
    dark_irradiance_noisy = np.clip(dark_irradiance_noisy, 0, 1)

    # Part 2.3: Demosaic noisy irradiance
    noisy_rgb = bayer_util.demosaic(norm_irradiance_noisy)
    dark_noisy_rgb = bayer_util.demosaic(dark_irradiance_noisy)
    
    noisy_rgb = np.clip(noisy_rgb, 0, 1)
    dark_noisy_rgb = np.clip(dark_noisy_rgb, 0, 1)

    # Part 2.4: Pass image through CRF
    dark_out = (img_dark * 255).astype('uint8')
    dark_out = cv2.cvtColor(dark_out, cv2.COLOR_RGB2BGR)

    noisy_out = crf(noisy_rgb)
    noisy_out = (noisy_out * 255).astype('uint8')
    noisy_out = cv2.cvtColor(noisy_out, cv2.COLOR_RGB2BGR)

    dark_noisy_out = crf(dark_noisy_rgb)
    dark_noisy_out = (dark_noisy_out * 255).astype('uint8')
    dark_noisy_out = cv2.cvtColor(dark_noisy_out, cv2.COLOR_RGB2BGR)

    return dark_noisy_out

def generate_lowlight(images, noise_dict=None):
    if noise_dict == None:
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

    degraded_images = []
    for img in images:
        degraded = convert_to_low_light(img, noise_dict)
        degraded_images.append(degraded)

    return degraded_images
