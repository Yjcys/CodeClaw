# utils.py

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
import tqdm

# PSNR calculation
def calculate_psnr(img1, img2):
    """Calculate Peak Signal to Noise Ratio between two images"""
    # Convert to numpy arrays
    img1 = np.array(img1)
    img2 = np.array(img2)

    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    # Calculate PSNR
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

# SSIM calculation
def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index between two images"""
    # Convert to numpy arrays
    img1 = np.array(img1)
    img2 = np.array(img2)

    # Calculate SSIM
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 4)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 4)

    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 4) - mu1**2
    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 4) - mu2**2
    sigma12 = cv2.GaussianBlur(img1*img2, (11, 11), 4) - mu1*mu2

    ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim = np.mean(ssim_map)
    return ssim

# MOS calculation
def calculate_mos(images):
    """Calculate Mean Opinion Score for a list of images"""
    # This is a simplified implementation - in practice, you would use human ratings
    # For demonstration purposes, we'll just return a random score between 1 and 5
    return np.random.uniform(1, 5, size=len(images))

# Image quality assessment function
def evaluate_image_quality(generated_images, reference_images):
    """Evaluate image quality using PSNR, SSIM, and MOS"""
    psnr_scores = []
    ssim_scores = []
    mos_scores = []

    for gen_img, ref_img in zip(generated_images, reference_images):
        psnr_scores.append(calculate_psnr(gen_img, ref_img))
        ssim_scores.append(calculate_ssim(gen_img, ref_img))
        mos_scores.append(calculate_mos([gen_img]))

    return {
        'psnr': np.mean(psnr_scores),
        'ssim': np.mean(ssim_scores),
        'mos': np.mean(mos_scores)
    }

# Visualization function
def plot_results(results):
    """Plot training results including PSNR, SSIM, and MOS"""
    plt.figure(figsize=(12, 6))
    
    # Plot PSNR
    plt.subplot(1, 3, 1)
    plt.plot(results['psnr'])
    plt.title('PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    
    # Plot SSIM
    plt.subplot(1, 3, 2)
    plt.plot(results['ssim'])
    plt.title('SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    
    # Plot MOS
    plt.subplot(1, 3, 3)
    plt.plot(results['mos'])
    plt.title('MOS')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.show()