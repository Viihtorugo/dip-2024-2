# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np
import math
from scipy.signal import convolve2d

import cv2 #retirar depois


def mse(i1, i2):
    v1 = i1.copy()
    v2 = i2.copy()

    v1 = np.array(v1).flatten().tolist()
    v2 = np.array(v2).flatten().tolist()
    
    if len(v1) != len(v2):
        print("Imagens tem que ter as mesmas dimensões")
        return 0
    
    sum = 0

    for y, _y in zip(v1, v2):
        sum += (y - _y) ** 2

    sum = sum
    n = len(v1)

    return float(sum/n)

def psnr(i1, i2):

    if i1.shape != i2.shape:
        print("Imagens tem que ter as mesmas dimensões")
        return 0
    
    e = mse(i1, i2)
    
    bits = 0

    for char in str(i1.dtype):
        if char.isdigit():
            bits = 10 * bits + int(char) 

    max = 2 ** bits - 1
    
    return float(10 * math.log10(max ** 2/ e))

def ssim(i1, i2,  window_size=8, K1=0.01, K2=0.03, L=255):

    if i1.shape != i2.shape:
        print("Imagens tem que ter as mesmas dimensões")
        return 0
    
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    i1 = i1.astype(np.float64)
    i2 = i2.astype(np.float64)

    window = np.ones((window_size, window_size)) / (window_size ** 2)

    mu1 = convolve2d(i1, window, mode='valid')
    mu2 = convolve2d(i2, window, mode='valid')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(i1 ** 2, window, mode='valid') - mu1_sq
    sigma2_sq = convolve2d(i2 ** 2, window, mode='valid') - mu2_sq
    sigma12 = convolve2d(i1 * i2, window, mode='valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)

def calculate_std(vector, mean):
        variance = sum((x - mean) ** 2 for x in vector) / len(vector)
        return variance ** 0.5

def npcc(i1, i2):

    v1 = i1.copy()
    v2 = i2.copy()

    v1 = np.array(v1).flatten().tolist()
    v2 = np.array(v2).flatten().tolist()

    if len(v1) != len(v2):
        print("As imagens devem ter o mesmo tamanho.")
        return 0
    
    mean1 = sum(v1) / len(v1)
    mean2 = sum(v2) / len(v2)

    std1 = calculate_std(v1, mean1)
    std2 = calculate_std(v2, mean2)

    covariance = sum((v1[i] - mean1) * (v2[i] - mean2) for i in range(len(v1))) / len(v1)

    if std1 == 0 or std2 == 0:
        return 0.0
    else:
        npcc = covariance / (std1 * std2)
        return npcc
    

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    
    result = {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }

    print(result)

    return result
    pass

#i1 = cv2.imread('img/lena_gray.png', cv2.IMREAD_GRAYSCALE)
#i2 = cv2.imread('img/lena_gray_noise.png', cv2.IMREAD_GRAYSCALE)

#compare_images(i1, i2)