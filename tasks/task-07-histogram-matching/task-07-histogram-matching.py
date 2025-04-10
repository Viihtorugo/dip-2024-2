# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import skimage as ski
#import matplotlib.pyplot as plt

def count_pixels(img):
    
    freq_pixels = []

    for i in range(0, 256):
        freq_pixels.append(0)
        
    h, w, c = img.shape

    for c in range(c):
        for y in  range(h):
            for x in  range(w):
                freq_pixels[img[y, x][c]] += 1
    
    return freq_pixels

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:

    #hist_source = count_pixels(source_img)
    #hist_reference = count_pixels(reference_img)
    matched = ski.exposure.match_histograms(source_img, reference_img, channel_axis=-1)

    
    #plt.imshow(matched)
    #plt.show()

    return matched

#img = cv.imread('source.jpg')
#img_ref = cv.imread('reference.jpg')

#print(match_histograms_rgb(img, img_ref))