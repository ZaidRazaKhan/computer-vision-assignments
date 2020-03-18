import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
import pywt.data
from PIL import Image

import random

def high_threshold(high, threshold):
    for row in range(len(high)):
        for col in range(len(high)):
            if high[row][col] < threshold:
                high[row][col] = threshold
    return high


def sp_noise(image, prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def apply_kernel(image, kernel):
    return np.sum(np.multiply(image, kernel))

def zero_padding(image_matrix, kernel_matrix):
    left_padding = int(kernel_matrix.shape[0]/2)
    upper_padding = int(kernel_matrix.shape[1]/2)
    padded_image = np.zeros((image_matrix.shape[0]+kernel_matrix.shape[0]-1, image_matrix.shape[1]+kernel_matrix.shape[1]-1))
    for i in range(len(image_matrix)):
        for j in range(len(image_matrix[i])):
            padded_image[i+left_padding][j+upper_padding] = image_matrix[i][j]
    return padded_image

def convolution(image, kernel):
    convolved_image = np.zeros(image.shape)
    padded_image = zero_padding(image, kernel)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = padded_image[i:i+kernel.shape[0],j:j+kernel.shape[1]]
            convolved_image[i][j] = apply_kernel(temp, kernel)
    return convolved_image

def gaussian_kernel(size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# Load image
original = Image.open('image_3.png')

# Gray Conversion
original = rgb2gray(np.array(original))

# Laplacian Kernel
kernel = np.array([[0,-1, 0],[-1, 4, -1],[0, -1, 0]])

# Pre processed Image
original = original+sp_noise(original, 0.05)+convolution(original, kernel)
original = Image.fromarray(original)


g_kernel = gaussian_kernel(3)

coeffs2 = pywt.dwt2(original, 'haar')

LL, (LH, HL, HH) = coeffs2

print(LL.max()*0.7)
# LL = convolution(LL, g_kernel)
# HH = cv2.GaussianBlur(HH,(5,5),0)

LL = high_threshold(LL, (LL.max()-LL.min())/2)


coffs2=LL,(LH,HL,HH)
origin=pywt.idwt2(coffs2,'haar')
im = Image.fromarray(origin)
im.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(origin, interpolation = "nearest", cmap = plt.cm.gray)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
plt.show()