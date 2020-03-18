import numpy as np
from PIL import Image
import random
import pywt
class Image_processing:
    def __init__(self):
        print('Image Processor object is created')
    
    def rgb_to_grey(self, colored_image):
        return np.dot(colored_image[...,:3], [0.299, 0.587, 0.144])
    
    def read_image(self, path):
        return Image.open(path)
    
    def image_to_numpy_array(self, image):
        return np.array(image)
    
    def numpy_array_to_image(self, numpy_array):
        return Image.fromarray(numpy_array)
    
    def show_image(self, image):
        image.show()

    def __apply_kernel(self, image, kernel):
        return np.sum(np.multiply(image, kernel))
    
    def __zero_padding(self, image_matrix, kernel_matrix):
        left_padding = int(kernel_matrix.shape[0]/2)
        upper_padding = int(kernel_matrix.shape[1]/2)
        padded_image = np.zeros((image_matrix.shape[0]+kernel_matrix.shape[0]-1, image_matrix.shape[1]+kernel_matrix.shape[1]-1))
        for i in range(len(image_matrix)):
            for j in range(len(image_matrix[i])):
                padded_image[i+left_padding][j+upper_padding] = image_matrix[i][j]
        return padded_image
    
    def convolution(self, image, kernel):
        convolved_image = np.zeros(image.shape)
        padded_image = self.__zero_padding(image, kernel)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                temp = padded_image[i:i+kernel.shape[0],j:j+kernel.shape[1]]
                convolved_image[i][j] = self.__apply_kernel(temp, kernel)
        return convolved_image

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    def sp_noise(self, image, prob):
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
    
    def __median_filter(self, matrix):
        matrix = np.array(matrix)
        flat_matrix = matrix.flatten()
        sorted_array = np.sort(flat_matrix)
        return sorted_array[len(sorted_array)//2]
    
    def median_filtering(self, image, kernel_size):
        convolved_image = np.zeros(image.shape)
        padded_image = self.__zero_padding(image, np.zeros((kernel_size, kernel_size)))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                temp = padded_image[i:i+kernel_size,j:j+kernel_size]
                convolved_image[i][j] = self.__median_filter(temp)
        return convolved_image
    
    def sobel_filtering(self, img, Kx, Ky):
        Ix = self.convolution(img, Kx)
        Iy = self.convolution(img, Ky) 
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)