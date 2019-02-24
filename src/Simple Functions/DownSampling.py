# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import os

sigma = 1.6 #Gaussian sigma = 1.6
N = 3 #filter
N_col = 2*N + 1 #size
F = np.zeros((N_col,N_col))

#获得高斯卷积核
for i in range(N_col):
    for j in range(N_col):
        temp = float(math.pow(i-N-1, 2)+math.pow(j-N-1 ,2))
        F[i, j] = math.exp(- temp / (2 * sigma * sigma)) / (2 * math.pi * sigma)
F = F / np.sum(F)

def Gaussian_filter(img,fil):
    conv_B = my_convolve(img[:,:,0],fil)
    conv_G = my_convolve(img[:,:,1],fil)
    conv_R = my_convolve(img[:,:,2],fil)
    newImage = np.dstack([conv_B,conv_G,conv_R])
    return newImage

def my_convolve(array, fil):
    fil_h = fil.shape[0]
    fil_w = fil.shape[1]
    conv_h = array.shape[0] - fil_h + 1
    conv_w = array.shape[1] -fil_w + 1
    conv = np.zeros((conv_h, conv_w), dtype=np.uint8)
    for i in range(conv_h):
        for j in range(conv_w):
            conv[i][j] = (array[i:i + fil_h, j:j + fil_w]*fil).sum()
    return conv

def operat_img(filepath):
    for name in os.listdir(filepath):
        if name[-4:] == ".bmp":
            img = cv2.imread(filepath + "/" + name)
            print (img)
            pic = Gaussian_filter(img, F)
            pic = cv2.resize(pic, (img.shape[1] / 3, img.shape[0] / 3))
            cv2.imwrite("./DownSamplingTest/"+name, pic)
        print(name)

operat_img("./test_img")

# img = cv2.imread("test.jpg")
# pic = Gaussian_filter(img, F)
# pic = cv2.resize(pic,(img.shape[1] / 3, img.shape[0] / 3))
# cv2.imwrite("test_pic.jpg", pic)


