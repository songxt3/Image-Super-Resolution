import numpy as np
from scipy import signal
import cv2
import math
import os



def compute_ssim(img1, img2):
    sigma = 1.5 #Gaussian sigma = 1.6
    N = 5 #filter
    N_col = 2*N + 1 #size
    k1 = 0.01
    k2 = 0.03
    F = np.zeros((N_col,N_col))
    for i in range(N_col):
        for j in range(N_col):
            temp = float(math.pow(i-N-1, 2)+math.pow(j-N-1 ,2))
            F[i, j] = math.exp(- temp / (2 * sigma * sigma)) / (2 * math.pi * sigma)
    F = F / np.sum(F)

    img1 = img1.astype(np.float)
    img2 = img2.astype(np.float)

    img1_sq = img1 ** 2
    img2_sq = img2 ** 2
    img_12 = img1 * img2
    img1_mul = signal.convolve2d(img1, F)
    img2_mul = signal.convolve2d(img2, F)

    # img1_mul = my_convolve(img1, F)
    # img2_mul = my_convolve(img2, F)

    img1_mul_sq = img1_mul ** 2
    img2_mul_sq = img2_mul ** 2
    img_mul_12 = img1_mul * img2_mul

    img1_sigma = signal.convolve2d(img1_sq, F)

    #img1_sigma = my_convolve(img1_sq, F)
    img1_sigma = img1_sigma - img1_mul_sq

    img2_sigma = signal.convolve2d(img2_sq, F)

    #img2_sigma = my_convolve(img2_sq, F)
    img2_sigma = img2_sigma - img2_mul_sq

    img12_sigma = signal.convolve2d(img_12, F)

    #img12_sigma = my_convolve(img_12, F)
    img12_sigma = img12_sigma - img_mul_12

    c1 = (k1 * 255) ** 2
    c2 = (k2 * 255) ** 2

    up_part = (2 * img_mul_12 + c1) * (2 * img12_sigma + c2)
    down_part = (img1_mul_sq + img2_mul_sq + c1) * (img1_sigma + img2_sigma + c2)

    ssim_map = up_part / down_part

    #print (ssim_map)
    ssim = np.mean(ssim_map)

    return ssim

def RGB2YCrCb(img):
    Y = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Y[i][j] = 0.257 * img[i][j][2] + 0.564 * img[i][j][1] + 0.098 * img[i][j][0] + 16;
    return Y

def my_convolve(array, fil):
    fil_h = fil.shape[0]
    fil_w = fil.shape[1]
    conv_h = array.shape[0] - fil_h + 1
    conv_w = array.shape[1] - fil_w + 1
    conv = np.zeros((conv_h, conv_w), dtype=np.uint8)
    for i in range(conv_h):
        for j in range(conv_w):
            conv[i][j] = (array[i:i + fil_h, j:j + fil_w]*fil).sum()
    return conv


def operat_img(filepath): #operate file
    for name in os.listdir(filepath):
        print (name)
        if name[-4:] == ".bmp":
            name = name[:-4]
            img1 = cv2.imread(filepath + name + ".bmp")
            img2 = cv2.imread("./testDown/" + name + "2.bmp")
            y1 = RGB2YCrCb(img1)
            y2 = RGB2YCrCb(img2)
            SSIM = compute_ssim(y1, y2)
            print(name,SSIM)


operat_img("./Set14/")

# img1 = cv2.imread('./Set14/baboon.jpg')
# img2 = cv2.imread('./Set14/baboon2.jpg')
# img_ycc1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
# img_ycc2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
# y1 = cv2.split(img_ycc1)[0]
# y2 = cv2.split(img_ycc2)[0]
# y1 = RGB2YCrCb(img1)
# y2 = RGB2YCrCb(img2)
#
# print compute_ssim(y1, y2)