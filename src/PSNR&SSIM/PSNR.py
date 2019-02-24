import cv2
import numpy as np
import math
import os


def PSNR(input_img1, input_img2):
    M = input_img1.shape[0]
    N = input_img1.shape[1]
    MSE = 0
    for i in range(M):
        for j in range(N):
            temp = (input_img1[i][j] - input_img2[i][j]) * (input_img1[i][j] - input_img2[i][j])
            MSE = MSE + temp
    mse = MSE / float(input_img1.shape[0] * input_img1.shape[1])
    psnr = 20 * np.log10(float(255) / math.sqrt(mse))
    return psnr

def RGB2YCrCb(img):
    Y = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Y[i][j] = 0.257 * img[i][j][2] + 0.564 * img[i][j][1] + 0.098 * img[i][j][0] + 16
    return Y
    # YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # Y = YCrCb[:, :, 0]
    # return Y
#
def operat_img(filepath): #operate file
    for name in os.listdir(filepath):
        print (name)
        if name[-4:] == ".bmp":
            name = name[:-4]
            img1 = cv2.imread(filepath + name + ".bmp")
            img2 = cv2.imread("./testDown/" + name + "2.bmp")
            y1 = RGB2YCrCb(img1)
            y2 = RGB2YCrCb(img2)
            psnr = PSNR(y1, y2)
            print(name,psnr)


# operat_img("./Set14/")
name = "man"
img1 = cv2.imread("./Set14/" + name+".bmp")
img2 = cv2.imread("./Set14/" + name+"3.bmp")
y1 = RGB2YCrCb(img1)
y2 = RGB2YCrCb(img2)
psnr = PSNR(y1, y2)
print (psnr)

