#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def getWvalue(x):
    x = abs(x)
    if x <= 1:
        temp = float((1.5 * x - 2.5) * x * x + 1)
    elif x <= 2:
        temp = float(((-0.5 * x + 2.5) * x - 4) * x + 2)
    else:
        temp = 0
    return temp

def controlRGB(num):
    if num > 255:
        num = 255
    elif num < 0:
        num = 0
    return num

def Bicubic(pic, desH, desW):
    new_R = np.zeros((desH,desW))
    new_G = np.zeros((desH,desW))
    new_B = np.zeros((desH,desW))
    channels = pic.shape[2]
    m = pic.shape[0]
    n = pic.shape[1]
    radio_r = float(m) / float(desH)
    radio_c = float(n) / float(desW)
    for i in range(desH):
        src_r = float(i) * radio_r  ## 放大图像位于原图的位置
        r = int(src_r)
        t = src_r - r
        #print (src_r,r,t)
        for j in range(desW):
            src_c = float(j) * radio_c
            c = int(src_c)
            u = src_c - float(c)
            for k1 in range(4):
                for k2 in range(4):
                    if k1 == 0:
                        Wx = getWvalue(1 + t)
                    elif k1 == 1:
                        Wx = getWvalue(t)
                    elif k1 == 2:
                        Wx = getWvalue(1 - t)
                    elif k1 == 3:
                        Wx = getWvalue(2 - t)
                    if k2 == 0:
                        Wy = getWvalue(1 + u)
                    elif k2 == 1:
                        Wy = getWvalue(u)
                    elif k2 == 2:
                        Wy = getWvalue(1 - u)
                    elif k2 == 3:
                        Wy = getWvalue(2 - u)
                    if int(r + k1 - 1) < 0 or int(c + k2 - 1) < 0 or int(r + k1 - 1) >= m or int(c + k2 - 1) >= n:
                        tG = 0
                        tB = 0
                        tR = 0
                    else:
                        tB = pic[int(r + k1 - 1)][int(c + k2 - 1)][0]
                        tG = pic[int(r + k1 - 1)][int(c + k2 - 1)][1]
                        tR = pic[int(r + k1 - 1)][int(c + k2 - 1)][2]
                    new_B[i][j] += tB * Wx * Wy
                    new_G[i][j] += tG * Wx * Wy
                    new_R[i][j] += tR * Wx * Wy
                    new_B[i][j] = controlRGB(new_B[i][j])
                    new_G[i][j] = controlRGB(new_G[i][j])
                    new_R[i][j] = controlRGB(new_R[i][j])
    newImage = np.dstack([new_B, new_G, new_R])
    return newImage

def operat_img(filepath): #operate file
    for name in os.listdir(filepath):
        print (name)
        if name[-4:] == ".bmp":
            img = cv2.imread(filepath + name)
            name = name[:-4]
            cv2.imwrite("./result/" + name + ".bmp", img)
            src_r = img.shape[0]
            src_c = img.shape[1]
            temp_r = src_r / 3
            temp_c = src_c / 3
            print (temp_r, temp_c)
            pic1 = Bicubic(img, temp_r, temp_c)
            cv2.imwrite("./result/" + name + "1.bmp", pic1)
            img1 = cv2.imread("./result/" + name + "1.bmp")
            pic2 = Bicubic(img1, src_r, src_c)
            cv2.imwrite("./result/" + name + "2.bmp", pic2)

operat_img("./Set14/")

# name = "bridge"
# img = cv2.imread("./Set14/"+name+".bmp")
# src_r = img.shape[0]
# src_c = img.shape[1]
# temp_r = src_r / 3
# temp_c = src_c / 3
# print (temp_r, temp_c)
# pic1 = Bicubic(img, temp_r, temp_c)
# cv2.imwrite("./Set14/"+name+"1.bmp", pic1)
# img1 = cv2.imread("./Set14/"+name+"1.bmp")
# pic2 = Bicubic(img1, src_r, src_c)
# cv2.imwrite("./Set14/"+name+"2.bmp", pic2)
