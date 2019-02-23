# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import numpy.linalg as LA

LR_patchs_size = 7
half_LR_patchs_size = LR_patchs_size / 2
LR_patchs_dem = 45
HR_patchs_size = 12
half_HR_patchs_size = HR_patchs_size / 2
HR_patchs_dem = 144
cluster_times = 512
train_pic_num = 42 #pic_num + 1 save total LRpatchs_num
patchs_num = 800
filenamelist = "LRimagelist.txt"

def operat_img(filepath):
    for name in os.listdir(filepath):
        if name[-4:] == ".jpg":
            print (name)
            img = cv2.imread(filepath + "/" + name)
            Y = RGB2YCrCb(img)
            LRpatchs(Y)

def LRpatchs(filepath):
    LRpatchs_num = 1000000
    LR_feature = np.empty((LRpatchs_num, 49))
    HR_feature = np.empty((LRpatchs_num, 144))
    center_point = np.empty((LRpatchs_num, 2))
    LR_mean = np.empty((LRpatchs_num))
    pic_boundary = np.empty(42, dtype=int)
    index = 0
    boundary_index = 0
    fobj = open(filenamelist, 'w')
    for name in os.listdir(filepath):
        if name[-4:] == ".jpg":
            fobj.write(name + "\n")
            pic_boundary[boundary_index] = index
            boundary_index = boundary_index + 1
            print name
            img = cv2.imread(filepath + "/" + name)
            img1 = cv2.imread("./Train/" + name)
            Y = RGB2YCrCb(img)
            Y1 = RGB2YCrCb(img1)
            for i in range(half_LR_patchs_size + 1, Y.shape[0] - half_LR_patchs_size):
                for j in range(half_LR_patchs_size + 1, Y.shape[1] - half_LR_patchs_size):
                    if index < LRpatchs_num:
                        center_point[index, :] = [i, j]
                        r_center = int(i * 3)
                        c_center = int(j * 3)
                        LR_patch = Y[i - half_LR_patchs_size:i + half_LR_patchs_size + 1, \
                                   j - half_LR_patchs_size:j + half_LR_patchs_size + 1]
                        HR_patch = Y1[r_center - half_HR_patchs_size:r_center + half_HR_patchs_size,\
                                   c_center - half_HR_patchs_size:c_center + half_HR_patchs_size]
                        print HR_patch.shape
                        mean_value = np.sum(LR_patch)
                        # mean_value = mean_value - (LR_patch[0][0] + LR_patch[0][6] + LR_patch[6][0] + LR_patch[6][6])
                        mean_value = mean_value / 49
                        LR_mean[index] = mean_value
                        LR_patch = LR_patch - mean_value
                        HR_patch = HR_patch - mean_value
                        vector_LR = LR_patch.flatten()
                        vector_HR = HR_patch.flatten()
                        LR_feature[index, :] = vector_LR
                        HR_feature[index, :] = vector_HR
                        index = index + 1
                        print (index, boundary_index)
                    else:
                        np.save("LRpatch.npy", LR_feature)
                        np.save("HRpatch.npy", HR_feature)
                        np.save("LR_mean.npy", LR_mean)
                        np.save("center_point.npy", center_point)
                        pic_boundary[boundary_index] = LRpatchs_num
                        np.save("pic_boundary.npy", pic_boundary)
                        fobj.close()
                        return 0


def Cluster(array):
    print ("begin Cluster")
    Kmean_modle = KMeans(n_clusters=512, max_iter=100)
    s = Kmean_modle.fit(array)
    np.save("cluster_center.npy", s.cluster_centers_)
    np.save("cluster_labels.npy", s.labels_)
    print (s.cluster_centers_)


def Coef_Cau():
    LR_patchs = np.load("LRpatch.npy")
    HR_patchs = np.load("HRpatch.npy")
    cluster_lable = np.load("cluster_labels.npy")
    V = np.zeros((cluster_times, 49, patchs_num))
    W = np.zeros((cluster_times, 144, patchs_num))
    for cluster_flag in range(cluster_times):
        count = 0
        for i in range(1000000):
            if cluster_lable[i] == cluster_flag:
                if count < patchs_num:
                    V[cluster_flag, :, count] = LR_patchs[i, :]
                    W[cluster_flag, :, count] = HR_patchs[i, :]
                    count = count + 1
        print count
    np.save("LR_V.npy", V)
    np.save("HR_W.npy", W)

def Cluster_Coef():
    V = np.load("LR_V.npy")
    W = np.load("HR_W.npy")
    V_2 = np.zeros((50, patchs_num))
    Coef_of_Cluster = np.zeros((cluster_times, 144, 50))
    for cluster_flag in range(cluster_times):
        V_temp = V[cluster_flag, :, :] # 49 * 1000
        for i in range(50):
            for j in range(patchs_num):
                if i < 49:
                    V_2[i][j] = V_temp[i][j]
                else:
                    V_2[i][j] = 1 # 50 * 1000
        W_temp = W[cluster_flag, :, :] # 144 * 1000
        P = LA.pinv(V_2.T)
        P3 = np.dot(P, W_temp.T)
        P3 = P3.T
        Coef_of_Cluster[cluster_flag, :, :] = P3
        print cluster_flag
    np.save("Cluster_Coef.npy", Coef_of_Cluster)


def RGB2YCrCb(img):
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0]
    return Y

def Lablefeature(array, clustercenter):
    lable = []
    for i in range(array[1]):
        cur_feature = array[:,i]
        dif = np.tile(cur_feature, [clustercenter.shape[1], 1]) - clustercenter
        ls_temp = np.sqrt(np.sum(dif*2,2))
        lable.append(np.min())
        return lable


LRpatchs("./DownSampling")
LR_cluster = np.load("LRpatch.npy")
print "begin cluster"
Cluster(LR_cluster)
Coef_Cau()
Cluster_Coef()