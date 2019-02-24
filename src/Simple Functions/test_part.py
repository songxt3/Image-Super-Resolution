# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

LR_patchs_size = 7
half_LR_patchs_size = LR_patchs_size / 2
cluster_times = 512
img_h = 288
img_l = 352
HR_patchs_size = 12
half_HR_patchs_size = HR_patchs_size / 2

name = "test2"
img1 = cv2.imread("./Set14/" + name + ".bmp") #获取原图的长宽
target_M = img1.shape[0]
target_N = img1.shape[1]

def getLrpatchs(Y):
    cluster_center = np.load("cluster_center.npy")
    coef_matrix = np.load("Cluster_Coef.npy")
    dis_rec = np.empty(cluster_times)
    m = Y.shape[0]
    n = Y.shape[1]
    print (m, n)
    HR_patch_intensity = np.empty((m * n, 12, 12))
    LR_upsize = np.zeros((50, 1))
    for i in range(0, m * n):
        r = int(np.mod(i, m))
        c = int(np.ceil(i / m))
        r1 = int(r + LR_patchs_size - 1)
        c1 = int(c + LR_patchs_size - 1)
        # print (r, r1, c, c1)
        if r1 >= m - 1 and c1 <= n - 1: # m-1 = 95 && n-1 = 116
            r1_t = r1 - m + 1
            LR_patch = Y[r - r1_t:r1 - r1_t + 1, c:c1+1]
            # print ("1", LR_patch.shape)
        elif r1 <= m - 1 and c1 >= n - 1:
            c1_t = c1 - n + 1
            LR_patch = Y[r:r1 + 1, c - c1_t:c1 - c1_t + 1]
            # print ("2", LR_patch.shape)
        elif r1 >= m - 1 and c1 >= n - 1:
            c1_t = c1 - n + 1
            r1_t = r1 - m + 1
            LR_patch = Y[r - r1_t:r1 - r1_t + 1, c - c1_t:c1 - c1_t + 1]
        else:
            LR_patch = Y[r:r1+1, c:c1+1]
            # print ("4", LR_patch.shape)
        mean_value = np.sum(LR_patch)
        mean_value = mean_value / 49
        LR_patch = LR_patch - mean_value
        vector_LR = LR_patch.flatten()
        for k1 in range(cluster_times):
            dis_rec[k1] = distance(vector_LR, cluster_center[k1, :])
        min_dis = min(dis_rec)
        cluster_index = np.argwhere(dis_rec == min_dis)  # min_dis cluster point
        for t1 in range(50):
            if t1 < 49:
                LR_upsize[t1][0] = vector_LR[t1]
            else:
                LR_upsize[t1][0] = 1
        cluster_index = int(cluster_index)
        HR_patch = np.dot(coef_matrix[cluster_index, :, :], LR_upsize)
        HR_patch_intensity[i] = (HR_patch + mean_value).reshape(HR_patchs_size, HR_patchs_size)
    np.save("HRpatchs_test.npy", HR_patch_intensity)



def rec_img(YCrCb):
    HRpathchs = np.load("HRpatchs_test.npy")
    print HRpathchs.shape
    img_hr_ext_sum = np.zeros((target_M, target_N))
    img_hr_ext_count = np.ones((target_M, target_N))
    target_N_1 = target_N / 3
    target_M_1 = target_M / 3
    for i in range(1, target_M_1 * target_N_1):
        r = np.mod(i, target_M_1)
        c = np.ceil(i / target_M_1)
        rh = int((r) * 3)
        rh1 = int(rh + HR_patchs_size - 1)
        ch = int((c) * 3)
        ch1 = int(ch + HR_patchs_size - 1)
        # print (rh, rh1, ch, ch1)
        if rh < target_M and ch < target_N and rh1 < target_M and ch1 < target_N:
            # print "1"
            img_hr_ext_sum[rh:rh1 + 1, ch:ch1 + 1] = img_hr_ext_sum[rh:rh1 + 1, ch:ch1 + 1] + HRpathchs[i , :]
            img_hr_ext_count[rh:rh1 + 1, ch:ch1 + 1] = img_hr_ext_count[rh:rh1 + 1, ch:ch1 + 1] + 1
        elif (rh < target_M and rh1 > target_M) and ch1 < target_N:
            # print "2"
            pos_rh = rh1 - target_M + 1
            img_hr_ext_sum[rh - pos_rh:rh1 - pos_rh + 1, ch:ch1 + 1] = img_hr_ext_sum[rh - pos_rh:rh1 - pos_rh + 1, ch:ch1 + 1] + HRpathchs[i, :]
            img_hr_ext_count[rh - pos_rh:rh1 - pos_rh + 1, ch:ch1 + 1] = img_hr_ext_count[rh - pos_rh:rh - pos_rh + 1, ch:ch1 + 1] + 1
        elif (ch < target_N and ch1 > target_N) and rh1 < target_M:
            # print "3"
            pos_ch = ch1 - target_N + 1
            img_hr_ext_sum[rh:rh1 + 1, ch - pos_ch:ch1 - pos_ch + 1] = img_hr_ext_sum[rh:rh1 + 1, ch - pos_ch:ch1 - pos_ch + 1] + HRpathchs[i, :]
            img_hr_ext_count[rh:rh1 + 1, ch - pos_ch:ch1 - pos_ch + 1] = img_hr_ext_count[rh:rh1 + 1, ch - pos_ch:ch1 - pos_ch + 1] + 1
        elif (rh < target_M and rh1 < target_M) and (ch1 >= target_N):
            pos_ch = ch - 1
            pos_ch1 = ch1 - 1
            img_hr_ext_sum[rh:rh1 + 1, pos_ch:pos_ch1 + 1] = img_hr_ext_sum[rh:rh1 + 1, pos_ch:pos_ch1 + 1] + HRpathchs[i, :]
            img_hr_ext_count[rh:rh1 + 1, pos_ch:pos_ch1 + 1] = img_hr_ext_count[rh:rh1 + 1, pos_ch:pos_ch1 + 1] + 1
            # print "6"
        elif (ch < target_N and ch1 > target_N) and (rh < target_M and rh1 > target_M):
            # print "4"
            pos_ch = ch1 - target_N + 1
            pos_rh = rh1 - target_M + 1
            img_hr_ext_sum[rh - pos_rh:rh1 - pos_rh + 1, ch - pos_ch:ch1 - pos_ch + 1] = img_hr_ext_sum[rh - pos_rh:rh1 - pos_rh + 1, ch - pos_ch:ch1 - pos_ch + 1] + HRpathchs[i, :]
            img_hr_ext_count[rh - pos_rh:rh1 - pos_rh + 1, ch - pos_ch:ch1 - pos_ch + 1] = img_hr_ext_count[rh - pos_rh:rh1 - pos_rh + 1, ch - pos_ch:ch1 - pos_ch + 1] + 1
        else:
            pass
    img_hr_ext_sum = img_hr_ext_sum / img_hr_ext_count

    new_Cr = Bicubic(YCrCb[:, :, 1], target_M, target_N)
    new_Cb = Bicubic(YCrCb[:, :, 2], target_M, target_N)
    new_YCrCb = np.dstack([img_hr_ext_sum, new_Cr, new_Cb])
    new_img = YCrCb2RGB(new_YCrCb)
    return new_img


def distance(vec1,vec2):
    return np.linalg.norm((vec1 - vec2), 2)

def getYch(img):
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0]
    return Y


def RGB2YCrCb(img):
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return YCrCb


def YCrCb2RGB(YCrCb):
    m = YCrCb.shape[0]
    n = YCrCb.shape[1]
    R = np.zeros((m, n), dtype=int)
    G = np.zeros((m, n), dtype=int)
    B = np.zeros((m, n), dtype=int)
    for i in range(m):
        for j in range(n):
            R[i][j] = 1.164 * (YCrCb[i][j][0] - 16) + 1.596 * (YCrCb[i][j][1] - 128)
            G[i][j] = 1.164 * (YCrCb[i][j][0] - 16) - 0.813 * (YCrCb[i][j][1] - 128) - 0.392 * (YCrCb[i][j][2] - 128)
            B[i][j] = 1.164 * (YCrCb[i][j][0] - 16) + 2.017 * (YCrCb[i][j][2] - 128)
            controlRGB(R[i][j])
            controlRGB(G[i][j])
            controlRGB(B[i][j])
    bgr = np.dstack([B, G, R])
    return bgr

def controlRGB(num):
    if num > 255:
        num = 255
    elif num < 0:
        num = 0
    return num

def Bicubic(array, desH, desW):
    res = np.zeros((desH,desW))
    m = array.shape[0]
    n = array.shape[1]
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
            last_one = 0
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
                        tB = last_one
                    else:
                        tB = array[int(r + k1 - 1)][int(c + k2 - 1)]
                        last_one = tB
                    res[i][j] += tB * Wx * Wy
    return res

def getWvalue(x):
    x = abs(x)
    if x <= 1:
        temp = float((1.5 * x - 2.5) * x * x + 1)
    elif x <= 2:
        temp = float(((-0.5 * x + 2.5) * x - 4) * x + 2)
    else:
        temp = 0
    return temp


img = cv2.imread("./DownSamplingTest/" + name + "1.bmp") # 下采样图片

time_start = time.time()
Y = getYch(img)
getLrpatchs(Y)

YCrCb = RGB2YCrCb(img)
my_img = rec_img(YCrCb)
time_end = time.time()
print time_end - time_start

B = my_img[:, :, 0]
G = my_img[:, :, 1]
R = my_img[:, :, 2]

B_ = np.power(B, 0.98)
G_ = np.power(G, 0.98)
R_ = np.power(R, 0.98)

my_img = np.dstack([B_, G_, R_])

cv2.imwrite("./testDown/" + name + "2.bmp", my_img) # 新图片