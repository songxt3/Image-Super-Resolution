import cv2
import numpy as np

# def W(x):
#     x = np.abs(x)
#     if 0 <= x <= 1 :
#         return 1.5 * x * x * x - 2.5 * x * x + 1
#     if 1 < x <= 2 :
#         return -0.5 * x * x * x + 2.5 * x * x - 4 * x + 2
#     else :
#         return 0
#
# def adjust(value):
#     if value > 255:
#         value = 255
#     elif value < 0:
#         value = 0
#     return value
#
# def bicubic(img, m, n):
#     height,width,channels = img.shape
#     emptyImage = np.zeros((m, n, channels), np.uint8)
#     sh = height / m
#     sw = width / n
#     for i in range(m):
#         for j in range(n):
#             x = i * sh
#             y = j * sw
#             # print(x, y)
#             p = i * sh - int(x)
#             q = j * sw - int(y)
#             # print(p, q)
#             x = int(x) - 2
#             y = int(y) - 2
#             A = np.array([[W(1 + p), W(p), W(1 - p), W(2 - p)]])
#
#             if x >= m - 3:
#                 m - 1
#             print(n)
#             if y >= n - 3:
#                 n - 1
#                 print("-1", n)
#             if 1 <= x <= m - 3 and 1 <= y <= n -3 :
#                 B = np.array([[img[x - 1, y - 1], img[x - 1, y],img[x - 1, y + 1],img[x - 1, y + 1]],[img[x, y - 1],
#                 img[x, y],img[x, y + 1], img[x, y + 2]],[img[x + 1, y - 1], img[x + 1, y],img[x + 1, y + 1],
#                 img[x + 1, y + 2]],[img[x + 2, y - 1], img[x + 2, y],img[x + 2, y + 1], img[x + 2, y + 1]],])
#
#                 C = np.array([[W(1 + q)], [W(q)], [W(1 - q)], [W(2 - q)]])
#
#                 blue = np.dot(np.dot(A, B[:, :, 0]), C)[0, 0]
#                 green = np.dot(np.dot(A, B[:, :, 1]), C)[0, 0]
#                 red = np.dot(np.dot(A, B[:, :, 2]), C)[0, 0]
#
#                 blue = adjust(blue)
#                 green = adjust(green)
#                 red = adjust(red)
#                 emptyImage[i][j] = np.array([blue, green, red], dtype=np.uint8)
#     return emptyImage
#
#



def RGB2YCrCb(img):
    m = img.shape[0]
    n = img.shape[1]
    Y = np.zeros((m, n), dtype=np.uint8)
    Cb = np.zeros((m, n), dtype=np.uint8)
    Cr = np.zeros((m, n), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            Y[i][j] = 0.257 * img[i][j][2] + 0.564 * img[i][j][1] + 0.098 * img[i][j][0] + 16
            Cb[i][j] = -0.148 * img[i][j][2] - 0.291 * img[i][j][1] + 0.439 * img[i][j][0] + 128
            Cr[i][j] = 0.439 * img[i][j][2] - 0.368 * img[i][j][1] + 0.071 * img[i][j][0] + 128
            # Y[i][j] = controlRGB(Y[i][j])
            # Cb[i][j] = controlRGB(Cb[i][j])
            # Cr[i][j] = controlRGB(Cr[i][j])
    YCrCb = np.dstack([Y, Cr, Cb])
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
            R[i][j] = controlRGB(R[i][j])
            G[i][j] = controlRGB(G[i][j])
            B[i][j] = controlRGB(B[i][j])
    bgr = np.dstack([B, G, R])
    return bgr

def controlRGB(num):
    if num > 255:
        num = 255
    elif num < 0:
        num = 0
    return num

img = cv2.imread("./Set14/lenna.bmp")
# YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
YCrCb = RGB2YCrCb(img)
img_ycc2 = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
# img1 = YCrCb2RGB(YCrCb)
# cv2.imwrite("cubic.bmp", img1)