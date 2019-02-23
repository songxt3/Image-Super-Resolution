import cv2
import numpy as np

name = "ppt3"


img = cv2.imread("./testDown/" + name + "2.bmp")

B = img[:, :, 0]
G = img[:, :, 1]
R = img[:, :, 2]

B_ = np.power(B/1.0, 0.99)
G_ = np.power(G/1.0, 0.99)
R_ = np.power(R/1.0, 0.99)

bgr = np.dstack([B_, G_, R_])

cv2.imwrite("./testDown/" + name + "3.bmp", bgr)
