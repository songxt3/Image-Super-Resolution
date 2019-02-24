#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

filenamelist = "Imagelist.txt"

def getlist(imgpath):
    fobj = open(filenamelist, 'w')
    img_name = [0]*(8)
    flag = 0
    for name in os.listdir(imgpath):
        if name[-4:] == ".jpg":
            img_name.append(name)
            flag += 1
            fobj.write(str(flag) + name + "\n")
            print(name)
    np.save("Imagelist.npy", img_name)
    fobj.close()

getlist("./DownSampling")

file = np.load("Imagelist.npy")
print file