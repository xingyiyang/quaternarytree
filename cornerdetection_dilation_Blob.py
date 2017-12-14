# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from skimage import io,data,color
from skimage.measure import label,regionprops
import matplotlib.pyplot as plt

img = cv.imread('baidu_firefox.jpg')

#转成灰度图
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#为了不丢失精度
gray=np.float32(gray)

"""
cornerHarris(img, blocksize, ksize, k [, dst [, borderType ]]) -> dst

img 输入图像，数据类型为float32
blockSize 角点检测当中的邻域值。
ksize 使用Sobel函数求偏导的窗口大小
k 角点检测参数，取值为0.04到0.06

"""
#dst是一个矩阵
dst=cv.cornerHarris(gray,2,3,0.04)

#kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
kernel = cv.getStructuringElement(cv.MORPH_RECT,(3, 3))
#膨胀harris结果
dst=cv.dilate(dst,kernel)

#对角点赋值颜色

img[dst>0.01*dst.max()]=[0,0,255]
img[dst<=0.01*dst.max()]=[0,0,0]
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


ret,img_binary=cv.threshold(img_gray,50,255,cv.THRESH_BINARY)

labels = label(img_binary, connectivity=2)

newimg=color.label2rgb(labels)

region = regionprops(labels)

for prop in region:
    print("bobox = {}, {} {} {} {}".format(prop.bbox,prop.bbox[0],prop.bbox[1],prop.bbox[2],prop.bbox[3]))
    cv.rectangle(newimg,(prop.bbox[1],prop.bbox[0]),(prop.bbox[3],prop.bbox[2]),(0,0,255),1)


cv.imshow('dst',newimg)
if cv.waitKey(0) & 0xff==27:
    cv.destroyAllWindows()

