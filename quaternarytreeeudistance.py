# -*- coding: utf-8 -*-
import functools
from PIL import Image, ImageDraw
import numpy as np
import math
import random
from enum import Enum

"""
一个矩形区域的象限划分：:
          
       UL(0)   |    UR(1)
     ----------|-----------
       LL(2)   |    LR(3)
"""
class QuadrantEnum(Enum):
    UL = 0
    UR = 1
    LL = 2
    LR = 3

"""
四叉树的类
一个根节点下有四个节点
"""
class QuaternaryNode(object):
    def __init__(self,img,box,dis,avg):
        self.img = img
        self.box = box
        self.dis = dis
        self.avg = avg
        self.ul = None
        self.ur = None
        self.ll = None
        self.lr = None

    def setErr(self,err):
        self.err = err
    def getErr(self):
        return self.err

    def setAvg(self,avg):
        self.avg = avg
    def getAvg(self):
        return self.avg

    #先序遍历，先是根节点，然后从左至右遍历节点
    def preorder_visit(self, visit, node=None):
        if node is None:
            node = self
        visit('x={}, y={}, w={}, h={}, dis={} coloravg={}'.format(node.box[0],node.box[1],node.box[2],node.box[3],node.dis,node.avg))
        if node.ul is not None:
            self.preorder_visit(visit, node.ul)
        if node.ur is not None:
            self.preorder_visit(visit, node.ur)
        if node.ll is not None:
            self.preorder_visit(visit, node.ll)
        if node.lr is not None:
            self.preorder_visit(visit, node.lr)

    #把图片转变成一棵四叉树
    @staticmethod
    def build_img_tree(quatuple):
        quatree = QuaternaryNode()
        #print('{} @ {}'.format(tree[0], btree))

        #左上
        if len(tree[2]) != 0:
            quatree.uperLeft = QuaternaryTree.build_list_tree(tree[2])

        #右上
        if len(tree[1]) != 0:
            quatree.uperRight = QuaternaryTree.build_list_tree(tree[1])

        #左下
        if len(tree[3]) != 0:
            quatree.lowerLeft = QuaternaryTree.build_list_tree(tree[3])

        #右下
        if len(tree[4]) != 0:
            quatree.lowerRight = QuaternaryTree.build_list_tree(tree[4])

        return quatree

"""
计算单个通道的均值
"""
def calSingleColor(imgarray,rgb):
    sumrgb = 0
    for i in range(imgarray.shape[0]):
        for j in range(imgarray.shape[1]):
            sumrgb += imgarray[i, j, rgb]
    return sumrgb*3/imgarray.size

"""
分别计算图像3个通道的均值，返回list
RGB三通道分别对应0，1，2
"""
def calculateImgRGB(img, x, y, w, h):
    avgRGB = [0]*3
    imgarray = np.array(img.crop((x,y,x+w,y+h)))
    
    #计算R通道
    avgRGB[0] = calSingleColor(imgarray, 0)
    #计算G通道
    avgRGB[1] = calSingleColor(imgarray, 1)
    #计算B通道
    avgRGB[2] = calSingleColor(imgarray, 2)

    return avgRGB


"""
计算图像的欧式距离的均值
"""
def euDistance(img, x, y, w, h):
    imgarray = np.array(img.crop((x,y,x+w,y+h)))
    avgRGB = calculateImgRGB(img, x, y, w, h)
    listAverage = np.array(avgRGB)
    distanceAll = 0
    for i in range(imgarray.shape[0]):
        for j in range(imgarray.shape[1]):
            r = imgarray[i,j,0]
            g = imgarray[i,j,1]
            b = imgarray[i,j,2]
            listpixel = np.array([r,g,b])
            distanceAll += np.sqrt(np.sum(np.square(listpixel-listAverage)))
    return distanceAll*3/imgarray.size

"""
给图片的四边画一个红框
"""
def SurroundEdge(img):
    # 获取图片的宽和长
    img_w, img_h = img.size
    #可编辑的图片
    img_d = ImageDraw.Draw(img)

    #四周画根红线
    img_d.line(((0, 0), (0, img_h)), (255, 0, 0))
    img_d.line(((0, 0), (img_w, 0)), (255, 0, 0))

    img_d.line(((0, img_h-1), (img_w, img_h-1)), (255, 0, 0))
    img_d.line(((img_w-1, 0), (img_w-1, img_h)), (255, 0, 0))

"""
把一个矩形框分割成4块，在中间画两条线
"""
def drawcenterline(img, x, y, w, h):
    img_d = ImageDraw.Draw(img)
    #横轴
    img_d.line(((x, y+h/2), (x+w, y+h/2)), (255,0,0))
    #纵轴
    img_d.line(((x+w/2, y), (x+w/2, y+h)), (255,0,0))

"""
把图像切割成4份，返回分割好的图像
   1   |    2 
-------|----------- 
   3   |    4 
"""
def splitPic(img, numId, x, y):
    w, h = img.size
    #print('split w={},h={},x={},y={}'.format(w,h,x,y))
    if (numId == 1):
        return img.crop((x, y, x+w/2, y+h/2))
    elif (numId == 2):
        return img.crop((x+w/2, y, x+w, y+h/2))
    elif (numId == 3):
        return img.crop((x, y+h/2, x+w/2, y+h))
    elif (numId == 4):
        return img.crop((x+w/2, y+h/2, x+w, y+h))
    else:
        print("分割错误")
        return

"""
判断图像的欧式距离，如果大于阈值，则划线分割成4份
"""
def buildline(img, x, y, w, h, count):
    dis = euDistance(img, x, y, w, h)
    if dis < 110:
        return None
    
    qt = QuaternaryNode(img, [x, y, w, h], dis, calculateImgRGB(img, x, y, w, h))

    print('x={} y={} w={} h={} distance={} count={}'.format(x,y,w,h,dis,count))
    
    drawcenterline(img, x, y, w, h)
    count += 1
    #print('{} count = {}'.format('=' * 10, count))
    if count<5:

        qt.ul = buildline(img, x, y, w/2, h/2, count)
        
        qt.ur = buildline(img, x+w/2, y, w/2, h/2, count)
     
        qt.ll = buildline(img, x, y+h/2,w/2, h/2, count)
       
        qt.lr = buildline(img, x+w/2, y+h/2,w/2, h/2, count)

    return qt
    
    
if __name__=='__main__':
    #打开图像
    img = Image.open('test.jpg').resize((256,256))
    imggray = img.convert('L')

    SurroundEdge(img)

    #初始化图片的左上角坐标
    x = 0
    y = 0
    count = 0
    w, h = img.size

    #根据欧式距离分割
    qt = buildline(img, x, y, w, h, count)
    
    img.save('quater.jpg')
    print()
    #遍历四叉树
    qt.preorder_visit(print)
    print('保存成功')
