# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import math

def invert(img):
    for i in range(256):
        for j in range(256):
            img[i][j] = (255-img[i][j])
    return img

def extend(img):
    '''
                                x/2            0<=x<64 
    默认对比度拓展函数为:f(x) =    1.48x-62.788   64<=x<194
                                x/2+127.5       194<=x<=255
    '''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 0 <= img[i][j] < 64:
                img[i][j] = max(0,(int)(img[i][j]/2+0.5))
            elif 64 <= img[i][j] < 194 :
                img[i][j] = min(255,(int)(1.48*img[i][j]-62.788 + 0.5))
            elif 194 <= img[i][j] < 256 :
                img[i][j] = min(255,(int)(img[i][j]/2 + 127.5 + 0.5))
    return img

'''
灰度拉伸,灰度映射0-255,多为傅立叶变换表示频谱图时用
'''
def extend2(img):
    maxv = 0
    minv = 255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > maxv:
                maxv = img[i][j]
            if img[i][j] < minv:
                minv = img[i][j]
    step = 255/(maxv-minv)    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = (img[i][j]-minv)*step
    return img

def slipe(img):
    '''
    默认抑制范围为[0,64)U[194,256]
    '''
    for i in range(256):
        for j in range(256):
            if 0 <= img[i][j] < 64:
                img[i][j] = 0
            elif 194 <= img[i][j] < 256 :
                img[i][j] = 0
    return img

def bin_div(img):
    '''
    默认阈值为127
    '''
    T = 127
    for i in range(256):
        for j in range(256):
            if img[i][j] <= T:
                img[i][j] = 0
            elif T < img[i][j] :
                img[i][j] = 255
    return img

def noLinearP1(img,b = 1.09058,c = 0.00041819385):
    '''
    指数变化 s = T(r) = c*b**r
    b>1时，图像右凸，低灰度级被压缩，高灰度级被拉伸
    '''
    for i in range(256):
        for j in range(256):
            img[i][j] = min(255,(int)(c * math.pow(b,img[i][j])))
    return img

def noLinearP2(img,b = 0.99558,c = 200):
    '''
    指数变化 s = T(r) = c*b**r
    b<1时，图像左凸，低灰度级被拉伸，高灰度级被压缩
    '''
    for i in range(256):
        for j in range(256):
            img[i][j] = min(255,(int)(c * math.pow(b,img[i][j])))
    return img

def noLinearLog1(img,b = 1.05,c = 1):
    '''
    对数变化 s = T(r) = c*logb(1+r)
    b>1时，图像上凸，低灰度级被拉伸，高灰度级被压缩
    '''
    for i in range(256):
        for j in range(256):
            img[i][j] = min(255,(int)(c * math.log(img[i][j]+1,b)))
    return img

def noLinearLog2(img,b = 0.995,c = 1):
    '''
    对数变化 s = T(r) = c*logb(1+r)
    b<1时，图像下凸，低灰度级被拉伸，高灰度级被压缩
    '''
    for i in range(256):
        for j in range(256):
            img[i][j] = min(255,(int)(c * math.log(img[i][j]+1,b)))
    return img

def noLinearY1(img,y = 2,c = 0.01075):
    '''
    幂函数 s = T(r) = c*pow(r,y)
    y校正就是幂函数的反函数
    y>1时，图像下凸，低灰度级被压缩，高灰度级被拉伸
    '''
    for i in range(256):
        for j in range(256):
            img[i][j] = min(255,(int)(c * math.pow(img[i][j],y)))
    return img

def noLinearY2(img,y = 0.9,c = 0.9945):
    '''
    幂函数 s = T(r) = c*pow(r,y)
    y校正就是幂函数的反函数
    y<1时，图像上凸，低灰度级被拉伸，高灰度级被压缩
    '''
    for i in range(256):
        for j in range(256):
            img[i][j] = min(255,(int)(c * math.pow(img[i][j],y)))
    return img


def tranform(img,f):
    return f(img)

img = io.imread(r'.\image\peppers.bmp')
io.imshow(img)

plt.figure(figsize=(5,5))
plt.hist(img.flatten(),bins=256)
plt.xlabel('before')
plt.show()

img = tranform(img,extend2)

io.imshow(img)

plt.figure(figsize=(5,5))
plt.hist(img.flatten(),bins=256)
plt.xlabel('after')
plt.show()