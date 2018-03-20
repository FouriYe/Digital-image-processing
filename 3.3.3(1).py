# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:27:41 2017

@author: Zihan YE
"""
'''
参考文章
http://blog.sciencenet.cn/home.php?mod=space&uid=425437&do=blog&id=1052070
'''
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import math
import fourier as f

def compass(n,mode = 0):
    if n > 255:
        n = 255
    if n < 0:
        if mode == 0:
            n = 0
        elif mode == 1:
            n = -n
    return n

def d(u,v,cent_x,cent_y):
    return math.sqrt((u-cent_x)**2+(v-cent_y)**2)

'''
图像读取
'''
res = io.imread(r'.\image\peppers.bmp')
io.imshow(res)

plt.figure(figsize=(5,5))
plt.hist(res.flatten(),bins=256)
plt.xlabel('resource image')
plt.show()

'''
转换为double类型，取对数
'''
img = res.copy().astype("f8")
"""
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i][j] = math.log(img[i][j]+1,math.e)
"""

'''傅里叶变换,得到频谱'''
RP = np.zeros(img.shape,"f8")
IP = np.zeros(img.shape,"f8")
sp = f.fft2d(img,RP,IP,mode = 1)
io.imshow(sp)

plt.figure(figsize=(5,5))
plt.hist(sp.flatten(),bins=256)
plt.xlabel('sp')
plt.show()

'''
选择滤波器
'''

'''
同态滤波
'''

def GHPF(i,j,x0,y0,c,d0):
    return (1-math.exp(-c*(d(i,j,x0,y0)**2/(d0**2))))

def IHPF(i,j,x0,y0,c,d0):
    if d(i,j,x0,y0) >= d0:
        return 1
    else:
        return 0

def homo_filter(RP,IP,yL,yH,c,d0,PF):
    for i in range(RP.shape[0]):
        for j in range(RP.shape[1]):
            RP[i][j] *= (yH-yL)*PF(i,j,RP.shape[1]/2,RP.shape[0]/2,c,d0)+yL
    for i in range(IP.shape[0]):
        for j in range(IP.shape[1]):
            IP[i][j] *= (yH-yL)*PF(i,j,IP.shape[1]/2,IP.shape[0]/2,c,d0)+yL

yH = 2.2
yL = 0.25
c = 2
d0 = 300
"""
homo_filter(RP,IP,yL,yH,c,d0,IHPF)
"""
for i in range(RP.shape[0]):
    for j in range(RP.shape[1]):
        RP[i][j] *= GHPF(i,j,RP.shape[1]/2,RP.shape[0]/2,c,d0)
for i in range(IP.shape[0]):
    for j in range(IP.shape[1]):
        IP[i][j] *= GHPF(i,j,IP.shape[1]/2,IP.shape[0]/2,c,d0)
'''图像重建'''
inv = f.ifft2d(RP,IP,"f8",mode = 1)

'''
取指数,转回uint8类型
'''
"""
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        inv[i][j] = math.exp(inv[i][j])
"""
inv = inv.astype("uint8")
io.imshow(inv)

plt.figure(figsize=(5,5))
plt.hist(inv.flatten(),bins=256)
plt.xlabel('after')
plt.show()
