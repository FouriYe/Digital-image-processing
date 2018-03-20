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

'''
图像读取
'''
res = io.imread(r'.\image\lena256.bmp')
res = f.paintsquare(res)
io.imshow(res)

plt.figure(figsize=(5,5))
plt.hist(res.flatten(),bins=256)
plt.xlabel('resource image')
plt.show()

'''
取对数
'''
img = res.copy().astype("f8")
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i][j] = math.log(img[i][j]+1,math.e)
'''

'''
R = 0.0
I = 0.0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        R += (math.cos(-2*math.pi*(0*i/img.shape[0]+0*j/img.shape[1]))*img[i][j])
        I += (math.sin(-2*math.pi*(0*i/img.shape[0]+0*j/img.shape[1]))*img[i][j])
print(R)



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
def d(u,v,cent_x,cent_y):
    return math.sqrt((u-cent_x)**2+(v-cent_y)**2)

'''
同态滤波
'''
yH = 10
yL = 0.1
c = 2
d0 = 100
print(RP[0][0])
"""
for i in range(RP.shape[0]):
    for j in range(RP.shape[1]):
        RP[i][j] *= (yH-yL)*(1-math.exp(-c*(d(i,j,RP.shape[0]/2,RP.shape[1]/2)**2/(d0**2))))+yL
for i in range(IP.shape[0]):
    for j in range(IP.shape[1]):
        IP[i][j] *= (yH-yL)*(1-math.exp(-c*(d(i,j,IP.shape[0]/2,IP.shape[1]/2)**2/(d0**2))))+yL
print(RP[0][0])
"""
"""
for i in range(RP.shape[0]):
    for j in range(RP.shape[1]):
        if d(i,j,RP.shape[0]/2,RP.shape[1]/2) < d0:
            RP[i][j] = 0
for i in range(IP.shape[0]):
    for j in range(IP.shape[1]):
        if d(i,j,IP.shape[0]/2,IP.shape[1]/2) < d0:
            IP[i][j] = 0
"""
'''图像重建'''
inv = f.ifft2d(RP,IP,"f8",mode = 1)

'''
取指数
'''

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        inv[i][j] = math.exp(inv[i][j])
inv = inv.astype("uint8")
io.imshow(inv)

plt.figure(figsize=(5,5))
plt.hist(inv.flatten(),bins=256)
plt.xlabel('after')
plt.show()
