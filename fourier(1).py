# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:39:41 2017

@author: Administrator
"""

import math
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def ft1d(f,RP,IP):
    F = np.zeros(f.shape,f.dtype)
    
    for u in range(f.shape[0]):
        R = 0.0
        I = 0.0
        for i in range(f.shape[0]):
            R += (math.cos(-2*math.pi*(u*i/f.shape[0]))*f[i])
            I += (math.sin(-2*math.pi*(u*i/f.shape[0]))*f[i])
        RP[u] = R
        IP[u] = I
        temp = math.sqrt(math.pow(R,2)+math.pow(I,2))
        F[u] = int(temp)
    return F

def ift1d(RP,IP,dtype="uint8"):
    f = np.zeros(RP.shape,dtype)
    
    for x in range(RP.shape[0]):
        temp = 0
        for u in range(RP.shape[0]):
            temp += (RP[u]*math.cos(2*math.pi*(u*x/RP.shape[0]))-IP[u]*math.sin(2*math.pi*(u*x/RP.shape[0])) )
        temp /= RP.shape[0]
        f[x] = temp

    return f

def ft2d(img,RP,IP):
    spectrum = np.zeros(img.shape,img.dtype) 
    for u in range(img.shape[0]):
        for v in range(img.shape[1]):
            '''if v == 0:
                print(v)
            elif 100 == v:
                print(v)
            elif 200 == v:
                print(v)'''
            R = 0.0
            I = 0.0
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    R += (math.cos(-2*math.pi*(u*i/img.shape[0]+v*j/img.shape[1]))*img[i][j])
                    I += (math.sin(-2*math.pi*(u*i/img.shape[0]+v*j/img.shape[1]))*img[i][j])
            RP[u][v] = R
            IP[u][v] = I
            temp = math.sqrt(math.pow(R,2)+math.pow(I,2))
            '''temp = math.log10(temp+1)'''
            spectrum[u][v] = int(temp)

    return spectrum
    


def ift2d(RP,IP,dtype="uint8"):
    img2 = np.zeros(RP.shape,dtype)
            
    for x in range(RP.shape[0]):
        for y in range(RP.shape[1]):
            temp = 0
            for u in range(RP.shape[0]):
                for v in range(RP.shape[1]):
                    temp += (RP[u][v]*math.cos(2*math.pi*(u*x/RP.shape[0]+v*y/RP.shape[1]))-IP[u][v]*math.sin(2*math.pi*(u*x/RP.shape[0]+v*y/RP.shape[1])))
            temp /= (RP.shape[0]*RP.shape[1])
            img2[x][y] = temp
            
    return img2


def fft1d(fR,fI,RP,IP):
    spectrum = np.zeros(fR.shape,fR.dtype)
    FR = fR.astype("f8")
    FI = fI.astype("f8")
    f2R = np.zeros(fR.shape,"f8")
    f2I = np.zeros(fI.shape,"f8")
    '''
    得到数字二进制位数
    '''
    p = 0
    N = fR.shape[0]
    while(N>0):
        N>>=1
        p = p+1
    p = p-1
    N = fR.shape[0]
    '''
    交换f(x)次序
    '''
    for i in range(N):
        order = i
        b = bin(order)
        b = b[2:]
        b = "0"*(p-len(b))+b
        b = b[::-1]
        order = int(b, 2)  
        if order > i:
            FR[i],FR[order] = FR[order],FR[i]
            FI[i],FI[order] = FI[order],FI[i]
    '''
    预先算出Wn节省时间
    '''
    WR = np.zeros(fR.shape[0]//2,"f8")
    WI = np.zeros(fI.shape[0]//2,"f8")
    for i in range(WR.shape[0]):
        WR[i] = math.cos(-2*math.pi*i/N)
        WI[i] = math.sin(-2*math.pi*i/N)
    '''
    蝶式算法主体
    '''
    for i in range(int(math.log(N,2))):
        index = 0
        unit = 2**(i+1)
        f2R = FR.copy()
        f2I = FI.copy()
        FR = np.zeros(FR.shape,"f8")
        FI = np.zeros(FI.shape,"f8")
        while(index<fR.shape[0]):
            for j in range(index,index+unit//2):
                FR[j] += f2R[j]
                FI[j] += f2I[j]
                FR[j+unit//2] += f2R[j]
                FI[j+unit//2] += f2I[j]
            for j in range(index+unit//2,index+unit):
                ad = 2**(int(math.log(N,2))-i-1)
                base = j-index-unit//2
                FR[j-unit//2] += (WR[base*ad]*f2R[j]-WI[base*ad]*f2I[j])
                FI[j-unit//2] += (WR[base*ad]*f2I[j]+WI[base*ad]*f2R[j])
                FR[j] -= (WR[base*ad]*f2R[j]-WI[base*ad]*f2I[j])
                FI[j] -= (WR[base*ad]*f2I[j]+WI[base*ad]*f2R[j])
            index += unit
    '''
    算出频谱图
    '''
    for i in range(fR.shape[0]):
        spectrum[i] = math.sqrt(math.pow(FR[i],2)+math.pow(FI[i],2))
    '''
    变换后的F(x),RP是实部，IP虚部
    '''
    for i in range(FR.shape[0]):
        RP[i] = FR[i]
        IP[i] = FI[i]
    return spectrum.astype("uint8")

def ifft1d(RP,IP,dtype = "f8"):
    f = np.zeros(RP.shape,dtype)
    fR = RP.astype("f8")
    fI = IP.astype("f8")
    f2R = np.zeros(RP.shape,"f8")
    f2I = np.zeros(IP.shape,"f8")
    '''
    得到数字二进制位数
    '''
    p = 0
    N = RP.shape[0]
    while(N>0):
        N>>=1
        p = p+1
    p = p-1
    N = RP.shape[0]
    
    for i in range(N):
        order = i
        b = bin(order)
        b = b[2:]
        b = "0"*(p-len(b))+b
        b = b[::-1]
        order = int(b, 2)  
        if order > i:
            fR[i],fR[order] = fR[order],fR[i]
            fI[i],fI[order] = fI[order],fI[i]
    WR = np.zeros(RP.shape[0]//2,"f8")
    WI = np.zeros(IP.shape[0]//2,"f8")
    for i in range(WR.shape[0]):
        WR[i] = math.cos(2*math.pi*i/N)
        WI[i] = math.sin(2*math.pi*i/N)
    for i in range(int(math.log(N,2))):
        index = 0
        unit = 2**(i+1)
        f2R = fR.copy()
        f2I = fI.copy()
        fR = np.zeros(RP.shape,"f8")
        fI = np.zeros(IP.shape,"f8")
        while(index<f.shape[0]):
            for j in range(index,index+unit//2):
                fR[j] += f2R[j]
                fI[j] += f2I[j]
                fR[j+unit//2] += f2R[j]
                fI[j+unit//2] += f2I[j]
            for j in range(index+unit//2,index+unit):
                ad = 2**(int(math.log(N,2))-i-1)
                base = j-index-unit//2
                fR[j-unit//2] += (WR[base*ad]*f2R[j]-WI[base*ad]*f2I[j])
                fI[j-unit//2] += (WR[base*ad]*f2I[j]+WI[base*ad]*f2R[j])
                fR[j] -= (WR[base*ad]*f2R[j]-WI[base*ad]*f2I[j])
                fI[j] -= (WR[base*ad]*f2I[j]+WI[base*ad]*f2R[j])
            index += unit
    for i in range(RP.shape[0]):
        RP[i] = fR[i]/N
    for i in range(IP.shape[0]):
        IP[i] = fI[i]/N
    for i in range(f.shape[0]):
        f[i] = fR[i]/N
    return f.astype(dtype)



def fft2d(img,RP,IP,mode = 0):

    spectrum = np.zeros(img.shape,img.dtype)
    f = img.T
    for i in range(f.shape[0]):
        spectrum[i] = fft1d(f[i],np.zeros(f[i].shape,f[i].dtype),RP[i],IP[i])
    RP = RP.T
    IP = IP.T
    for i in range(spectrum.shape[0]):
        spectrum[i] = fft1d(RP[i],IP[i],RP[i],IP[i])
    if mode == 1:
        spectrum = shift(spectrum)
    return spectrum.astype("uint8")

def ifft2d(RP,IP,dtype = "uint8",mode = 0):
    if mode == 1:
        RP = shift(RP)
        IP = shift(IP)
    img2 = np.zeros(RP.shape,dtype)
    RP2 = RP
    IP2 = IP
    for i in range(img2.shape[0]):
        ifft1d(RP2[i],IP2[i])
    RP2 = RP2.T
    IP2 = IP2.T
    for i in range(img2.shape[0]):
        img2[i] = ifft1d(RP2[i],IP2[i]).astype(dtype)
    return img2

def shift(img):
    xl = img.shape[0]//2
    yl = img.shape[1]//2
    for i in range(yl):
        for j in range(xl):
            img[i][j],img[i+yl][j+xl] = img[i+yl][j+xl],img[i][j]
    for i in range(yl):
        for j in range(xl):
            img[i+yl][j],img[i][j+xl] = img[i][j+xl],img[i+yl][j]
    return img

def paintsquare(img):
    img2 = np.zeros(img.shape,img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = 0
            if img.shape[0]*0.125 <= i <= img.shape[0]*0.86:
                if img.shape[1]*0.36 <= j <= img.shape[1]*0.64:
                    img[i][j] = 255

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img2[i][j] = img[i][j]
            
    
    return img2



'''
img = io.imread(r'.\image\lena256.bmp')
io.imshow(img)
plt.figure(figsize=(5,5))
plt.hist(img.flatten(),bins=256)
plt.xlabel('before')
plt.show()

inv = paintsquare(img)
io.imshow(inv)
    
plt.figure(figsize=(5,5))
plt.hist(inv.flatten(),bins=256)
plt.xlabel('after')
plt.show()
'''

"""
'''
一维傅立叶测试代码
'''
f = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],"f8")
RP = np.zeros((25,),"f8")
IP = np.zeros((25,),"f8")
ft1d(f,RP,IP)
f = ift1d(RP,IP,"f8")
print(f)
f2 = f.astype("uint8")
print(f2)
"""

"""
'''
二维傅立叶测试代码
'''
img = np.zeros((64,64),"uint8") 
img = paintsquare(img)

io.imshow(img)

plt.figure(figsize=(5,5))
plt.hist(img.flatten(),bins=256)
plt.xlabel('after')
plt.show()

RP = np.zeros(img.shape,"f8")
IP = np.zeros(img.shape,"f8")
sp = ft2d(img,RP,IP)
io.imshow(sp)

plt.figure(figsize=(5,5))
plt.hist(sp.flatten(),bins=256)
plt.xlabel('after')
plt.show()

inv = ift2d(RP,IP)
io.imshow(inv)
    
plt.figure(figsize=(5,5))
plt.hist(inv.flatten(),bins=256)
plt.xlabel('after')
plt.show()
"""

"""
'''
一维快速傅立叶测试代码
'''
f = np.array([1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1],"uint8")
RP = np.zeros(f.shape,"f8")
IP = np.zeros(f.shape,"f8")
F2 = fft1d2(f,np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"uint8"),RP,IP)
print("\nRP = ",RP,"\nIP = ",IP)
F = ft1d(f,RP,IP)
print("\nRP = ",RP,"\nIP = ",IP)
print(ift1d(RP,IP))
print(ifft1d(RP,IP))
"""


'''
二维快速傅立叶测试代码
'''
"""
img = io.imread(r'.\image\peppers.bmp')

RP = np.zeros(img.shape,"f8")
IP = np.zeros(img.shape,"f8")
sp = fft2d(img,RP,IP)
img2 = ifft2d(RP,IP)
f2 = ift1d(RP[100],IP[100])
f3 = ifft1d(RP[100],IP[100])
io.imshow(img2)
"""

"""
'''
shift测试代码
'''
img = io.imread(r'.\image\peppers.bmp')
img = shift(img)
RP = np.zeros(img.shape,"f8")
IP = np.zeros(img.shape,"f8")
sp = fft2d(img,RP,IP)
img2 = ifft2d(RP,IP)
img2 = shift(img2)
io.imshow(img2)
"""