# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:24:20 2017

@author: Administrator
"""

from skimage import io
import matplotlib.pyplot as plt
import my_template as tp
import noise
import math

height = 256
width= 256
 
def compass(n,mode = 0):
    if n > 255:
        n = 255
    if n < 0:
        if mode == 0:
            n = 0
        elif mode == 1:
            n = -n
    return n

def solve(img,T,size = 3):
    offset  = (int)((size-1)/2)
    img2 = [[0 for x in range(width)] for y in range(height)]
    for i in range(height):
        for j in range(width):
            img2[i][j] = img[i][j]
    for i in range(height):
        for j in range(width):
            if 0 + (size-1)/2 <= i <= height-1 - (size-1)/2:
                if 0 + (size-1)/2 <= j <= width-1 - (size-1)/2:
                    sum = 0
                    for k in range(size):
                        for g in range(size):
                            sum += img[i+k-offset][j+g-offset]*T[k][g]
                    img2[i][j] = compass(sum,1)
    for i in range(height):
        for j in range(width):
            img[i][j] = img2[i][j]
    return img

img = io.imread(r'.\image\lena256.bmp')
io.imshow(img)

plt.figure(figsize=(5,5))
plt.hist(img.flatten(),bins=256)
plt.xlabel('before')
plt.show()

'''
img = noise.pepper(img,probility = 0.01)
io.imshow(img)

plt.figure(figsize=(5,5))
plt.hist(img.flatten(),bins=256)
plt.xlabel('pepper')
plt.show()
'''

img = solve(img,tp.sobelx,3)+solve(img,tp.sobely,3)

io.imshow(img)

plt.figure(figsize=(5,5))
plt.hist(img.flatten(),bins=256)
plt.xlabel('after')
plt.show()