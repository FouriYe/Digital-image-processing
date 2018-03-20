# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:39:41 2017

@author: Administrator
"""

import random

def pepper(img,height = 256,width = 256,probility = 0.2):
    for i in range(height):
        for j in range(width):
            k = random.random()
            if(k<probility):
                img[i][j] = 255
    return img
        