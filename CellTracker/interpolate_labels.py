#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:42:01 2017

@author: wen
"""

from skimage.filters import gaussian
import numpy as np

def gaussian_filter(img, z_scaling=10, smooth_sigma=5):
    """
    Generate smoothed label image of cells
    Input: 
        img: label image
        z_scaling: factor of interpolations along z axis, should be <10
        smooth_sigma: sigma used for making Gaussian blur
    Return:
        output_img: Generated smoothed label image
        mask: mask image indicating the overlapping of multiple cells (0: background;
        1: one cell; >1: multiple cells)
    """
    img_10x = np.repeat(img, z_scaling, axis=2)
    shape_10x = img_10x.shape
    labels_num = np.max(img)
    output_img = np.zeros((shape_10x[0]+10,shape_10x[1]+10,shape_10x[2]+10),dtype='int')
    mask = output_img.copy()
    
    for label in range(1, labels_num+1):
        print(label, end=" ")
        x,y,z = np.where(img_10x==label)
        xi, yi, zi = np.mgrid[x.min()-5:x.max()+6, y.min()-5:y.max()+6, z.min()-5:z.max()+6]
        d = np.zeros(xi.shape)
        d[x-x.min()+5,y-y.min()+5,z-z.min()+5]=0.5
        
        percentage = 1-np.divide(np.size(x),np.size(d),dtype='float')
              
        img_smooth = gaussian(d, sigma=smooth_sigma, mode='constant')
        
        threshold = np.percentile(img_smooth, percentage*100)
        
        new_label_temp = img_smooth>threshold
        
        output_img[x.min():x.max()+11, y.min():y.max()+11, z.min():z.max()+11] = output_img[x.min():x.max()+11, 
             y.min():y.max()+11, z.min():z.max()+11] + new_label_temp*label
        mask[x.min():x.max()+11, y.min():y.max()+11, z.min():z.max()+11] = mask[x.min():x.max()+11, 
             y.min():y.max()+11, z.min():z.max()+11] + new_label_temp*1
    
    return [output_img, mask]

