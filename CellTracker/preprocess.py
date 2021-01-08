#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:12:41 2017

@author: wen
"""

from scipy import ndimage
import numpy as np
import keras
from keras.models import Model
from keras.layers import Conv3D,  Input


def lcn(img3d, filter_size=(3,27,27)):
    """
    Local contrast normalization by cpu
    Input: 
        img3d: raw 3D image
        filter_size: the window size to apply constrast normalization (z_siz, x_siz, y_siz)
    Return:
        the normalized image
    """
    filter=np.ones(filter_size)
    filter=filter/filter.size    
    avg=ndimage.convolve(img3d,filter,mode='reflect')
    diff_sqr=np.square(img3d-avg)
    std=np.sqrt(ndimage.convolve(diff_sqr,filter,mode='reflect'))
    norm=np.divide(img3d-avg,std+5)    
    return norm


def conv3d_keras(filter_size, img3d_siz):
    """
    Generate a keras model for applying 3D convolution
    Input: 
        filter_size: the window size to apply constrast normalization (z_size, x_size, y_size)
        img3d_siz: the size of the image (x_size, y_size, z_size)
    Return:
        result: the keras model
    """
    inputs=Input((img3d_siz[2],img3d_siz[0],img3d_siz[1],1))    
    conv_3d=Conv3D(1,filter_size, kernel_initializer = keras.initializers.Ones(), padding='same')(inputs)    
    result=Model(inputs=inputs, outputs=conv_3d)      
    return result


def lcn_gpu(img3d, noise=5, filter_size=(3,27,27), img3d_siz=(512,1024,21)):
    """
    Local contrast normalization by gpu
    Input: 
        img3d: raw 3D image
        noise: a tiny value (close to the background noise) added to the denominator
        filter_size: the window size to apply constrast normalization (z_size, x_size, y_size)
        img3d_siz: the size of the image (x_size, y_size, z_size)
    Return:
        the normalized image
    """
    volume = filter_size[0]*filter_size[1]*filter_size[2]
    conv3d_model = conv3d_keras(filter_size, img3d_siz)
    img3d = img3d.reshape((1,img3d_siz[2],img3d_siz[0],img3d_siz[1],1))
    avg=conv3d_model.predict(img3d)/volume
    diff_sqr=np.square(img3d-avg)
    std=np.sqrt(conv3d_model.predict(diff_sqr)/volume)
    norm=np.divide(img3d-avg,std+noise)
    return norm[0,:,:,:,0]

