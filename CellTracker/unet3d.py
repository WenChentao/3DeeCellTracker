#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:37:20 2017

@author: wen
"""

import numpy as np
from keras.models import Model
from keras.layers import Conv3D, LeakyReLU, Activation, Input, MaxPooling3D, \
    UpSampling3D, concatenate, BatchNormalization, Dense, Flatten, Dropout, \
    regularizers
import math


def unet3_a():
    """
    Generate a 3D unet model used in figure 2-S1a
    """
    inputs=Input((160,160,16,1))
    
    conv_2=Conv3D(8,3, padding='same')(inputs)
    conv_2=LeakyReLU()(conv_2)
    conv_2=BatchNormalization()(conv_2)
    
    conv_2b=Conv3D(16,3, padding='same')(conv_2)
    conv_2b=LeakyReLU()(conv_2b)
    conv_2b=BatchNormalization()(conv_2b)
    
    pool_2=MaxPooling3D(pool_size=(2,2,1))(conv_2b)
    
    conv_1=Conv3D(16,3, padding='same')(pool_2)
    conv_1=LeakyReLU()(conv_1)
    conv_1=BatchNormalization()(conv_1)
    
    conv_1b=Conv3D(32,3, padding='same')(conv_1)
    conv_1b=LeakyReLU()(conv_1b)
    conv_1b=BatchNormalization()(conv_1b)
    
    pool_1=MaxPooling3D(pool_size=(2,2,1))(conv_1b)
    
    conv_0=Conv3D(32,3, padding='same')(pool_1)
    conv_0=LeakyReLU()(conv_0)
    conv_0=BatchNormalization()(conv_0)
    
    conv_0b=Conv3D(64,3, padding='same')(conv_0)
    conv_0b=LeakyReLU()(conv_0b)
    conv_0b=BatchNormalization()(conv_0b)
    
    pool_0=MaxPooling3D(pool_size=(2,2,1))(conv_0b)
    
    conv_add=Conv3D(64,3, padding='same')(pool_0)
    conv_add=LeakyReLU()(conv_add)
    conv_add=BatchNormalization()(conv_add)
    
    conv_addb=Conv3D(64,3, padding='same')(conv_add)
    conv_addb=LeakyReLU()(conv_addb)
    conv_addb=BatchNormalization()(conv_addb)
    
    up0=concatenate([UpSampling3D(size=(2,2,1))(conv_addb),conv_0b])
    
    conv_add2=Conv3D(32,3, padding='same')(up0)
    conv_add2=LeakyReLU()(conv_add2)
    conv_add2=BatchNormalization()(conv_add2)
    
    conv_add2b=Conv3D(32,3, padding='same')(conv_add2)
    conv_add2b=LeakyReLU()(conv_add2b)
    conv_add2b=BatchNormalization()(conv_add2b)
    
    up1=concatenate([UpSampling3D(size=(2,2,1))(conv_add2b),conv_1b])
    
    conv1=Conv3D(16,3, padding='same')(up1)
    conv1=LeakyReLU()(conv1)
    conv1=BatchNormalization()(conv1)
    
    conv1b=Conv3D(16,3, padding='same')(conv1)
    conv1b=LeakyReLU()(conv1b)
    conv1b=BatchNormalization()(conv1b)
    
    up2=concatenate([UpSampling3D(size=(2,2,1))(conv1b),conv_2b])
    
    conv2=Conv3D(8,3, padding='same')(up2)
    conv2=LeakyReLU()(conv2)
    conv2=BatchNormalization()(conv2)
    
    conv2b=Conv3D(8,3, padding='same')(conv2)
    conv2b=LeakyReLU()(conv2b)
    conv2b=BatchNormalization()(conv2b)
    
    predictions=Conv3D(1,1,padding='same',activation='sigmoid')(conv2b)
    
    g_model=Model(inputs=inputs, outputs=predictions)
      
    return g_model


def unet3_b():
    """
    Generate a 3D unet model used in figure 2-S1b
    """
    inputs = Input((96, 96, 8, 1))

    conv_2 = Conv3D(64, 3, padding='same', activation='relu')(inputs)
    conv_2 = BatchNormalization()(conv_2)
    
    conv_2b = Conv3D(64, 3, padding='same', activation='relu')(conv_2)
    conv_2b = BatchNormalization()(conv_2b)
    
    pool_2 = MaxPooling3D(pool_size=(2, 2, 1))(conv_2b)
    
    conv_1 = Conv3D(128, 3, padding='same', activation='relu')(pool_2)
    conv_1 = BatchNormalization()(conv_1)
    
    conv_1b = Conv3D(128,3, padding='same',activation='relu')(conv_1)
    conv_1b = BatchNormalization()(conv_1b)
    
    pool_1 = MaxPooling3D(pool_size=(2, 2, 1))(conv_1b)
    
    conv_0 = Conv3D(256, 3, padding='same', activation='relu')(pool_1)
    conv_0 = BatchNormalization()(conv_0)
    
    conv_0b = Conv3D(256, 3, padding='same', activation='relu')(conv_0)
    conv_0b = BatchNormalization()(conv_0b)
    
    up1 = concatenate([UpSampling3D(size=(2, 2, 1))(conv_0b), conv_1b])
    
    conv1 = Conv3D(128, 3, padding='same', activation='relu')(up1)
    conv1 = BatchNormalization()(conv1)
    
    conv1b = Conv3D(128, 3, padding='same', activation='relu')(conv1)
    conv1b = BatchNormalization()(conv1b)
    
    up2 = concatenate([UpSampling3D(size=(2, 2, 1))(conv1b), conv_2b])
    
    conv2 = Conv3D(64, 3, padding='same', activation='relu')(up2)
    conv2 = BatchNormalization()(conv2)
    
    conv2b = Conv3D(64, 3, padding='same', activation='relu')(conv2)
    conv2b = BatchNormalization()(conv2b)
    
    predictions = Conv3D(1, 1, padding='same', activation='sigmoid')(conv2b)
    
    g_model = Model(inputs=inputs, outputs=predictions)
      
    return g_model


def unet3_c():
    """
    Generate a 3D unet model used in figure 2-S1c
    """
    inputs=Input((64,64,64,1))
    
    conv_2=Conv3D(8,3, padding='same')(inputs)
    conv_2=LeakyReLU()(conv_2)
    conv_2=BatchNormalization()(conv_2)
    
    conv_2b=Conv3D(16,3, padding='same')(conv_2)
    conv_2b=LeakyReLU()(conv_2b)
    conv_2b=BatchNormalization()(conv_2b)
    
    pool_2=MaxPooling3D(pool_size=(2,2,2))(conv_2b)
    
    conv_1=Conv3D(16,3, padding='same')(pool_2)
    conv_1=LeakyReLU()(conv_1)
    conv_1=BatchNormalization()(conv_1)
    
    conv_1b=Conv3D(32,3, padding='same')(conv_1)
    conv_1b=LeakyReLU()(conv_1b)
    conv_1b=BatchNormalization()(conv_1b)
    
    pool_1=MaxPooling3D(pool_size=(2,2,2))(conv_1b)
    
    conv_0=Conv3D(32,3, padding='same')(pool_1)
    conv_0=LeakyReLU()(conv_0)
    conv_0=BatchNormalization()(conv_0)
    
    conv_0b=Conv3D(64,3, padding='same')(conv_0)
    conv_0b=LeakyReLU()(conv_0b)
    conv_0b=BatchNormalization()(conv_0b)
    
    pool_0=MaxPooling3D(pool_size=(2,2,2))(conv_0b)
    
    conv_add=Conv3D(64,3, padding='same')(pool_0)
    conv_add=LeakyReLU()(conv_add)
    conv_add=BatchNormalization()(conv_add)
    
    conv_addb=Conv3D(64,3, padding='same')(conv_add)
    conv_addb=LeakyReLU()(conv_addb)
    conv_addb=BatchNormalization()(conv_addb)
    
    up0=concatenate([UpSampling3D(size=(2,2,2))(conv_addb),conv_0b])
    
    conv_add2=Conv3D(32,3, padding='same')(up0)
    conv_add2=LeakyReLU()(conv_add2)
    conv_add2=BatchNormalization()(conv_add2)
    
    conv_add2b=Conv3D(32,3, padding='same')(conv_add2)
    conv_add2b=LeakyReLU()(conv_add2b)
    conv_add2b=BatchNormalization()(conv_add2b)
    
    up1=concatenate([UpSampling3D(size=(2,2,2))(conv_add2b),conv_1b])
    
    conv1=Conv3D(16,3, padding='same')(up1)
    conv1=LeakyReLU()(conv1)
    conv1=BatchNormalization()(conv1)
    
    conv1b=Conv3D(16,3, padding='same')(conv1)
    conv1b=LeakyReLU()(conv1b)
    conv1b=BatchNormalization()(conv1b)
    
    up2=concatenate([UpSampling3D(size=(2,2,2))(conv1b),conv_2b])
    
    conv2=Conv3D(8,3, padding='same')(up2)
    conv2=LeakyReLU()(conv2)
    conv2=BatchNormalization()(conv2)
    
    conv2b=Conv3D(8,3, padding='same')(conv2)
    conv2b=LeakyReLU()(conv2b)
    conv2b=BatchNormalization()(conv2b)
    
    predictions=Conv3D(1,1,padding='same',activation='sigmoid')(conv2b)
    
    g_model=Model(inputs=inputs, outputs=predictions)
      
    return g_model


def unet3_prediction(input_image, model, shrink=(24, 24, 2)):
    """
    Predict cell/non-cell regions by applying 3D U-net on each sub-image. 
    Input:
        input_image: the raw image to be segmented.
        model: the pre-trained 3D U-net model.
        shrink: the surrounding voxles of each predicted sub-region to be discarded. 
    Return:
        out_img: predicted image of cell regions 
    Note: input image is expanded outside by "reflection" to predict the voxels near borders.
    """
    siz1, siz2, siz3 = model.output_shape[1]-shrink[0]*2, \
        model.output_shape[2]-shrink[1]*2, model.output_shape[3]-shrink[2]*2
    inputsiz1, inputsiz2, inputsiz3 = model.input_shape[1], \
        model.input_shape[2], model.input_shape[3]

    new_siz1 = int(math.ceil(input_image.shape[1]*1.0/siz1))*siz1
    new_siz2 = int(math.ceil(input_image.shape[2]*1.0/siz2))*siz2
    new_siz3 = int(math.ceil(input_image.shape[3]*1.0/siz3))*siz3
    
    pad1a, pad2a, pad3a = shrink[0], shrink[1], shrink[2]
    pad1b, pad2b, pad3b = new_siz1+pad1a-input_image.shape[1], \
        new_siz2+pad2a-input_image.shape[2], new_siz3+pad3a-input_image.shape[3]

    input_pad = np.pad(input_image[0, :, :, :, 0], ((pad1a, pad1b), (pad2a, pad2b),
                       (pad3a, pad3b)), 'reflect')
    input_pad = input_pad.reshape(1, input_pad.shape[0], input_pad.shape[1],
                                  input_pad.shape[2], 1)
    
    # the expanded image was predicted on each sub-image
    img = np.zeros((1, new_siz1, new_siz2, new_siz3, 1), dtype='float32')
    for i in range(0, new_siz1//siz1):
        for j in range(0, new_siz2//siz2):
            for k in range(0, new_siz3//siz3):
                temp_img = model.predict(
                        input_pad[:, i*siz1:i*siz1+inputsiz1,
                                  j*siz2:j*siz2+inputsiz2,
                                  k*siz3:k*siz3+inputsiz3, :])
                img[0, i*siz1:(i+1)*siz1, j*siz2:(j+1)*siz2,
                    k*siz3:(k+1)*siz3, 0] = temp_img[0, pad1a:pad1a+siz1,
                           pad2a:pad2a+siz2, pad3a:pad3a+siz3, 0]
    out_img = img[:, 0:input_image.shape[1], 0:input_image.shape[2], 0:input_image.shape[3], :]
    return out_img
