#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:12:41 2017

@author: wen
"""
import os
import cv2
from scipy import ndimage
import numpy as np
import keras
from keras.models import Model
from keras.layers import Conv3D, Input


def _make_folder(path_i, print_=True):
    """
    Make a folder and print its relative path
    Args:
        path_i: (str), the folder path

    Returns:
        str: return the same path
    """
    if not os.path.exists(path_i):
        os.makedirs(path_i)
    if print_:
        print(os.path.relpath(path_i, os.getcwd()))
    return path_i


def _get_files(folder_path):
    """
    Get paths of all files in the folder
    Args:
        folder_path: (str), the path of the folder containing images

    Returns:
        list: paths of all files
    """
    img_path = []
    for img_filename in sorted(os.listdir(folder_path)):
        img_path.append(folder_path + "/" + img_filename)
    return img_path


def load_image(folder_path, print_=True):
    """
    Load a 3D image from 2D layers (without time information)
    Args:
        folder_path: (str), the path of the folder containing images

    Returns:
        obj, numpy.ndarray: 3D image

    Notes:
        Currently only tested with 2D .tif files.
    """
    img_file_path = _get_files(folder_path)
    img = []
    for img_path in img_file_path:
        img.append(cv2.imread(img_path, -1))
    img_array = np.array(img).transpose((1, 2, 0))
    if print_:
        print("Load images with shape:", img_array.shape)
    return img_array


def lcn(img3d, noise=5, filter_size=(27, 27, 1)):
    """
    Local contrast normalization by cpu
    Input: 
        img3d: raw 3D image
        filter_size: the window size to apply constrast normalization (z_siz, x_siz, y_siz)
    Return:
        the normalized image
    """
    filter = np.ones(filter_size)
    filter = filter / filter.size
    avg = ndimage.convolve(img3d, filter, mode='reflect')
    diff_sqr = np.square(img3d - avg)
    std = np.sqrt(ndimage.convolve(diff_sqr, filter, mode='reflect'))
    norm = np.divide(img3d - avg, std + noise)
    return norm


def conv3d_keras(filter_size, img3d_siz):
    """
    Generate a keras model for applying 3D convolution
    Input: 
        filter_size: the window size to apply local contrast normalization
        img3d_siz: the size of the images
    Return:
        result: the keras model
    """
    inputs = Input((img3d_siz[0], img3d_siz[1], img3d_siz[2], 1))
    conv_3d = Conv3D(1, filter_size, kernel_initializer=keras.initializers.Ones(), padding='same')(inputs)
    result = Model(inputs=inputs, outputs=conv_3d)
    return result


def lcn_gpu(img3d, noise=5, filter_size=(27, 27, 1)):
    """
    Local contrast normalization by gpu
    Input: 
        img3d: raw 3D image
        noise: a tiny value (close to the background noise) added to the denominator
        filter_size: the window size to apply contrast normalization
    Return:
        the normalized image
    """
    img3d_siz = img3d.shape
    volume = filter_size[0] * filter_size[1] * filter_size[2]
    conv3d_model = conv3d_keras(filter_size, img3d_siz)
    img3d = np.expand_dims(img3d, axis=(0,4))
    avg = conv3d_model.predict(img3d) / volume
    diff_sqr = np.square(img3d - avg)
    std = np.sqrt(conv3d_model.predict(diff_sqr) / volume)
    norm = np.divide(img3d - avg, std + noise)
    return norm[0, :, :, :, 0]


def _normalize_image(image, noise_level):
    """
    Normalize an 3D image by local contrast normalization
    Args:
        image: (obj, numpy.ndarray), shape (x, y, z), 3D image
        noise_level: (float), the parameter controlling tiny noise to be ignored

    Returns:
        obj, numpy.ndarray: shape (x, y, z), the normalized image
    """
    image_norm = image - np.median(image)
    image_norm[image_norm < 0] = 0
    return lcn_gpu(image_norm, noise_level, filter_size=(27, 27, 1))


def _normalize_label(label_img):
    """Transform label image into 0/1, with shape (1, x, y, z, 1)"""
    return (label_img > 0).astype(int)