#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:22:51 2017

@author: wen
"""

import numpy as np
from scipy.ndimage import filters, distance_transform_edt
from skimage.feature import peak_local_max
import skimage.morphology as morphology
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, random_walker


def watershed_2d(image_pred, z_range=21, min_distance=7):
    boundary=np.zeros(image_pred.shape,dtype='bool')
    for z in range(z_range):
        bn_image=image_pred[:,:,z]>0.5
        dist=distance_transform_edt(bn_image,sampling=[1,1])
        dist_smooth=filters.gaussian_filter(dist,2,mode='constant')
        
        local_maxi = peak_local_max(dist_smooth, min_distance=min_distance, indices=False)
        markers = morphology.label(local_maxi)
        labels_ws = morphology.watershed(-dist_smooth, markers, mask=bn_image)
        labels_bd=find_boundaries(labels_ws, connectivity=2, mode='outer', background=0)
        
        boundary[:,:,z]=labels_bd       
    
    bn_output=image_pred>0.5
    bn_output[boundary==1]=0
    
    return [bn_output,boundary]

def watershed_3d(image_watershed2d, samplingrate=[1,1,9.2],min_size=100,min_distance=1):
    
    dist=distance_transform_edt(image_watershed2d,sampling=samplingrate)    
    dist_smooth=filters.gaussian_filter(dist,(2,2,0.3),mode='constant')
    local_maxi = peak_local_max(dist_smooth, min_distance=min_distance, indices=False)
    markers = morphology.label(local_maxi)
    labels_ws = morphology.watershed(-dist_smooth, markers, mask=image_watershed2d)
    labels_clear=remove_small_objects(labels_ws, min_size=min_size,connectivity=3)
    
    labels_bd=find_boundaries(labels_clear, connectivity=3, mode='outer', background=0)
    labels_wo_bd = labels_clear.copy()
    labels_wo_bd[labels_bd==1]=0
    labels_wo_bd=remove_small_objects(labels_wo_bd, min_size=min_size,connectivity=3)
    
    return [labels_wo_bd, labels_clear]

def watershed_markers(image_watershed2d, markers, samplingrate=[1,1,9.2], min_size=100):

    dist=distance_transform_edt(image_watershed2d,sampling=samplingrate)    
    dist_smooth=filters.gaussian_filter(dist,(4,4,0.5),mode='constant')

    labels_ws = morphology.watershed(-dist_smooth, markers, mask=image_watershed2d, compactness=0.001)
    labels_clear=remove_small_objects(labels_ws, min_size=min_size,connectivity=3)
       
    return [labels_clear, dist_smooth]

def watershed_2d_markers(image_pred, mask, z_range=21):
    
    labels_ws=np.zeros(image_pred.shape,dtype='int')
    for z in range(z_range):
        bn_image= np.logical_or(image_pred[:,:,z]>0,mask[:,:,z]>1)
        markers = image_pred[:,:,z]
        markers[np.where(mask[:,:,z]>1)] = 0
        dist=distance_transform_edt(mask[:,:,z]>1,sampling=[1,1])
        labels_ws[:,:,z] = morphology.watershed(dist, markers, mask=bn_image)
       
    return labels_ws




