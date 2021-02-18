#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 2019

@author: wen

"""

#####################################
# import packages and functions
#####################################
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.ndimage.measurements as snm
from keras.models import Model, load_model
from PIL import Image
from skimage.segmentation import relabel_sequential
import tensorflow as tf
from keras.backend import tensorflow_backend

from CellTracker.preprocess import lcn_gpu
from CellTracker.unet3d import unet3_a, unet3_prediction
from CellTracker.watershed import watershed_2d, watershed_3d, \
    watershed_2d_markers
from CellTracker.interpolate_labels import gaussian_filter
from CellTracker.track import initial_matching, pr_gls, transform_cells, \
    tracking_plot, tracking_plot_zx

%matplotlib qt

#######################################
# global parameters
#######################################
# parameters according to imaging conditions
volume_num = 171 # number of volumes the 3D + T image
x_siz,y_siz,z_siz = 512, 1024, 21 # size of each 3D image
z_xy_resolution_ratio = 9.2 # the resolution ratio between the z axis and the x-y plane 
                            # (does not need to be very accurate)
z_scaling = 10 # (integer) for interpolating images along z. z_scaling = 1 makes no interpolation.
               # z_scaling > 1 generates smoother images.
shrink = (24,24,2) # pad and shrink for u-net prediction, corresponding to (x,y,z). Large values
                   # lead to more accurate segmentations, but it should be less than (input sizes of u-net)/2.

# parameters manually determined by experience
noise_level = 20 # a threshold to discriminate noise/artifacts from cells
min_size = 100 # a threshold to remove small objects which may be noise/artifacts
BETA = 300 # control coherence using a weighted average of movements of nearby points;
           # larger BETA includes more points, thus generates more coherent movements
LAMBDA = 0.1 # control coherence by adding a loss of incoherence, large LAMBDA 
             # generates larger penalty for incoherence.
max_iteration = 20 # maximum number of iterations; large values generate more accurate tracking.

# folder path, file names
folder_path = './Projects/3DeeCellTracker-master/Demo_Tracking' # path of the folder storing related files
files_name = "aligned_t%03i_z%03i.tif" # file names for the raw image files
unet_weight_file = "unet3_f2b_weights.hdf5" # weight file of the trained 3D U-Net. f2b:structure a; fS2a:b; fS2b:c
FFN_weight_file = "FNN_model.h5" # weight file of the trained FFN model

####################################################
# Create folders for storing data and results
####################################################
raw_image_path = os.path.join(folder_path,"data/")
if not os.path.exists(raw_image_path):
    os.makedirs(raw_image_path)

auto_segmentation_vol1_path = os.path.join(folder_path,"auto_vol1/")
if not os.path.exists(auto_segmentation_vol1_path):
    os.mkdir(auto_segmentation_vol1_path)

manual_segmentation_vol1_path = os.path.join(folder_path,"manual_vol1/")
if not os.path.exists(manual_segmentation_vol1_path):
    os.mkdir(manual_segmentation_vol1_path)
    
manual_name = "manual_labels%03i.tif"

track_results_path = os.path.join(folder_path,"track_results/")
if not os.path.exists(track_results_path):
    os.mkdir(track_results_path)
    
track_information_path = os.path.join(folder_path,"track_information/")
if not os.path.exists(track_information_path):
    os.mkdir(track_information_path)

models_path = os.path.join(folder_path,"models/")
if not os.path.exists(models_path):
    os.mkdir(models_path)
    
########################################################################################################
# users should move raw image files to the "data" folder created above with names defined in files_name,
# and move trained u-net model weight and FFN model weight files to the "models" folder
########################################################################################################

#######################################
# load the pre-trained 3D U-net model
#######################################
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

unet_model = unet3_a()
unet_model.load_weights(os.path.join(models_path,unet_weight_file))

############################################
# generate automatic segmetation of volume 1
############################################
    
def segmentation(vol):
    """
    Make segmentation (unet + watershed)
    Input:
        vol: a specific volume
    Gloabal variables:
        center_coordinates: center coordinates of segmented cells by watershed
        image_cell_bg: the cell/background regions obtained by unet.
        layer_num: number of the layers in the 3D image
        segmentation_auto: individual cells segmented by watershed
        image_gcn: raw image / 65536
    """
    global center_coordinates, image_cell_bg, layer_num, segmentation_auto, image_gcn
    
    # read raw 3D image of a specific volume
    t = time.time()
    image_raw=[]
    layer_num = z_siz
    for z in range(1,layer_num+1):
        path = raw_image_path+files_name%(vol,z)
        image_raw.append(cv2.imread(path, -1))
    
    # pre-processing: local contrast normalization
    image_norm = np.array(image_raw)
    image_gcn = image_norm.copy()/65536.0 # this intensity is used to correct tracking results
    image_gcn = image_gcn.transpose(1,2,0)
    background_pixles = np.where(image_norm<np.median(image_norm))
    image_norm = image_norm-np.median(image_norm)
    image_norm[background_pixles]=0
    image_norm = lcn_gpu(image_norm, noise_level, img3d_siz=(x_siz,y_siz,z_siz))
    image_norm = image_norm.reshape(1,z_siz,x_siz,y_siz,1)
    image_norm = image_norm.transpose(0,2,3,1,4)
    elapsed = time.time() - t
    print('pre-processing took %.1f s'%elapsed)
    
    # predict cell-like regions using 3D U-net
    t = time.time()
    image_cell_bg = unet3_prediction(image_norm,unet_model,shrink=shrink)
    
    # segment connected cell-like regions using watershed
    [image_watershed2d_wo_border, border]=watershed_2d(image_cell_bg[0,:,:,:,0],z_range=z_siz, min_distance=7)
    image_watershed3d_wo_border, image_watershed3d_wi_border, _, _ = watershed_3d(image_watershed2d_wo_border,
                    samplingrate=[1,1,z_xy_resolution_ratio], method="min_size", 
                    min_size=min_size, neuron_num=0, min_distance=1)
    segmentation_auto, fw, inv = relabel_sequential(image_watershed3d_wi_border)
    
    # calculate coordinates of the centers of each segmented cell
    center_coordinates = snm.center_of_mass(segmentation_auto>0,segmentation_auto, range(1, segmentation_auto.max()+1))
    elapsed = time.time() - t
    print('segmentation took %.1f s'%elapsed)

# segment 3D image of volume #1
segmentation(1)
# save the segmented cells of volume #1
for z in range(1,layer_num+1):
    auto_segmentation = (segmentation_auto[:,:,z-1]).astype(np.uint8)
    Image.fromarray(auto_segmentation).save(auto_segmentation_vol1_path+"auto_R_t%04i_z%04i.tif"%(1,z))   
    
# calculate coordinates of the current "previous" volume (here is #1) 
coordinates_pre = np.asarray(center_coordinates)
coordinates_pre_real=coordinates_pre.copy()
coordinates_pre_real[:,2]=coordinates_pre[:,2]*z_xy_resolution_ratio

#################################
# test tracking parameters
#################################
# load weights of the feedforward network 
FNN_model = load_model(os.path.join(models_path,FFN_weight_file))

def test_tracking(vol1, vol2):
    """
    Test whether the parameters for tracking are proper (generating figures for the transformation)
    Input: 
        vol1, vol2: the numbers of two volumes for testing the registration between them
    """
    segmentation(vol1)
    coordinates_pre = np.asarray(center_coordinates)
    coordinates_pre_real=coordinates_pre.copy()
    coordinates_pre_real[:,2]=coordinates_pre[:,2]*z_xy_resolution_ratio
    
    segmentation(vol2)
    coordinates_post = np.asarray(center_coordinates)
    coordinates_post_real=coordinates_post.copy()
    coordinates_post_real[:,2]=coordinates_post[:,2]*z_xy_resolution_ratio
    
    init_match = initial_matching(FNN_model, coordinates_pre_real, coordinates_post_real, 20)

    [P, pre_transformation, C] = pr_gls(coordinates_pre_real, 
        coordinates_post_real, init_match, BETA=BETA, max_iteration=max_iteration, 
        LAMBDA=LAMBDA)
    
    tracking_plot(coordinates_pre_real, coordinates_post_real, pre_transformation)
    tracking_plot_zx(coordinates_pre_real, coordinates_post_real, pre_transformation)
    
"""
test_tracking(1,2) # choose two neighboring time points with challenging (large) movements
"""
######################################################################################
# users should manually correct the automatic segmentation of volume 1 in other soft, 
# e.g. in ITK-SNAP
######################################################################################

#######################################################################
# load manually corrected segmentation of volume 1 and interpolate it 
#######################################################################
    
# load manually corrected segmentation
layer_num = z_siz
segmentation_manual = []
for i in range(0, layer_num):
    path = manual_segmentation_vol1_path+manual_name%i
    segmentation_manual.append(cv2.imread(path, -1))

# relabeling indexes of manually corrected neurons
segmentation_manual = np.array(segmentation_manual)
segmentation_manual = segmentation_manual.transpose(1,2,0)
segmentation_manual_relabels, fw, inv = relabel_sequential(segmentation_manual)

# interpolate layers in z axis
print("interpolating...")
seg_cells_interpolated, seg_cell_or_bg = gaussian_filter(segmentation_manual_relabels,z_scaling=z_scaling, smooth_sigma=2.5)
seg_cells_interpolated_corrected = watershed_2d_markers(seg_cells_interpolated, seg_cell_or_bg, z_range=z_siz*z_scaling+10)
seg_cells_interpolated_corrected = seg_cells_interpolated_corrected[5:x_siz+5,5:y_siz+5,5:z_siz*z_scaling+5]
 
# save labels in the first volume (interpolated)
for z in range((z_scaling+1)//2,seg_cells_interpolated_corrected.shape[2]+1,z_scaling):
    cells_vol1 = (seg_cells_interpolated_corrected[:,:,z-1]).astype(np.uint8)
    Image.fromarray(cells_vol1).save(track_results_path+"track_results_t%03i_z%03i.tif"%(1,z/z_scaling))    
    
# calculate coordinates of centers (the corrected coordinates of cells in the first volume)
center_points0 = snm.center_of_mass(segmentation_manual_relabels>0,segmentation_manual_relabels, 
                                    range(1, segmentation_manual_relabels.max()+1))
coordinates_tracked = np.asarray(center_points0)
coordinates_tracked_real=coordinates_tracked.copy()
coordinates_tracked_real[:,2]=coordinates_tracked[:,2]*z_xy_resolution_ratio

# save a copy of the coordinates in volume 1
coordinates_tracked_real_vol1 = coordinates_tracked_real.copy()

###############################################################################
# tracking following volumes
###############################################################################

# initialize all variables
displacement_from_vol1 = coordinates_tracked_real*0
cells_on_boundary = displacement_from_vol1[:,0].astype(int)
displacements=[]
segmented_coordinates=[]; segmented_coordinates.append(coordinates_pre_real)
tracked_coordinates=[]; tracked_coordinates.append(coordinates_tracked_real)

# segmentation and tracking in following volumes
t_1 = time.time()
for volume in range(2,volume_num+1):
    
    print('t=%i'%volume)
    
    ########################################################
    # generate automatic segmentation in current volume
    ########################################################
    segmentation(volume)
    
    t = time.time()
    coordinates_post = np.asarray(center_coordinates)
    coordinates_post_real=coordinates_post.copy()
    coordinates_post_real[:,2]=coordinates_post[:,2]*z_xy_resolution_ratio
    
    ###############################################################################
    # use the pre-trained FNN model to calculate the initial matching
    ###############################################################################
 
    # predict the initial matching using the feedforward network
    init_match = initial_matching(FNN_model, coordinates_pre_real, coordinates_post_real, 20)
    elapsed = time.time() - t
    print('initial matching by FNN took %.1f s'%elapsed)
    
    ########################################################################
    # use pr_gls method to calculate a coherent transformation function
    ########################################################################
    
    # estimate the transformation function using two points set: previous volume and current volume
    t = time.time()
    [P, pre_transformation, C] = pr_gls(coordinates_pre_real, 
        coordinates_post_real, init_match, BETA=BETA, max_iteration=max_iteration, 
        LAMBDA=LAMBDA)
    
    # apply the transformation function to calculate new coordinates of points set in previous volume (tracked coordinates)
    length_cells = np.size(coordinates_tracked_real, axis=0)
    length_auto_segmentation = np.size(coordinates_pre_real, axis=0)
    Gram_matrix = np.zeros((length_auto_segmentation,length_cells))
    for idx_i in range(length_cells):
        for idx_j in range(length_auto_segmentation):
            Gram_matrix[idx_j,idx_i] = np.exp(-np.sum(np.square(coordinates_tracked_real[idx_i,:]-coordinates_pre_real[idx_j,:]))/
                       (2*BETA*BETA)) 
    coordinates_prgls = np.matrix.transpose(np.matrix.transpose(coordinates_tracked_real)+np.dot(C,Gram_matrix))

    # calculate displacements from the first volume
    # displacement_from_vol1: accurate displacement; displacement_in_image: displacement using voxels numbers as unit
    displacement_from_vol1 = displacement_from_vol1 + coordinates_prgls-coordinates_tracked_real
    displacement_in_image = displacement_from_vol1.copy()
    displacement_in_image[:,2] = displacement_in_image[:,2]/z_xy_resolution_ratio
    displacement_in_image = np.rint(displacement_in_image).astype(int)

    # generate current image of labels from the manually corrected segmentation in volume 1
    tracked_cells_prgls, overlap_prgls = transform_cells(segmentation_manual_relabels,displacement_in_image)
    elapsed = time.time() - t
    print('estimate transformation function by pr-gls took %.1f s'%elapsed)
    
    ###################################
    # accurate correction 
    ###################################
    
    t = time.time()
    # delete those cells moved to the boundary areas of the image
    cells_bd = np.where(np.logical_or(np.logical_or(np.logical_or(coordinates_prgls[:,1]<5, coordinates_prgls[:,1]>y_siz-5), 
                           coordinates_prgls[:,0]<5), coordinates_prgls[:,0]>x_siz-5))
    print("cells on boundary:", cells_bd)
    cells_on_boundary[cells_bd]=1
    
    for i in np.where(cells_on_boundary==1)[0]:
        tracked_cells_prgls[np.where(tracked_cells_prgls==(i+1))]=0
    
    # overlapping regions of multiple cells are discarded before correction to avoid cells merging
    tracked_cells_prgls[np.where(overlap_prgls>1)]=0
    
    # accurate correction of displacement
    coordinates_prgls_int_move = coordinates_tracked_real_vol1*np.array([1,1,1/z_xy_resolution_ratio]) + displacement_in_image
    centers_unet_x_prgls = snm.center_of_mass(image_cell_bg[0,:,:,:,0]+image_gcn,tracked_cells_prgls, range(1, seg_cells_interpolated_corrected.max()+1))
    centers_prgls = np.asarray(coordinates_prgls_int_move)
    centers_unet_x_prgls = np.asarray(centers_unet_x_prgls)
    lost_cells = np.where(np.isnan(centers_unet_x_prgls)[:,0])
    print("cells lost (on boundary + merge):", lost_cells)
    displacement_correction = centers_unet_x_prgls - centers_prgls
    displacement_correction[lost_cells,:] = 0
    displacement_correction[:,2]=displacement_correction[:,2]*z_xy_resolution_ratio
    
    # calculate the corrected displacement from vol #1 
    displacement_from_vol1 = displacement_in_image*np.array([1,1,z_xy_resolution_ratio]) + displacement_correction
    displacement_in_image = displacement_from_vol1.copy()
    displacement_in_image[:,2] = displacement_in_image[:,2]*z_scaling/z_xy_resolution_ratio
    displacement_in_image = np.fix(displacement_in_image).astype(int)

    # generate current image of labels (more accurate)
    tracked_cells_corrected, overlap_corrected =  transform_cells(seg_cells_interpolated_corrected, displacement_in_image)
    elapsed = time.time() - t
    print('accurate correction of displacement took %.1f s'%elapsed)
    
    # discard those lost cells
    for i in np.where(cells_on_boundary==1)[0]:
        tracked_cells_corrected[np.where(tracked_cells_corrected==(i+1))]=0
    
    # re-calculate boundaries of overlapped cells using watershed
    tracked_cells_corrected[np.where(overlap_corrected>1)]=0
    label_T_watershed = watershed_2d_markers(tracked_cells_corrected[:,:,z_scaling//2:z_siz*z_scaling:z_scaling], 
                                             overlap_corrected[:,:,z_scaling//2:z_siz*z_scaling:z_scaling], z_range=z_siz)
    
    ####################################################    
    # save tracked labels and tracking information
    ####################################################
    for z in range(1,layer_num+1):
        tracked_cells = (label_T_watershed[:,:,z-1]).astype(np.uint8)
        Image.fromarray(tracked_cells).save(track_results_path+"track_results_t%04i_z%04i.tif"%(volume,z))
    
    # update and save points locations
    coordinates_pre_real = coordinates_post_real.copy()
    coordinates_tracked_real = coordinates_tracked_real_vol1 + displacement_in_image*np.array([1,1,z_xy_resolution_ratio/z_scaling])
    
    displacements.append(displacement_from_vol1)
    segmented_coordinates.append(coordinates_pre_real)
    tracked_coordinates.append(coordinates_tracked_real)
elapsed = time.time() - t_1
print('tracking all volumes took %.1f s'%elapsed)
# 5415s (50vol); 18021s (171vol)
    
# save tracking information in the folder    
np.save(track_information_path+"displacements",displacements)
np.save(track_information_path+"segmented_coordinates",segmented_coordinates)
np.save(track_information_path+"tracked_coordinates",tracked_coordinates)
