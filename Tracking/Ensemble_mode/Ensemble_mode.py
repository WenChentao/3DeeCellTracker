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
from scipy.stats import trim_mean
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.models import Model, load_model
from PIL import Image
from skimage.segmentation import relabel_sequential
from skimage.measure import label
from scipy.ndimage import median_filter

from CellTracker.preprocess import lcn_gpu
from CellTracker.unet3d import unet3_b, unet3_prediction
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
volume_num = 500 # number of volumes the 3D + T image
x_siz,y_siz,z_siz = 168, 401, 128 # size of each 3D image
z_xy_resolution_ratio = 1 # the resolution ratio between the z axis and the x-y plane 
                            # (does not need to be very accurate)
z_scaling = 1 # (integer) for interpolating images along z. z_scaling = 1 makes no interpolation.
               # z_scaling > 1 generates smoother images.
shrink = (24,24,2) # pad and shrink for u-net prediction, corresponding to (x,y,z). Large values
                   # lead to more accurate segmentations, but it should be less than (input sizes of u-net)/2.
miss_frame = [79, 135, 406] # frames that cannot be processed

# parameters manually determined by experience
noise_level = 200 # a threshold to discriminate noise/artifacts from cells
min_size = 400 # a threshold to remove small objects which may be noise/artifacts
BETA = 1000 # control coherence using a weighted average of movements of nearby points;
           # larger BETA includes more points, thus generates more coherent movements
LAMBDA = 0.00001 # control coherence by adding a loss of incoherence, large LAMBDA 
             # generates larger penalty for incoherence.
max_iteration = 10 # maximum number of iterations; large values generate more accurate tracking.
ensemble = 20 # how many predictions to make for making averaged prediction
  
# paths for saving images and results
folder_path = '/home/wen/eLife_revision/Leifer/Tracking/' 
raw_image_path = os.path.join(folder_path,"data/")
files_name = "aligned_t%04i_z%04i.tif"
auto_segmentation_vol1_path = os.path.join(folder_path,"auto_vol1/auto_R_")
unet_path = os.path.join(folder_path,"unet/")
manual_segmentation_vol1_path = os.path.join(folder_path,"manual_vol1/")
manual_name = "manual_labels%04i.tif"
track_results_path = os.path.join(folder_path,"track_results/track_results_")
track_information_path = os.path.join(folder_path,"track_information_ensembleWide_correctionIter/")

######################
# functions 
######################
def read_image(vol):
    """
    Read a raw 3D image
    Input:
        vol: a specific volume
    Return:
        an array of the image
    """
    image_raw=[]
    layer_num = z_siz
    for z in range(1,layer_num+1):
        path = raw_image_path+files_name%(vol,z)
        image_raw.append(cv2.imread(path, -1))

    return np.array(image_raw)

def segmentation(vol, method, min_size, neuron_num):
    """
    Make segmentation (unet + watershed)
    Input:
        vol: a specific volume
        method: used for watershed_3d(). "neuron_num" or "min_size"
        min_size: used for watershed_3d()
        neuron_num: used for watershed_3d()
    Return:
        image_norm: the normalized image
        image_cell_bg: the cell/background regions obtained by unet.
        min_size, neuron_num: output from watershed_3d().
    Gloabal variables:
        center_coordinates: center coordinates of segmented cells by watershed
        image_cell_bg: the cell/background regions obtained by unet.
        segmentation_auto: individual cells segmented by watershed
        image_gcn: raw image / 65536
    """
    global center_coordinates, image_cell_bg, segmentation_auto, image_gcn
    
    image_norm = read_image(vol)
    
    # pre-processing: local contrast normalization
    t = time.time()
    image_gcn = image_norm.copy()/65536.0 # this intensity is used to correct tracking results
    image_gcn = image_gcn.transpose(1,2,0)
    background_pixles = np.where(image_norm<np.median(image_norm))
    image_norm = image_norm-np.median(image_norm)
    image_norm[background_pixles]=0
    image_norm = lcn_gpu(image_norm, noise_level, filter_size=(1,27,27),
                         img3d_siz=(x_siz,y_siz,z_siz))
    image_norm = image_norm.reshape(1,z_siz,x_siz,y_siz,1)
    image_norm = image_norm.transpose(0,2,3,1,4)
    elapsed = time.time() - t
    print('pre-processing took %.1f s'%elapsed)

    # predict cell-like regions using 3D U-net
    t = time.time()
    try:
        image_cell_bg = np.load(folder_path+"unet/t%04i.npy"%(vol), allow_pickle=True)
    except OSError:
        image_cell_bg = unet3_prediction(image_norm,unet_model,shrink=shrink)
        np.save(unet_path+"unet/t%04i.npy"%(vol), np.array(image_cell_bg,dtype="float16"))
    
    # segment connected cell-like regions using watershed
    [image_watershed2d_wo_border, border]=watershed_2d(image_cell_bg[0,:,:,:,0],z_range=z_siz, min_distance=7)
    [image_watershed3d_wo_border, image_watershed3d_wi_border,
     min_size, neuron_num] = watershed_3d(image_watershed2d_wo_border,
                    samplingrate=[1,1,z_xy_resolution_ratio], method=method, 
                    min_size=min_size, neuron_num=neuron_num, min_distance=3)

    segmentation_auto, fw, inv = relabel_sequential(image_watershed3d_wi_border)
    
    # calculate coordinates of the centers of each segmented cell
    center_coordinates = snm.center_of_mass(segmentation_auto>0,segmentation_auto, range(1, segmentation_auto.max()+1))
    elapsed = time.time() - t
    print('segmentation took %.1f s'%elapsed)
    return image_norm, image_cell_bg, min_size, neuron_num


def segment_for_tracking(vol1, vol2):
    """
    Get center coordinates of cells from two specific volumes by unet + watershed
    Input: 
        vol1, vol2: the specific volumes
    Return:
        coordinates_pre_real, coordinates_post_real: coordinates of the two volumes
    """
    _,_,_,_ = segmentation(vol1, method="neuron_num", min_size=min_size, neuron_num=neuron_num)
    coordinates_pre = np.asarray(center_coordinates)
    coordinates_pre_real=coordinates_pre.copy()
    coordinates_pre_real[:,2]=coordinates_pre[:,2]*z_xy_resolution_ratio
    
    _,_,_,_ = segmentation(vol2, method="neuron_num", min_size=min_size, neuron_num=neuron_num)
    coordinates_post = np.asarray(center_coordinates)
    coordinates_post_real=coordinates_post.copy()
    coordinates_post_real[:,2]=coordinates_post[:,2]*z_xy_resolution_ratio
    
    return coordinates_pre_real, coordinates_post_real


def fnn_prgls(coordinates_pre_real, coordinates_post_real, rep, draw=True):
    """
    Appliy FFN + PR-GLS from t1 to t2 (multiple times) to get transformation 
        parameters to predict cell coordinates
    Input:
        coordinates_pre_real, coordinates_post_real: segmented cell coordinates from two volumes
        rep: the number of reptitions of (FFN + max_iteration times of PR-GLS)
        draw: if True, draw predicted coordinates from t1 and segmented coordinates of cells at t2
    Return:
        C_t: list of C in each repetition (to predict the transformed coordinates)
        BETA_t: list of the parameter beta used in each repetition (to predict coordinates)
        coor_real_t: list of the pre-transformed coordinates of automatically 
            segmented cells in each repetition (to predict coordinates)
    """
    pre_transformation = coordinates_pre_real.copy()
    C_t = []
    BETA_t = []
    coor_real_t = []
    for i in range(rep):
        coor_real_t.append(pre_transformation)
        init_match = initial_matching(FNN_model, pre_transformation, coordinates_post_real, 20)
        pre_transformation_pre = pre_transformation.copy()
        [P, pre_transformation, C] = pr_gls(pre_transformation, 
            coordinates_post_real, init_match, BETA=BETA*(0.8**i), max_iteration=max_iteration, 
            LAMBDA=LAMBDA)
        C_t.append(C)
        BETA_t.append(BETA*(0.8**i))
        if draw:
            plt.subplot(rep,2,i*2+1)
            tracking_plot(pre_transformation_pre, coordinates_post_real, pre_transformation)
            plt.subplot(rep, 2, i*2+2)
            tracking_plot_zx(pre_transformation_pre, coordinates_post_real, pre_transformation)
    return C_t, BETA_t, coor_real_t


def transform_real(coordinates_prgls, coor_pre_real_t, BETA_t, C_t, i, rep, draw=True):
    """
    Predict cell coordinates using one set of the transformation parameters
        from fnn_prgls()
    Input:
        coordinates_prgls: the coordinates before transformation
        coor_pre_real_t, BETA_t, C_t: one set of the transformation parameters
        i: the number of the repetition the set of parameters come from (used if draw==True)
        rep: the total number of repetition (used if draw==True)
        draw: whether draw the intermediate results or not
    Return:
        coordinates_prgls_2: the coordinates after transformation
    """
    length_cells = np.size(coordinates_prgls, axis=0)
    length_auto_segmentation = np.size(coor_pre_real_t, axis=0)
    Gram_matrix = np.zeros((length_auto_segmentation,length_cells))
    for idx_i in range(length_cells):
        for idx_j in range(length_auto_segmentation):
            Gram_matrix[idx_j,idx_i] = np.exp(-np.sum(np.square(
                    coordinates_prgls[idx_i,:]-coor_pre_real_t[idx_j,:]))/
                       (2*BETA_t*BETA_t)) 
    coordinates_prgls_2 = np.matrix.transpose(np.matrix.transpose(
            coordinates_prgls)+np.dot(C_t,Gram_matrix))
    
    if draw:
        plt.subplot(rep,2,i*2+1)
        tracking_plot(coordinates_prgls, coordinates_post_real, coordinates_prgls_2)
        plt.subplot(rep, 2, i*2+2)
        tracking_plot_zx(coordinates_prgls, coordinates_post_real, coordinates_prgls_2)
    
    return coordinates_prgls_2


def transformation_once(coordinates_pre_real, coordinates_post_real, coordinates_tracked_real):
    """
    Predict cell coordinates using the transformation parameters in all repetitions
        from fnn_prgls()
    Input:
        coordinates_pre_real, coordinates_post_real: coordinates of the automatically
            segmented cells at t1 and t2
        coordinates_tracked_real: the coordinates of the confirmed cells tracked at t1 (from vol=1)
    Return:
        coordinates_prgls: the predicted coordinates of the confirmed cells at t2
    """
    C_t, BETA_t, coor_real_t = fnn_prgls(coordinates_pre_real, coordinates_post_real, 5, draw=False)
    
    # apply the transformation function to calculate new coordinates of points set in previous volume (tracked coordinates)
    coordinates_prgls = coordinates_tracked_real.copy()
    
    for i in range(len(C_t)):
        coordinates_prgls = transform_real(coordinates_prgls, coor_real_t[i],
                                           BETA_t[i], C_t[i], i, len(C_t), draw=False)
    return coordinates_prgls
    

def get_reference_vols(ensemble, vol):
    """
    Get the reference volumes to calculate multiple prediction from which
    Input: 
        ensemble: the maximum number of predictions
        vol: the current volume number at which the prediction was made
    Return:
        vols_list: the list of the reference volume numbers
    """
    if vol-1 < ensemble:
        vols_list = list(range(vol-1))
    else:
        interval = (vol-1)//ensemble
        start = np.mod(vol-1, ensemble)
        vols_list = list(range(start, vol-interval, interval))
    return vols_list


def correction_once(displacement_from_vol1, displacement_in_image):
    """
    Make once correction of the predicted coordinates
    Input:
        displacement_from_vol1: real coordinates before correction
        displacement_in_image: coordinates in image (unit: voxel) before correction
    Return:
        displacement_from_vol1, displacement_in_image: coordinates after correction
        displacement_correction: the change of coordinates in this correction (used to judge the convergence).
    """
    # generate current image of labels from the manually corrected segmentation in volume 1
    tracked_cells_prgls, overlap_prgls = transform_cells(segmentation_manual_relabels,
                                                         displacement_in_image)        
    
    # overlapping regions of multiple cells are discarded before correction to avoid cells merging
    tracked_cells_prgls[np.where(overlap_prgls>1)]=0
    
    # accurate correction of displacement
    coordinates_prgls_int_move = coordinates_tracked_real_vol1*np.array([1,1,1/z_xy_resolution_ratio]) + displacement_in_image
    centers_unet_x_prgls = snm.center_of_mass(image_cell_bg[0,:,:,:,0]+image_gcn,tracked_cells_prgls, range(1, seg_cells_interpolated_corrected.max()+1))
    centers_prgls = np.asarray(coordinates_prgls_int_move)
    centers_unet_x_prgls = np.asarray(centers_unet_x_prgls)

    displacement_correction = centers_unet_x_prgls - centers_prgls
    displacement_correction[:,2]=displacement_correction[:,2]*z_xy_resolution_ratio
    
    # calculate the corrected displacement from vol #1 
    displacement_from_vol1 = displacement_in_image*np.array([1,1,z_xy_resolution_ratio]) + displacement_correction
    displacement_in_image = displacement_real_to_image(displacement_from_vol1)
    
    return displacement_from_vol1, displacement_in_image, displacement_correction
    

def displacement_real_to_image(real_disp):
    """
    Transform the coordinates from real to voxel
    Input: 
        real_disp: coordinates in real scale
    Return: 
        coordinates in voxel
    """
    displacement_in_image = real_disp.copy()
    displacement_in_image[:,2] = displacement_in_image[:,2]/z_xy_resolution_ratio
    return np.rint(displacement_in_image).astype(int)


#######################################
# load the pre-trained 3D U-net model
#######################################
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

unet_model = unet3_b()
unet_model.load_weights(os.path.join(folder_path,"models/unet3_fS2a_Leifer_0.0086.hdf5"))

############################################
# generate automatic segmetation of volume 1
############################################
# segment 3D image of volume #1      
img1_norm, cell1, min_size, neuron_num = segmentation(1, method="min_size", 
                                                      min_size=min_size, neuron_num=0)

for i in range(1, volume_num+1):
    print(i)
    segmentation(i, method="neuron_num", min_size=min_size, neuron_num=neuron_num)

# save the segmented cells of volume #1
for z in range(1,z_siz+1):
    auto_segmentation = segmentation_auto[:,:,z-1].astype(np.uint8)
    Image.fromarray(auto_segmentation).save(auto_segmentation_vol1_path+"t%04i_z%04i.tif"%(1,z))   
    
# calculate coordinates of the current "previous" volume (here is #1) 
coordinates_pre = np.asarray(center_coordinates)
coordinates_pre_real=coordinates_pre.copy()
coordinates_pre_real[:,2]=coordinates_pre[:,2]*z_xy_resolution_ratio

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
"""    
np.save("segmentation_manual_relabels.npy",segmentation_manual_relabels)
segmentation_manual_relabels = np.load("segmentation_manual_relabels.npy")
"""
    
# interpolate layers in z axis
print("interpolating...")
seg_cells_interpolated, seg_cell_or_bg = gaussian_filter(segmentation_manual_relabels,z_scaling=z_scaling, smooth_sigma=2.5)
seg_cells_interpolated_corrected = watershed_2d_markers(seg_cells_interpolated, seg_cell_or_bg, z_range=z_siz*z_scaling+10)
seg_cells_interpolated_corrected = seg_cells_interpolated_corrected[5:x_siz+5,5:y_siz+5,5:z_siz*z_scaling+5]

# save labels in the first volume (interpolated)
for z in range((z_scaling+1)//2,seg_cells_interpolated_corrected.shape[2]+1,z_scaling):
    cells_vol1 = (seg_cells_interpolated_corrected[:,:,z-1]).astype(np.uint8)
    Image.fromarray(cells_vol1).save(track_results_path+"t%04i_z%04i.tif"%(1,z/z_scaling))    
    

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
# load weights of the feedforward network 
FNN_model = load_model(os.path.join(folder_path,"models/FNN_model.h5"))

# initialize all variables
displacement_from_vol1 = coordinates_tracked_real*0
cells_on_boundary = displacement_from_vol1[:,0].astype(int)
displacements=[]
segmented_coordinates=[]; segmented_coordinates.append(coordinates_pre_real)
tracked_coordinates=[]; tracked_coordinates.append(coordinates_tracked_real)

t_1 = time.time()
for volume in range(2,volume_num+1):

    print('t=%i'%volume)
    #######################################################
    # skip frames that cannot be tracked
    #######################################################
    if volume in miss_frame:
        displacements.append(displacement_from_vol1)
        segmented_coordinates.append(coordinates_pre_real)
        tracked_coordinates.append(coordinates_tracked_real)
        for z in range(1,layer_num+1):
            tracked_cells = (label_T_watershed[:,:,z-1]).astype(np.uint8)
            Image.fromarray(tracked_cells).save(track_results_path+"t%04i_z%04i.tif"%(volume,z))
        continue

    ########################################################
    # generate automatic segmentation in current volume
    ########################################################
    _ = segmentation(volume, method="neuron_num", min_size=min_size, neuron_num=neuron_num)
    
    t = time.time()
    coordinates_post = np.asarray(center_coordinates)
    coordinates_post_real=coordinates_post.copy()
    coordinates_post_real[:,2]=coordinates_post[:,2]*z_xy_resolution_ratio
    
    #######################################
    # track by fnn + prgls
    #######################################
    # calculate the mean predictions of each cell locations
    sum_coordinates_prgls = 0
    ref_list = get_reference_vols(ensemble, volume)
    list_coor = []
    for ref in ref_list:
        print("ref:"+str(ref))
        coordinates_prgls = transformation_once(segmented_coordinates[ref], 
                                                coordinates_post_real,
                                                tracked_coordinates[ref])
        list_coor.append(coordinates_prgls)
    coordinates_prgls = trim_mean(list_coor, 0.1, axis=0)
    print("len of ref:%d"%(len(list_coor)))

    elapsed = time.time() - t
    print('fnn + pr-gls took %.1f s'%elapsed)
    
    ###################################
    # accurate correction 
    ###################################
    t = time.time()
    # calculate displacements from the first volume
    # displacement_from_vol1: accurate displacement; displacement_in_image: displacement using voxels numbers as unit
    displacement_from_vol1 = displacement_from_vol1 + coordinates_prgls-coordinates_tracked_real
    displacement_in_image = displacement_real_to_image(displacement_from_vol1)

    for i in range(20):
        displacement_from_vol1, displacement_in_image, displacement_correction = \
            correction_once(displacement_from_vol1, displacement_in_image)
        if np.max(np.abs(displacement_correction)) >= 0.5:
            print("max correction: %f"%(np.max(np.abs(displacement_correction))))
        else:
            break
        
    # generate current image of labels (more accurate)
    tracked_cells_corrected, overlap_corrected =  transform_cells(seg_cells_interpolated_corrected, displacement_in_image)
    elapsed = time.time() - t
    print('accurate correction of displacement took %.1f s'%elapsed)
    
    # re-calculate boundaries of overlapped cells using watershed
    tracked_cells_corrected[np.where(overlap_corrected>1)]=0
    label_T_watershed = watershed_2d_markers(tracked_cells_corrected[:,:,z_scaling//2:z_siz*z_scaling:z_scaling], 
                                             overlap_corrected[:,:,z_scaling//2:z_siz*z_scaling:z_scaling], z_range=z_siz)
    
    ####################################################    
    # save tracked labels and tracking information
    ####################################################
    for z in range(1,layer_num+1):
        tracked_cells = (label_T_watershed[:,:,z-1]).astype(np.uint8)
        Image.fromarray(tracked_cells).save(track_results_path+"t%04i_z%04i.tif"%(volume,z))
    
    # update and save points locations
    coordinates_pre_real = coordinates_post_real.copy()
    coordinates_tracked_real = coordinates_tracked_real_vol1 + displacement_in_image*np.array([1,1,z_xy_resolution_ratio/z_scaling])
    
    displacements.append(displacement_from_vol1)
    segmented_coordinates.append(coordinates_pre_real)
    tracked_coordinates.append(coordinates_tracked_real)

elapsed = time.time() - t_1
print('tracking all volumes took %.1f s'%elapsed)
    
# save tracking information in the folder    
np.save(track_information_path+"displacements",displacements)
np.save(track_information_path+"tracked_coordinates",tracked_coordinates)
np.save(track_information_path+"segmented_coordinates",segmented_coordinates)
