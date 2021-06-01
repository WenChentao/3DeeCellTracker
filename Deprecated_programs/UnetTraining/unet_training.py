#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on May 27 2019

@author: wen
"""

#####################################
# import packages and functions
#####################################
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import random
import scipy.misc

from CellTracker.preprocess import lcn_gpu
from CellTracker.unet3d import unet3_a, unet3_prediction

%matplotlib qt

#######################################
# global parameters
#######################################
# parameters with determined values according to optic conditions
x_siz,y_siz,z_siz = 512, 1024, 21 # size of each 3D image
# parameters for training 3D U-net
x_input, y_input, z_input = 160, 160, 16 
epochs=30
steps_per_epoch=60
batch_size=8

# parameters for augmentation
datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect')

###################################
# folder path
###################################
folder_path = './Projects/3DeeCellTracker-master/Demo_UnetTraining' # path of the folder storing related files

####################################################
# Create folders for storing data and results
####################################################
train_image_path = os.path.join(folder_path,"train_image","train_image")
if not os.path.exists(os.path.dirname(train_image_path)):
    os.makedirs(os.path.dirname(train_image_path))

train_cells_path = os.path.join(folder_path,"train_cells","train_cells")
if not os.path.exists(os.path.dirname(train_cells_path)):
    os.makedirs(os.path.dirname(train_cells_path))
    
valid_image_path = os.path.join(folder_path,"valid_image","valid_image")
if not os.path.exists(os.path.dirname(valid_image_path)):
    os.makedirs(os.path.dirname(valid_image_path))
    
valid_cells_path = os.path.join(folder_path,"valid_cells","valid_cells")
if not os.path.exists(os.path.dirname(valid_cells_path)):
    os.makedirs(os.path.dirname(valid_cells_path))

weights_path = os.path.join(folder_path,"weights","weights.")
if not os.path.exists(os.path.dirname(weights_path)):
    os.makedirs(os.path.dirname(weights_path))

save_prediction_path = os.path.join(folder_path,"prediction","weights")
if not os.path.exists(os.path.dirname(save_prediction_path)):
    os.makedirs(os.path.dirname(save_prediction_path))

###############################################################################
# load training data and validation data
###############################################################################
def load_img(image_path, image_type):
    # load a single volume of 3D image
    # for raw image, apply a local contrast normalization
    # for cells annotation, transform all cell regions to 1 
    images=[]
    for z in range(0,z_siz):
        path = image_path+"%04i.tif"%(z)
        images.append(cv2.imread(path, -1))

    img_lcn=np.array(images)
    if image_type == "raw":
        background_pixles=np.where(img_lcn<np.median(img_lcn))
        img_lcn=img_lcn-np.median(img_lcn)
        img_lcn[background_pixles]=0
        img_lcn=lcn_gpu(img_lcn)
    elif image_type == "cells":
        img_lcn=img_lcn>0
        img_lcn=img_lcn.astype(int)
    else: raise NameError("image type should be raw or cells")
    img_lcn=img_lcn.reshape(1,z_siz,x_siz,y_siz,1)
    img_lcn=img_lcn.transpose(0,2,3,1,4)
    return img_lcn

train_image = load_img(train_image_path, "raw")
train_cells = load_img(train_cells_path, "cells")    
valid_image = load_img(valid_image_path, "raw")
valid_cells = load_img(valid_cells_path, "cells")   

###############################################################################
# divide raw and cell images into sub images to fit the input sizes of U-net
###############################################################################
def divide_img(img_initial_size):
    subimages=np.zeros((1,x_input, y_input, z_input,1),dtype='float32')
    for i in range(0,x_siz*2//x_input):
        for j in range(0,y_siz*2//y_input):
            for k in range(0,z_siz*2//z_input):
                idx_x = i*x_input//2 if i*x_input//2+x_input<=x_siz else x_siz-x_input
                idx_y = j*y_input//2 if j*y_input//2+y_input<=y_siz else y_siz-y_input
                idx_z = k*z_input//2 if k*z_input//2+z_input<=z_siz else z_siz-z_input
                subimages=np.concatenate((subimages,img_initial_size[:,idx_x:idx_x+x_input,
                                                                     idx_y:idx_y+y_input,
                                                                     idx_z:idx_z+z_input,:]))
    subimages=np.array(subimages[1:,],dtype='float32')
    return subimages

train_subimage = divide_img(train_image)
train_subcells = divide_img(train_cells)

valid_subimage = divide_img(valid_image)
valid_subcells = divide_img(valid_cells)

##################################################################
# increase training data by augmentation
##################################################################
def augmentation_generator(image,cells,imgGen,batch_siz):
    siz0, siz1, siz2, siz3 = np.shape(image)[0:4]
    while 1:
        seed_aug=random.randint(1,100000)
        image_gen=np.zeros((batch_siz,siz1,siz2,siz3,1),dtype='float32')
        cell_gen=np.zeros((batch_siz,siz1,siz2,siz3,1),dtype='int32')
        sample=random.randint(0,siz0//batch_siz-1)
        for z in range(0,siz3):
            gx=imgGen.flow(image[sample*batch_siz:sample*batch_siz+batch_siz,:,:,z,:], 
                           batch_size=batch_siz,seed=seed_aug)
            image_gen[:,:,:,z,:]=gx.next()
            gy=imgGen.flow(cells[sample*batch_siz:sample*batch_siz+batch_siz,:,:,z,:], 
                           batch_size=batch_siz,seed=seed_aug)
            cell_gen[:,:,:,z,:]=gy.next()
        yield (image_gen, cell_gen)
        
gen_train=augmentation_generator(train_subimage,train_subcells,datagen,batch_size)

# prepare validation data
valid_data=(valid_subimage,np.array(valid_subcells,dtype='int32'))

################################################
# train the 3D U-net
################################################
unet_model = unet3_a()
unet_model.compile(loss='binary_crossentropy', optimizer="adam")
model_callback=ModelCheckpoint(weights_path+'{epoch:02d}.hdf5',
                               save_best_only=True)
history = unet_model.fit_generator(gen_train,validation_data=valid_data, 
                                   callbacks=[model_callback], 
                                   epochs=epochs,steps_per_epoch=steps_per_epoch)

# plot loss function
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, epochs+1)
fig = plt.figure()
plt.plot(epochs_range, np.array(loss)**0.5, 'b', label='Training loss')
plt.plot(epochs_range, np.array(val_loss)**0.5, 'bo', label='Validation loss')
plt.yscale('log',basey=10) 
plt.title('Training and validationi loss')
plt.xlabel("epochs")
plt.legend()
plt.show()

# load weights and predict cell regions of the train and valid image
weight = 1
unet_model.load_weights(weights_path+'%02d.hdf5'%weight)
train_cell_bg = unet3_prediction(train_image,unet_model,shrink=(24,24,2))
max_projection = np.max(train_cell_bg[0,:,:,:,0],axis=2)
scipy.misc.toimage(max_projection, cmin=0.0, cmax=1.0).save(save_prediction_path+'%02d_train.tif'%weight)
valid_cell_bg = unet3_prediction(valid_image,unet_model,shrink=(24,24,2))
max_projection = np.max(valid_cell_bg[0,:,:,:,0],axis=2)
scipy.misc.toimage(max_projection, cmin=0.0, cmax=1.0).save(save_prediction_path+'%02d_valid.tif'%weight)
