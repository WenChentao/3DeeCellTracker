#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on May 29 2019

@author: wen
"""

#####################################
# import packages and functions
#####################################
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import keras
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from CellTracker.track import initial_matching, FFN_matching_plot

%matplotlib qt

#######################################
# global parameters
#######################################
# parameters for feedforward network
k_ptrs = 20 # number of neighbors for calculating relative coordinates

# parameters for generating data and for training
affine_level = 0.2
random_movement_level = 1
epochs=30
steps_per_epoch=60
batch_size=32

# paths for saving data
folder_path = '/home/ncfgpu3/3DeeCellTracker/FFNTraining/' 
data_path = os.path.join(folder_path,"data/")
weights_path = os.path.join(folder_path,"weights","weights.")
save_prediction_path = os.path.join(folder_path,"prediction","weights")
#######################################
# lood point set data
#######################################
train = np.loadtxt(data_path + "PointSet_t1.csv")
test = np.loadtxt(data_path + "PointSet_t2.csv")

###############################################
# generate data for training        
###############################################
def affine_transform(ptrs, affine_level,rand_move_level):
    ptrs_norm = ptrs-np.mean(ptrs,axis=0)
    Affine=np.array([[1,0,0],[0,1,0],[0,0,1]])
    Affine_noise=(np.random.rand(3,3)-0.5)*affine_level
    rand_noise=(np.random.rand(np.shape(ptrs)[0],
                               np.shape(ptrs)[1])-0.5)*4*rand_move_level
    ptrs_affine=np.dot(ptrs_norm,Affine+Affine_noise)+np.mean(ptrs,axis=0)+rand_noise
    return ptrs_affine

def points_match_generator(ptrs, k_ptrs, affine_level, rand_move_level, batch_size):    
    nbors=NearestNeighbors(n_neighbors=k_ptrs+1).fit(ptrs)
    while 1:
        coordinates_ref_batch = np.zeros((batch_size,k_ptrs*3+1),dtype='float32')
        coordinates_tgt_batch = np.zeros((batch_size,k_ptrs*3+1),dtype='float32')
        matching = np.zeros((batch_size,1),dtype='int')
        for batch in range(batch_size):
            # affine transform the reference point set
            ptrs_affine=affine_transform(ptrs,affine_level,rand_move_level)
            
            ##########################################################################
            # add additional errors to target set to simulate segmentation mistakes
            ##########################################################################
            random_idx = np.arange(np.shape(ptrs)[0])
            np.random.shuffle(random_idx)
            idx_errors = random_idx[0:k_ptrs+1]
            errors = (np.random.rand(k_ptrs,3)-0.5)*10
            ptrs_target = np.copy(ptrs_affine)
            ptrs_target[idx_errors[0:k_ptrs],:] = ptrs_affine[idx_errors[0:k_ptrs],:]+errors
            
            ###############################################################
            # calculate (relative) coordinates of a single reference point
            ###############################################################
            distances_ref,nbors_idx=nbors.kneighbors(ptrs[idx_errors[k_ptrs:k_ptrs+1],:],
                                     return_distance=True)
            mean_distance = np.mean(distances_ref)                                  
            coordinates_ref_relative=(ptrs[nbors_idx[0,1:k_ptrs+1],:]-ptrs[nbors_idx[0,0],:])/mean_distance
            coordinates_ref = np.zeros(k_ptrs*3+1)
            coordinates_ref[0:k_ptrs*3] = coordinates_ref_relative.reshape(k_ptrs*3)
            coordinates_ref[k_ptrs*3] = mean_distance       
            
            nbors_target=NearestNeighbors(n_neighbors=k_ptrs+1).fit(ptrs_target)            
            matching_flag=np.random.rand()
            
            if matching_flag>0.5:                  
                ###################################################################
                # calculate coordinates of corresponding point in target point set
                ###################################################################
                distances_tgt_true,nbors_idx=nbors_target.kneighbors(ptrs_target[idx_errors[k_ptrs:k_ptrs+1],:],
                                         return_distance=True)
                mean_distance=np.mean(distances_tgt_true)                                  
                coordinates_tgt_relative=(ptrs_target[nbors_idx[0,1:k_ptrs+1],:]-ptrs_target[nbors_idx[0,0],:])/mean_distance
                coordinates_tgt=np.zeros(k_ptrs*3+1)
                coordinates_tgt[0:k_ptrs*3]=coordinates_tgt_relative.reshape(k_ptrs*3)
                coordinates_tgt[k_ptrs*3]=mean_distance   
                
            elif matching_flag<=0.5:
                #########################################################################
                # calculate coordinates of a non-corresponding point in target point set
                #########################################################################
                nbors_idx=nbors_target.kneighbors(ptrs_target[idx_errors[k_ptrs:k_ptrs+1],:],
                                         return_distance=False)
                random_nbors_idx=np.copy(nbors_idx[0,1:k_ptrs+1])
                np.random.shuffle(random_nbors_idx)
                distances_tgt_false,nbors_idx_false=nbors_target.kneighbors(ptrs_target[random_nbors_idx[0:1],:],
                                         return_distance=True)                                   
                mean_distance=np.mean(distances_tgt_false)                                  
                coordinates_tgt_relative=(ptrs_target[nbors_idx_false[0,1:k_ptrs+1],:]-ptrs_target[nbors_idx_false[0,0],:])/mean_distance
                coordinates_tgt=np.zeros(k_ptrs*3+1)
                coordinates_tgt[0:k_ptrs*3]=coordinates_tgt_relative.reshape(k_ptrs*3)
                coordinates_tgt[k_ptrs*3]=mean_distance
                
            else: raise NameError("matching_flag has an abnormal value")
            
            coordinates_ref_batch[batch,:]=coordinates_ref.reshape(1,k_ptrs*3+1)
            coordinates_tgt_batch[batch,:]=coordinates_tgt.reshape(1,k_ptrs*3+1)
            matching_flag=int(matching_flag>0.5)
            matching[batch,:]=np.array(matching_flag).reshape(1,1)
            
        yield ([coordinates_ref_batch, coordinates_tgt_batch], matching)
        
gen_train = points_match_generator(train, k_ptrs, affine_level, random_movement_level, batch_size)

###############################################
#            Define the FNN model               
###############################################

ref = Input(shape=(k_ptrs*3+1, ))
tgt = Input(shape=(k_ptrs*3+1, ))
dense1=Dense(512, activation='relu')
ref1=dense1(ref)
ref1=BatchNormalization()(ref1)
tgt1=dense1(tgt)
tgt1=BatchNormalization()(tgt1)
merged_vector = keras.layers.concatenate([ref1, tgt1], axis=-1)
dense2=Dense(512, activation='relu')
merged2=dense2(merged_vector)
merged2=BatchNormalization()(merged2)
predictions = Dense(1, activation='sigmoid')(merged2)

FNN_model = Model(inputs=[ref, tgt], outputs=predictions)
FNN_model.summary()
FNN_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

###############################################
#            Train the FNN model              
###############################################
model_callback=ModelCheckpoint(weights_path+'{epoch:02d}.hdf5', monitor='loss',
                               save_best_only=True)
history = FNN_model.fit_generator(gen_train, callbacks=[model_callback], class_weight={0:0.5, 1:0.5}, 
                    epochs=epochs,steps_per_epoch=steps_per_epoch)

# plot loss function
loss = history.history['loss']
epochs_range = range(1, epochs+1)
fig = plt.figure()
plt.plot(epochs_range, np.array(loss)**0.5, 'b', label='Training loss')
plt.yscale('log',basey=10) 
plt.title('Training loss')
plt.xlabel("epochs")
plt.legend()
plt.show()

# load weights and predict initial matching in test data
weight = 1
FNN_model.load_weights(weights_path+'%02d.hdf5'%weight)
init_match = initial_matching(FNN_model, train, test, k_ptrs)
fig = FFN_matching_plot(train,test,init_match)
fig.savefig(save_prediction_path+'%02d_test.tif'%weight)
