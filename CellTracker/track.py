#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:37:20 2017

@author: wen
"""
from numpy import unravel_index
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def pr_gls(X,Y,corr,BETA=300, max_iteration=20, LAMBDA=0.1,vol=1E8):
    """
    Get coherent movements from the initial matching by PR-GLS algorithm
    Input: 
        X,Y: positions of two point sets
        corr: initial matching
        BETA, max_iteration, LAMBDA, vol: parameters of PR-GLS
    Return:
        P: updated matching
        T_X: transformed positions of X
        C: coefficients for transforming positions other than X.
    """
    ############################################################
    # initiate Gram matrix, C, sigma_square, init_match, T_X, P
    ############################################################
    
    # set parameters
    gamma = 0.1
    
    # Gram matrix (represents basis functions for transformation)
    length_X = np.size(X, axis=0)
    Gram_matrix = np.zeros((length_X,length_X))
    for idx_i in range(length_X):
        for idx_j in range(length_X):
            Gram_matrix[idx_i,idx_j] = np.exp(-np.sum(np.square(X[idx_i,:]-X[idx_j,:]))/
                       (2*BETA*BETA)) 
    
    # Vector C includes weights for each basis function        
    C = np.zeros((3,length_X))
    
    # sigma square: relates to the variance of differences between 
    # corresponding points in T_X and Y.
    length_Y = np.size(Y,axis=0)
    sigma_square=0
    for idx_X in range(length_X):
        for idx_Y in range(length_Y):
            sigma_square = sigma_square + np.sum(np.square(X[idx_X,:]-Y[idx_Y,:]))
    sigma_square = sigma_square/(3*length_X*length_Y)
    
    # set initial matching
    # only the most possible pairs are set with probalility of 0.9
    init_match=np.ones((length_Y,length_X))/length_X
    cc_ref_tgt_temp=np.copy(corr)
    for ptr_num in range(length_X):
        cc_max=cc_ref_tgt_temp.max()
        if cc_max<0.5:
            break
        cc_max_idx=unravel_index(cc_ref_tgt_temp.argmax(),cc_ref_tgt_temp.shape)
        init_match[cc_max_idx[0],:]=0.1/(length_X-1)    
        init_match[cc_max_idx[0],cc_max_idx[1]]=0.9
        cc_ref_tgt_temp[cc_max_idx[0],:]=0;
        cc_ref_tgt_temp[:,cc_max_idx[1]]=0;
    
    # initiate T_X, which equals to X+v(X).
    T_X=X.copy()
    
    # initiate P 
    P=np.zeros((length_Y,length_X))
    
    ############################################################################
    # iteratively update T_X, gamma, sigma_square, and P. Plot and save results
    ############################################################################
    
    # loop start
    
    for iteration in range(1,max_iteration):
        
        # calculate P
        for idx_Y in range(length_Y):
            denominator=0
            for idx_X in range(length_X):
                P[idx_Y,idx_X]=init_match[idx_Y,idx_X]*np.exp(
                        -np.sum(np.square(Y[idx_Y,:]-T_X[idx_X,:]))/(2*sigma_square))
                denominator=denominator+P[idx_Y,idx_X]
            denominator=denominator+gamma*(2*np.pi*sigma_square)**1.5/((1-gamma)*vol)
            P[idx_Y,:]=P[idx_Y,:]/denominator
            
        # solve the linear equations for vector C
        diag_P=np.diag(np.reshape(np.dot(np.ones((1,length_Y)),P),(length_X)))
        a = np.dot(Gram_matrix,diag_P)+LAMBDA*sigma_square*np.identity(length_X)
        b=np.dot(np.matrix.transpose(Y),P)-np.dot(np.matrix.transpose(X),diag_P)
        
        a = np.matrix.transpose(a)
        b = np.matrix.transpose(b)
        C = np.matrix.transpose(np.linalg.solve(a, b))
        
        # calculate T_X
        T_X=np.matrix.transpose(np.matrix.transpose(X)+np.dot(C,Gram_matrix))
        
        # update gamma and sigma square
        M_P=np.sum(P)
        gamma=1-M_P/length_Y
        
        sigma_square=0
        for idx_X in range(length_X):
            for idx_Y in range(length_Y):
                sigma_square=sigma_square+P[idx_Y,idx_X]*np.sum(
                        np.square(Y[idx_Y,:]-T_X[idx_X,:]))
        sigma_square = sigma_square/(3*M_P)
        
        # avoid using too small values of sigma_square (the sample error should be
                                                        # >=1 pixel)
        if sigma_square<1:
            sigma_square=1
            
    # loop end
    
    return [P, T_X, C]

def initial_matching(fnn_model,ref,tgt,k_ptrs):
    """
    Compute initial matching
    Input:
        fnn_model: pre-trained FNN model, keras object
        ref, tgt: two point sets
        k_ptrs: number of neighbor cells to calculate relative positions
    Return:
        corr: predicted probalilities that two cells are of the same cell.
    """
    nbors_ref=NearestNeighbors(n_neighbors=k_ptrs+1).fit(ref)
    nbors_tgt=NearestNeighbors(n_neighbors=k_ptrs+1).fit(tgt)
    
    corr=np.zeros((tgt.shape[0],ref.shape[0]),dtype='float32')
    
    for ref_i in range(ref.shape[0]):       
        
        ref_x_flat_batch=np.zeros((tgt.shape[0],k_ptrs*3+1),dtype='float32')
        tgt_x_flat_batch=np.zeros((tgt.shape[0],k_ptrs*3+1),dtype='float32') 
        
        # Generate 20 (k_ptrs) points near the specific point 
        # in the ref points set
        distance_ref,indices_ref=nbors_ref.kneighbors(ref[ref_i:ref_i+1,:],
                                 return_distance=True)
        mean_dist_ref=np.mean(distance_ref)                                  
        ref_x=(ref[indices_ref[0,1:k_ptrs+1],:]-ref[indices_ref[0,0],:])/mean_dist_ref
        ref_x_flat=np.zeros(k_ptrs*3+1)
        ref_x_flat[0:k_ptrs*3]=ref_x.reshape(k_ptrs*3)
        ref_x_flat[k_ptrs*3]=mean_dist_ref   
           
        for tgt_i in range(tgt.shape[0]):     
            distance_tgt,indices_tgt=nbors_tgt.kneighbors(tgt[tgt_i:tgt_i+1,:],
                                     return_distance=True)
            mean_dist_tgt=np.mean(distance_tgt)                                  
            tgt_x=(tgt[indices_tgt[0,1:k_ptrs+1],:]-tgt[indices_tgt[0,0],:])/mean_dist_tgt
            tgt_x_flat=np.zeros(k_ptrs*3+1)
            tgt_x_flat[0:k_ptrs*3]=tgt_x.reshape(k_ptrs*3)
            tgt_x_flat[k_ptrs*3]=mean_dist_tgt
            
            ref_x_flat_batch[tgt_i,:]=ref_x_flat.reshape(1,k_ptrs*3+1)
            tgt_x_flat_batch[tgt_i,:]=tgt_x_flat.reshape(1,k_ptrs*3+1)
        
        corr[:,ref_i]=np.reshape(fnn_model.predict([ref_x_flat_batch, tgt_x_flat_batch],batch_size=32),tgt.shape[0])     
                   
    return corr


def transform_cells(img3d, vectors3d):
    """
    Move individual cells in the label image.
    Input:
        img3d: label image, each cell with different labels.
        vectors3d: the movement vectors for each cell, of dtype 'int' (movement from input img to output img)
    Return:
        output: transformed label image
        mask: overlap between different labels (if value>1)
    """
    shape = np.shape(img3d)
    output = np.zeros((shape),dtype=np.dtype(img3d[0,0,0]))
    mask = np.zeros((shape),dtype=np.dtype(img3d[0,0,0]))
    for label in range(1, img3d.max()+1):

        v1 = vectors3d[label-1,0]; v2 = vectors3d[label-1,1]; v3 = vectors3d[label-1,2]; 
        
        if v1>=0:
            idx_1_start=0;idx_1_end=shape[0]-v1
        else:
            idx_1_start=-v1;idx_1_end=shape[0]
        if v2>=0:
            idx_2_start=0;idx_2_end=shape[1]-v2
        else:
            idx_2_start=-v2;idx_2_end=shape[1]
        if v3>=0:
            idx_3_start=0;idx_3_end=shape[2]-v3
        else:
            idx_3_start=-v3;idx_3_end=shape[2]
        
        image_temp = img3d[idx_1_start:idx_1_end, idx_2_start:idx_2_end, idx_3_start:idx_3_end]
        idx_label = np.where(image_temp==label)
        output[idx_label[0]+idx_1_start+v1,idx_label[1]+idx_2_start+v2,idx_label[2]+idx_3_start+v3] = image_temp[idx_label]
        mask[idx_label[0]+idx_1_start+v1,idx_label[1]+idx_2_start+v2,
             idx_label[2]+idx_3_start+v3] = mask[idx_label[0]+idx_1_start+v1,
                      idx_label[1]+idx_2_start+v2,idx_label[2]+idx_3_start+v3]+1
    
    return [output, mask]


def tracking_plot(ref_ptrs, tgt_ptrs, T_ref):

    ptrs_T_X=T_ref.copy()
    ptrs_T_X[:,0]=-T_ref[:,0]
    
    #plt.ion()
    #plt.figure(figsize=(9,3))
    plt.scatter(ref_ptrs[:,1],-ref_ptrs[:,0],facecolors='none',edgecolors='r')
    plt.plot(tgt_ptrs[:,1],-tgt_ptrs[:,0],'x')
    plt.axis('equal')
    
    length_X = np.size(ref_ptrs,axis=0)
    for ptr_num in range(length_X):
        plt.plot([ref_ptrs[ptr_num,1],ptrs_T_X[ptr_num,1]],
                     [-ref_ptrs[ptr_num,0],ptrs_T_X[ptr_num,0]],'r-')

        
def tracking_plot_zx(ref_ptrs, tgt_ptrs, T_ref):

    ptrs_T_X=T_ref.copy()
    ptrs_T_X[:,2]=-T_ref[:,2]
    
    #plt.ion()
    #plt.figure(figsize=(9,3))
    plt.scatter(ref_ptrs[:,1],-ref_ptrs[:,2],facecolors='none',edgecolors='r')
    plt.plot(tgt_ptrs[:,1],-tgt_ptrs[:,2],'x')
    plt.axis('equal')
    
    length_X = np.size(ref_ptrs,axis=0)
    for ptr_num in range(length_X):
        plt.plot([ref_ptrs[ptr_num,1],ptrs_T_X[ptr_num,1]],
                     [-ref_ptrs[ptr_num,2],ptrs_T_X[ptr_num,2]],'r-')


def FFN_matching_plot(ref_ptrs, tgt_ptrs, initial_match_score):
    length_ref_ptrs = np.size(ref_ptrs, axis=0)
    
    tgt_ptrs_y_bias = tgt_ptrs.copy()
    bias = (np.max(tgt_ptrs[:,0])-np.min(tgt_ptrs[:,0]))*2
    tgt_ptrs_y_bias[:,0] = tgt_ptrs_y_bias[:,0]+bias
    plt.ion()
    fig = plt.figure(figsize=(9,9))
    plt.scatter(ref_ptrs[:,1],-ref_ptrs[:,0],facecolors='none',edgecolors='r')
    plt.plot(tgt_ptrs_y_bias[:,1],-tgt_ptrs_y_bias[:,0],'x')
    plt.axis('equal')

    cc_ref_tgt_temp=np.copy(initial_match_score)
    for ptr_num in range(length_ref_ptrs):
        cc_max=cc_ref_tgt_temp.max()
        if cc_max<0.5:
            break
        cc_max_idx=unravel_index(cc_ref_tgt_temp.argmax(),cc_ref_tgt_temp.shape)
    
        plt.plot([ref_ptrs[cc_max_idx[1],1],tgt_ptrs_y_bias[cc_max_idx[0],1]],
             [-ref_ptrs[cc_max_idx[1],0],-tgt_ptrs_y_bias[cc_max_idx[0],0]],'r-')
        cc_ref_tgt_temp[cc_max_idx[0],:]=0
        cc_ref_tgt_temp[:,cc_max_idx[1]]=0  
    return fig
