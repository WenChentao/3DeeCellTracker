"""
A module including functions for tracking cells
Author: Chentao Wen

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


def pr_gls_quick(X, Y, corr, BETA=300, max_iteration=20, LAMBDA=0.1, vol=1E8):
    """
    Get coherent movements from the initial matching by PR-GLS algorithm

    Parameters
    ----------
    X : numpy.ndarray
    Y : numpy.ndarray
        positions of two point sets
    corr : numpy.ndarray
        initial matching
    BETA : float
    max_iteration : int
    LAMBDA : float
    vol : float
        parameters of PR-GLS

    Returns
    -------
    P : numpy.ndarray
        updated matching
    T_X : numpy.ndarray
        transformed positions of X
    C : numpy.ndarray
        coefficients for transforming positions other than X.
    """
    ############################################################
    # initiate Gram matrix, C, sigma_square, init_match, T_X, P
    ############################################################
    # set parameters
    gamma = 0.1

    # Gram matrix quick (represents basis functions for transformation)
    length_X = np.size(X, axis=0)
    X_tile = np.tile(X, (length_X, 1, 1))
    Gram_matrix = np.exp(-np.sum(np.square(X_tile - X_tile.transpose(1, 0, 2)), axis=2) / (2 * BETA * BETA))

    # Vector C includes weights for each basis function
    C = np.zeros((3, length_X))

    # sigma square (quick): relates to the variance of differences between
    # corresponding points in T_X and Y.
    length_Y = np.size(Y, axis=0)
    X_tile = np.tile(X, (length_Y, 1, 1))
    Y_tile = np.tile(Y, (length_X, 1, 1)).transpose(1, 0, 2)
    sigma_square = np.sum(np.sum(np.square(X_tile - Y_tile), axis=2)) / (3 * length_X * length_Y)

    # set initial matching
    # only the most possible pairs are set with probalility of 0.9
    init_match = np.ones((length_Y, length_X)) / length_X
    cc_ref_tgt_temp = np.copy(corr)
    for ptr_num in range(length_X):
        cc_max = cc_ref_tgt_temp.max()
        if cc_max < 0.5:
            break
        cc_max_idx = np.unravel_index(cc_ref_tgt_temp.argmax(), cc_ref_tgt_temp.shape)
        init_match[cc_max_idx[0], :] = 0.1 / (length_X - 1)
        init_match[cc_max_idx[0], cc_max_idx[1]] = 0.9
        cc_ref_tgt_temp[cc_max_idx[0], :] = 0;
        cc_ref_tgt_temp[:, cc_max_idx[1]] = 0;

    # initiate T_X, which equals to X+v(X).
    T_X = X.copy()

    ############################################################################
    # iteratively update T_X, gamma, sigma_square, and P. Plot and save results
    ############################################################################
    for iteration in range(1, max_iteration):

        # calculate P (quick)
        T_X_tile = np.tile(T_X, (length_Y, 1, 1))
        Y_tile = np.tile(Y, (length_X, 1, 1)).transpose(1, 0, 2)
        dist_square = np.sum(np.square(T_X_tile - Y_tile), axis=2)
        exp_dist_square = np.exp(-dist_square / (2 * sigma_square))
        P1 = init_match * exp_dist_square
        denominator = np.sum(P1, axis=1) + gamma * (2 * np.pi * sigma_square) ** 1.5 / ((1 - gamma) * vol)
        denominator_tile = np.tile(denominator, (length_X, 1)).transpose()
        P = P1 / denominator_tile

        # solve the linear equations for vector C
        diag_P = np.diag(np.reshape(np.dot(np.ones((1, length_Y)), P), (length_X)))
        a = np.dot(Gram_matrix, diag_P) + LAMBDA * sigma_square * np.identity(length_X)
        b = np.dot(np.matrix.transpose(Y), P) - np.dot(np.matrix.transpose(X), diag_P)

        a = np.matrix.transpose(a)
        b = np.matrix.transpose(b)
        C = np.matrix.transpose(np.linalg.solve(a, b))

        # calculate T_X
        T_X = np.matrix.transpose(np.matrix.transpose(X) + np.dot(C, Gram_matrix))

        # update gamma and sigma square (quick)
        M_P = np.sum(P)
        gamma = 1 - M_P / length_Y

        T_X_tile = np.tile(T_X, (length_Y, 1, 1))
        dist_square = np.sum(np.square(T_X_tile - Y_tile), axis=2)
        sigma_square = np.sum(P * dist_square) / (3 * M_P)

        # avoid using too small values of sigma_square (the sample error should be >=1 pixel)
        if sigma_square < 1:
            sigma_square = 1

    return P, T_X, C


def initial_matching_quick(ffn_model, ref, tgt, k_ptrs):
    """
    This function compute initial matching between all pairs of points in reference and target points set.

    Parameters
    ----------
    ffn_model : keras.Model
        The pretrained FFN model
    ref : numpy.ndarray
        The positions of the cells in the first volume
    tgt : numpy.ndarray
        The positions of the cells in the second volume
    k_ptrs : int
        The number of neighboring points used for FFN

    Returns
    -------
    corr : numpy.ndarray
        The correspondence matrix between two point sets
    """
    nbors_ref = NearestNeighbors(n_neighbors=k_ptrs + 1).fit(ref)
    nbors_tgt = NearestNeighbors(n_neighbors=k_ptrs + 1).fit(tgt)

    ref_x_flat_batch = np.zeros((ref.shape[0], k_ptrs * 3 + 1), dtype='float32')
    tgt_x_flat_batch = np.zeros((tgt.shape[0], k_ptrs * 3 + 1), dtype='float32')

    for ref_i in range(ref.shape[0]):
        # Generate 20 (k_ptrs) points near the specific point
        # in the ref points set

        distance_ref, indices_ref = nbors_ref.kneighbors(ref[ref_i:ref_i + 1, :],
                                                         return_distance=True)

        mean_dist_ref = np.mean(distance_ref)
        ref_x = (ref[indices_ref[0, 1:k_ptrs + 1], :] - ref[indices_ref[0, 0], :]) / mean_dist_ref
        ref_x_flat = np.zeros(k_ptrs * 3 + 1)
        ref_x_flat[0:k_ptrs * 3] = ref_x.reshape(k_ptrs * 3)
        ref_x_flat[k_ptrs * 3] = mean_dist_ref

        ref_x_flat_batch[ref_i, :] = ref_x_flat.reshape(1, k_ptrs * 3 + 1)

    ref_x_flat_batch_meshgrid = np.tile(ref_x_flat_batch, (tgt.shape[0], 1, 1)).reshape(
        (ref.shape[0] * tgt.shape[0], k_ptrs * 3 + 1))

    for tgt_i in range(tgt.shape[0]):
        distance_tgt, indices_tgt = nbors_tgt.kneighbors(tgt[tgt_i:tgt_i + 1, :],
                                                         return_distance=True)
        mean_dist_tgt = np.mean(distance_tgt)
        tgt_x = (tgt[indices_tgt[0, 1:k_ptrs + 1], :] - tgt[indices_tgt[0, 0], :]) / mean_dist_tgt
        tgt_x_flat = np.zeros(k_ptrs * 3 + 1)
        tgt_x_flat[0:k_ptrs * 3] = tgt_x.reshape(k_ptrs * 3)
        tgt_x_flat[k_ptrs * 3] = mean_dist_tgt

        tgt_x_flat_batch[tgt_i, :] = tgt_x_flat.reshape(1, k_ptrs * 3 + 1)

    tgt_x_flat_batch_meshgrid = np.tile(tgt_x_flat_batch, (ref.shape[0], 1, 1)).transpose(1, 0, 2).reshape(
        (ref.shape[0] * tgt.shape[0], k_ptrs * 3 + 1))

    corr = np.reshape(ffn_model.predict([ref_x_flat_batch_meshgrid, tgt_x_flat_batch_meshgrid], batch_size=1024),
                      (tgt.shape[0], ref.shape[0]))

    return corr

def pr_gls(X,Y,corr,BETA=300, max_iteration=20, LAMBDA=0.1,vol=1E8):
    """
    (Deprecated from v0.3) The old version of pr_gls_quick(). Much slower.
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
        cc_max_idx=np.unravel_index(cc_ref_tgt_temp.argmax(),cc_ref_tgt_temp.shape)
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
    (Deprecated from v0.3) The old version of initial_matching_quick(). Much slower.
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


def gaussian_filter(img, z_scaling=10, smooth_sigma=5):
    """
    Generate smoothed label image of cells
    Parameters
    ----------
    img : numpy.ndarray
        Label image
    z_scaling : int
        Factor of interpolations along z axis, should be <10
    smooth_sigma : float
        sigma used for making Gaussian blur
    Returns
    -------
    output_img : numpy.ndarray
        Generated smoothed label image
    mask : numpy.ndarray
        Mask image indicating the overlapping of multiple cells (0: background; 1: one cell; >1: multiple cells)
    """
    img_interp = np.repeat(img, z_scaling, axis=2)
    shape_interp = img_interp.shape
    output_img = np.zeros((shape_interp[0] + 10, shape_interp[1] + 10, shape_interp[2] + 10), dtype='int')
    mask = output_img.copy()

    for label in range(1, np.max(img) + 1):
        from skimage.filters import gaussian
        print(f"Interpolating... cell:{label}", end="\r")
        x_max, x_min, y_max, y_min, z_max, z_min, subregion_pad, voxels = _get_coordinates(label, img_interp)

        percentage = 1 - np.divide(voxels, np.size(subregion_pad), dtype='float')

        img_smooth = gaussian(subregion_pad, sigma=smooth_sigma, mode='constant')

        threshold = np.percentile(img_smooth, percentage * 100)

        cell_region_interp = img_smooth > threshold

        output_img[x_min:x_max + 11, y_min:y_max + 11, z_min:z_max + 11] += cell_region_interp * label
        mask[x_min:x_max + 11, y_min:y_max + 11, z_min:z_max + 11] += cell_region_interp * 1

    return output_img, mask


def transform_cells(img3d, vectors3d):
    """
    Move individual cells in the label image.

    Parameters
    ----------
    img3d : numpy.ndarray
        Label image, each cell with different labels.
    vectors3d : numpy.ndarray
        The movement vectors for each cell, of dtype 'int' (movement from input img to output img)

    Returns
    -------
    output : numpy.ndarray
        Transformed label image
    mask : numpy.ndarray
        Overlap between different labels (if value>1)
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
    
    return output, mask


def plot_arrow(ax, x1, y1, x2, y2):
    """Draw the arrows from (x1, y1) to (x2,y2)"""
    return ax.annotate("",
                xy=(x2, y2), xycoords='axes fraction',
                xytext=(x1, y1), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="wedge",  color="C0"))

def plot_tracking_2d(T_ref, ax, draw_point, ref_ptrs, tgt_ptrs, x_axis, y_axis, sizes):
    """Draw the tracking process between two point sets with layer-based coordinates"""
    element = []
    ax.invert_yaxis()
    if draw_point:
        element.append(ax.scatter(ref_ptrs[:, x_axis], ref_ptrs[:, y_axis], facecolors='none', edgecolors='r'))
        element.append(ax.plot(tgt_ptrs[:, x_axis], tgt_ptrs[:, y_axis], 'bx')[0])
    length_X = np.size(ref_ptrs, axis=0)
    for ptr_num in range(length_X):
        element.append(plot_arrow(ax,
            x1=ref_ptrs[ptr_num, x_axis]/sizes[0], y1=1-ref_ptrs[ptr_num, y_axis]/sizes[1],
            x2=T_ref[ptr_num, x_axis]/sizes[0], y2=1-T_ref[ptr_num, y_axis]/sizes[1]))
    ax.axis('equal')
    return element

def plot_tracking_2d_realcoord(T_ref, ax, draw_point, ref_ptrs, tgt_ptrs, x_axis, y_axis):
    """Draw the tracking process between two point sets with real-resolution coordinates"""
    ax.invert_yaxis()
    element = []
    if draw_point:
        element.append(ax.scatter(ref_ptrs[:, x_axis], ref_ptrs[:, y_axis], facecolors='none', edgecolors='r'))
        element.append(ax.plot(tgt_ptrs[:, x_axis], tgt_ptrs[:, y_axis], 'bx')[0])
    length_X = np.size(ref_ptrs, axis=0)
    for ptr_num in range(length_X):
        element.append(ax.arrow(
            x=ref_ptrs[ptr_num, x_axis], y=ref_ptrs[ptr_num, y_axis],
            dx=T_ref[ptr_num, x_axis] - ref_ptrs[ptr_num, x_axis],
            dy=T_ref[ptr_num, y_axis] - ref_ptrs[ptr_num, y_axis], color="C0", length_includes_head=True,
            head_length=4, head_width=3))
    ax.axis('equal')
    return element


def tracking_plot_xy(ax, ref_ptrs, tgt_ptrs, T_ref, yx_sizes, draw_point=True, layercoord=False):
    """Draw the tracking process between two point sets in x-y plane"""
    x_axis=1
    y_axis=0
    if layercoord:
        element = plot_tracking_2d(T_ref, ax, draw_point, ref_ptrs, tgt_ptrs, x_axis, y_axis, yx_sizes)
    else:
        element = plot_tracking_2d_realcoord(T_ref, ax, draw_point, ref_ptrs, tgt_ptrs, x_axis, y_axis)
    return element


def tracking_plot_zx(ax, ref_ptrs, tgt_ptrs, T_ref, yz_sizes, draw_point=True, layercoord=True):
    """Draw the tracking process between two point sets in z-x plane"""
    x_axis=1
    y_axis=2
    if layercoord:
        element = plot_tracking_2d(T_ref, ax, draw_point, ref_ptrs, tgt_ptrs, x_axis, y_axis, yz_sizes)
    else:
        element = plot_tracking_2d_realcoord(T_ref, ax, draw_point, ref_ptrs, tgt_ptrs, x_axis, y_axis)
    return element


def FFN_matching_plot(ref_ptrs, tgt_ptrs, initial_match_score):
    """(Deprecated from v0.3) Draw the FFN_matching process"""
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
        cc_max_idx=np.unravel_index(cc_ref_tgt_temp.argmax(),cc_ref_tgt_temp.shape)
    
        plt.plot([ref_ptrs[cc_max_idx[1],1],tgt_ptrs_y_bias[cc_max_idx[0],1]],
             [-ref_ptrs[cc_max_idx[1],0],-tgt_ptrs_y_bias[cc_max_idx[0],0]],'r-')
        cc_ref_tgt_temp[cc_max_idx[0],:]=0
        cc_ref_tgt_temp[:,cc_max_idx[1]]=0  
    return fig


def get_subregions(label_image, num):
    """
    Get individual regions of segmented cells

    Parameters
    ----------
    label_image : numpy.ndarray
        Image of segmented cells
    num : int
        Number of cells

    Returns
    -------
    region_list : list
        Cropped images of each cell
    region_width : list
        Width of each cell in x,y,and z axis
    region_coord_min : list
        Minimum coordinates of each element in region list
    """
    region_list = []
    region_width = []
    region_coord_min = []
    for label in range(1, num + 1):
        if label < num:
            print(f"Calculating subregions... cell: {label}", end="\r")
        else:
            print(f"Calculating subregions... cell: {label}")
        x_max, x_min, y_max, y_min, z_max, z_min = _get_coordinates(label, label_image, get_subregion=False)
        region_list.append(label_image[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] == label)
        region_width.append([x_max + 1 - x_min, y_max + 1 - y_min, z_max + 1 - z_min])
        region_coord_min.append([x_min, y_min, z_min])
    return region_list, region_width, region_coord_min


def _get_coordinates(label, label_image, get_subregion=True):
    """
    Get the coordinates of a specific label

    Parameters
    ----------
    label : int
        The number of the cell label
    label_image :
        The label image
    get_subregion : bool
        If True, return the image of the subregion and its size

    Returns
    -------
    x_max : float
    x_min : float
    y_max : float
    y_min : float
    z_max : float
    z_min : float
        Coordinates for the subregion
    subregion : numpy.ndarray
        The subregion containing the label
    np.size(region[0]) : int
        The size of the subregion
    """
    region = np.where(label_image == label)
    x_max, x_min = np.max(region[0]), np.min(region[0])
    y_max, y_min = np.max(region[1]), np.min(region[1])
    z_max, z_min = np.max(region[2]), np.min(region[2])
    if not get_subregion:
        return x_max, x_min, y_max, y_min, z_max, z_min
    else:
        subregion = np.zeros((x_max-x_min+11,y_max-y_min+11,z_max-z_min+11))
        subregion[region[0] - x_min + 5, region[1] - y_min + 5, region[2] - z_min + 5] = 0.5
        return x_max, x_min, y_max, y_min, z_max, z_min, subregion, np.size(region[0])


def get_reference_vols(ensemble, vol, adjacent=False):
    """
    Get the reference volumes to calculate multiple prediction from which

    Parameters
    ----------
    ensemble : int
        The maximum number of predictions
    vol : int
        The current volume number at which the prediction was made
    adjacent : bool
        If True, get reference volumes from adjacent previous volumes. If False, from distributed previous volumes

    Returns
    -------
    vols_list : list
        The list of the reference volume numbers
    """
    if not ensemble:
        return [vol - 1]
    if vol - 1 < ensemble:
        vols_list = list(range(1, vol))
    else:
        if adjacent:
            vols_list = list(range(vol - ensemble, vol))
        else:
            vols_list = get_remote_vols(ensemble, vol)
    return vols_list


def get_remote_vols(ensemble, vol):
    """Get distributed previous volumes"""
    interval = (vol - 1) // ensemble
    start = np.mod(vol - 1, ensemble) + 1
    vols_list = list(range(start, vol - interval+1, interval))
    return vols_list
