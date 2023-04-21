"""
A module for segmenting cells with watershed in 3D images
Author: Chentao Wen

"""

import numpy as np
import skimage.morphology as morphology
from numpy import ndarray
from scipy.ndimage import filters, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, watershed


def watershed_2d(image_pred, z_range=21, min_distance=7):
    """
    Segment cells in each layer of the 3D image by 2D _watershed
    
    Parameters
    ----------
    image_pred : 
        the binary image of cell region and background (predicted by 3D U-net)
    z_range : 
        number of layers
    min_distance : 
        the minimum cell distance allowed in the result
        
    Returns
    -------
    bn_output :
        binary image (cell/bg) removing boundaries detected by _watershed
    boundary :
        image of cell boundaries
    """
    boundary = np.zeros(image_pred.shape, dtype='bool')
    for z in range(z_range):
        bn_image = image_pred[:, :, z] > 0.5
        dist = distance_transform_edt(bn_image, sampling=[1, 1])
        dist_smooth = filters.gaussian_filter(dist, 2, mode='constant')

        local_maxi = peak_local_max(dist_smooth, min_distance=min_distance, indices=False)
        markers = morphology.label(local_maxi)
        labels_ws = watershed(-dist_smooth, markers, mask=bn_image)
        labels_bd = find_boundaries(labels_ws, connectivity=2, mode='outer', background=0)

        boundary[:, :, z] = labels_bd

    bn_output = image_pred > 0.5
    bn_output[boundary == 1] = 0

    return bn_output, boundary


def watershed_3d(image_watershed2d, samplingrate, method, min_size, cell_num, min_distance):
    """
    Segment cells by 3D _watershed
    
    Parameters
    ----------
    image_watershed2d :
        the binary image (cell/bg) obtained by watershed_2d
    samplingrate : list
        resolution in x, y, and z axis to calculate 3D distance
    method :
        "min_size" or "cell_num"
    min_size :
        minimum size of cells (unit: voxels)
    cell_num :
        determine the min_distance by setting neuron number. Ignored if method=="min_size"
    min_distance :
        the minimum cell distance allowed in the result. Ignored if method=="cell_num"

    Returns
    -------
    labels_wo_bd :
        label image of cells removing boundaries (set to 0)
    labels_clear :
        label image of cells before removing boundaries
    min_size :
        min_size used in this function
    cell_num :
        neuron number detected in this function

    Notes
    -----
    For peak_local_max function, exclude_border=0 is important. Without it, the function will exclude the cells
    within bottom/top layers (<=min_distance layers)
    """
    dist = distance_transform_edt(image_watershed2d, sampling=samplingrate)
    dist_smooth = filters.gaussian_filter(dist, (2, 2, 0.3), mode='constant')
    local_maxi = peak_local_max(dist_smooth, min_distance=min_distance, exclude_border=0, indices=False)
    markers = morphology.label(local_maxi)
    labels_ws = watershed(-dist_smooth, markers, mask=image_watershed2d)
    if method == "min_size":
        cell_num = np.sum(np.sort(np.bincount(labels_ws.ravel())) >= min_size) - 1
    elif method == "cell_num":
        min_size = np.sort(np.bincount(labels_ws.ravel()))[-cell_num - 1]
    else:
        raise ("The method parameter should be either min_size or cell_num")
    labels_clear = remove_small_objects(labels_ws, min_size=min_size, connectivity=3)

    labels_bd = find_boundaries(labels_clear, connectivity=3, mode='outer', background=0)
    labels_wo_bd = labels_clear.copy()
    labels_wo_bd[labels_bd == 1] = 0
    labels_wo_bd = remove_small_objects(labels_wo_bd, min_size=min_size, connectivity=3)

    return labels_wo_bd, labels_clear, min_size, cell_num


def recalculate_cell_boundaries(segmentation_xyz: ndarray, cell_overlaps_mask: ndarray, sampling_xy: tuple = (1, 1),
                                print_message: bool = True):
    """
    Recalculate cell boundaries when cell regions are overlapping

    Parameters
    ----------
    segmentation_xyz : numpy array
        A 3D label image of cells.
    cell_overlaps_mask : numpy array
        A 3D image indicating overlapping regions (0: background; 1: one cell; >1: multiple cells).
    sampling_xy : tuple, optional
        The resolution ratio of a pixel in x-y plane.

    Returns
    -------
    numpy array
        The recalculated label image after watershed segmentation.
    """
    # Create an empty numpy array to hold the recalculated label image
    recalculated_labels = np.zeros(segmentation_xyz.shape, dtype='int')

    # Loop over each z-slice of the label image
    for z in range(segmentation_xyz.shape[2]):
        if print_message:
            print(f"Recalculating... cell boundary at z = {z+1}", end="\r")
        # Create a binary image indicating the presence of cells or overlapping regions
        mask_image = np.logical_or(segmentation_xyz[:, :, z] > 0, cell_overlaps_mask[:, :, z] > 1)

        # Set the markers for the watershed segmentation to the label image, excluding overlapping regions
        markers = segmentation_xyz[:, :, z]
        markers[cell_overlaps_mask[:, :, z] > 1] = 0

        # Calculate the Euclidean distance transform of the overlapping regions binary image
        distance_map = distance_transform_edt(cell_overlaps_mask[:, :, z] > 1, sampling=sampling_xy)

        # Perform the watershed segmentation and store the result in the output array
        recalculated_labels[:, :, z] = watershed(distance_map, markers, mask=mask_image)

    # Return the recalculated label image
    return recalculated_labels
