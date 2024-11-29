from __future__ import division, absolute_import, print_function, unicode_literals, annotations

import os
from typing import Tuple

import h5py
import numpy as np
from csbdeep.utils import normalize
from numpy import ndarray
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from sklearn.decomposition import PCA


def load_2d_slices_at_time(images_path: dict, t: int, channel_name: str, do_normalize: bool = True):
    """Load all 2D slices at time t and normalize the resulted 3D image"""
    if isinstance(images_path, dict):
        file_extension = os.path.splitext(images_path["h5_file"])[1]
        assert file_extension in [".h5", ".hdf5"], "Currently only HDF5 dataset is supported"

        with h5py.File(images_path["h5_file"], 'r') as f_raw:
            x = f_raw[images_path["dset"]][t - 1, :, images_path[channel_name], :, :]
    else:
        raise ValueError("image_paths should be a dict for HDF5 dataset")

    if do_normalize:
        axis_norm = (0, 1, 2)  # normalize channels independently
        return normalize(x, axis=axis_norm)
    return x


def normalize_points(points_nx3: ndarray) -> Tuple[ndarray, Tuple[ndarray, float]]:
    """
    Normalize a set of 3D points by centering them at their mean and scaling them by three times
    their standard deviation along the principal component.

    Parameters
    ----------
    points_nx3 : ndarray
        A 2D array of shape (n, 3) containing the 3D coordinates of the points.
    """
    if points_nx3.ndim != 2:
        raise ValueError(f"Points should be a 2D array, but get {points_nx3.ndim}D")
    if points_nx3.shape[1] != 3:
        raise ValueError(f"Points should have 3D coordinates, but get {points_nx3.shape[1]}D")

    # Compute the mean and PCA of the input points
    mean = np.mean(points_nx3, axis=0)
    pca = PCA(n_components=1)
    pca.fit(points_nx3)
    # Compute the standard deviation of the projection
    std = np.std(pca.transform(points_nx3)[:, 0])
    # Normalize the points
    norm_points = (points_nx3 - mean) / (3 * std)
    return norm_points, (mean, 3 * std)



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
            print(f"Recalculating... cell boundary at z = {z + 1}", end="\r")
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


def del_datasets(h5_file, datasets: list[str]):
    for dset in datasets:
        if dset in h5_file:
            del h5_file[dset]


def debug_print(msg: str, do_print=False):
    if do_print:
        print(msg)
