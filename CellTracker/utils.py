from __future__ import division, absolute_import, print_function, unicode_literals, annotations

import os
from glob import glob
from typing import Union, Tuple

import numpy as np
from csbdeep.utils import normalize
from numpy import ndarray
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from sklearn.decomposition import PCA
from tifffile import imread


def load_2d_slices_at_time(images_path: str | dict, t: int, do_normalize: bool = True):
    """Load all 2D slices at time t and normalize the resulted 3D image"""
    if isinstance(images_path, str):
        file_extension = os.path.splitext(images_path)[1]
        assert file_extension in [".tif", ".tiff"], "Currently only TIFF sequences or HDF5/NWB dataset are supported"
        slice_paths_at_t = sorted(glob(images_path % t))
        if len(slice_paths_at_t) == 0:
            raise FileNotFoundError(f"No image at time {t} was found")
        x = imread(slice_paths_at_t)
    elif isinstance(images_path, dict):
        file_extension = os.path.splitext(images_path["h5_file"])[1]
        assert file_extension in [".h5", ".hdf5", ".nwb"], "Currently only TIFF sequences or HDF5/NWB dataset are supported"

        import h5py
        with h5py.File(images_path["h5_file"], 'r') as f:
            if file_extension != ".nwb":
                x = f[images_path["raw_path"]][t - 1, images_path["channel"], :, :, :]
            else:
                x = f[images_path["raw_path"]][t - 1, :, :, :, images_path["channel"]].transpose((2,0,1))
    else:
        raise ValueError("image_paths should be a str for TIFF sequences or dict for HDF5/NWB dataset")

    if do_normalize:
        axis_norm = (0, 1, 2)  # normalize channels independently
        return normalize(x, axis=axis_norm)
    return x

def normalize_points(points: ndarray, return_para: bool = False) -> Union[ndarray, Tuple[ndarray, Tuple[any, any]]]:
    """
    Normalize a set of 3D points by centering them at their mean and scaling them by three times
    their standard deviation along the principal component.

    Parameters
    ----------
    points : ndarray
        A 2D array of shape (n, 3) containing the 3D coordinates of the points.
    return_para : bool, optional
        If True, the function returns the mean and scaling factor used for normalization.
        Default is False.

    Returns
    -------
    Union[ndarray, Tuple[ndarray, Tuple[any, any]]]
        If return_para is False, returns the normalized points as a 2D array of shape (n, 3).
        If return_para is True, returns a tuple containing the normalized points and a tuple with
        the mean and scaling factor.

    Raises
    ------
    ValueError
        If the input points array is not a 2D array or if it does not contain 3D coordinates.
    """
    if points.ndim != 2:
        raise ValueError(f"Points should be a 2D table, but get {points.ndim}D")
    if points.shape[1] != 3:
        raise ValueError(f"Points should have 3D coordinates, but get {points.shape[1]}D")

    # Compute the mean and PCA of the input points
    mean = np.mean(points, axis=0)
    pca = PCA(n_components=1)
    pca.fit(points)

    # Compute the standard deviation of the projection
    std = np.std(pca.transform(points)[:, 0])

    # Normalize the points
    norm_points = (points - mean) / (3 * std)

    if return_para:
        return norm_points, (mean, 3 * std)
    else:
        return norm_points


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


def set_unique_xlim(ax1, ax2):
    x1_min, x1_max = ax1.get_xlim()
    x2_min, x2_max = ax2.get_xlim()
    ax1.set_xlim(min((x1_min, x2_min)), max((x1_max, x2_max)))
    ax2.set_xlim(min((x1_min, x2_min)), max((x1_max, x2_max)))

