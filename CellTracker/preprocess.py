"""
A module for preprocessing of 3D cell images
Author: Chentao Wen

"""
import os

import numpy as np
import tensorflow.keras as keras
from scipy import ndimage
from tensorflow.keras.layers import Conv3D, Input
from tensorflow.keras.models import Model
from tifffile import imread


def _make_folder(path_i, print_=True):
    """
    Make a folder

    Parameters
    ----------
    path_i : str
         The folder path
    print_ : bool, optional
        If True, print the relative path of the created folder. Default: True

    Returns
    -------
    path_i : str
        The folder path
    """
    if not os.path.exists(path_i):
        os.makedirs(path_i)
    if print_:
        print(os.path.relpath(path_i, os.getcwd()))
    return path_i


def _get_files(folder_path):
    """
    Get paths of all files in the folder

    Parameters
    ----------
    folder_path : str
        The path of the folder containing images

    Returns
    -------
    img_path : list
        A list of the file paths in the folder
    """
    img_path = []
    for img_filename in sorted(os.listdir(folder_path)):
        img_path.append(folder_path + "/" + img_filename)
    return img_path


def load_image(folder_path, print_=True):
    """
    Load a 3D image from 2D layers (without time information)

    Parameters
    ----------
    folder_path : str
        The path of the folder containing images
    print_ : int, optional
        If True, print the shape of the loaded 3D image. Default: True

    Returns
    -------
    img_array : numpy.ndarray
        The 3D array of the loaded image
    """
    img_file_path = _get_files(folder_path)
    img = []
    for img_path in img_file_path:
        img.append(imread(img_path))
    img_array = np.array(img).transpose((1, 2, 0))
    if print_:
        print("Load images with shape:", img_array.shape)
    return img_array


def lcn_cpu(img3d, noise_level, filter_size=(27, 27, 1)):
    """
    Local contrast normalization by cpu
    
    Parameters
    ----------
    img3d : numpy.ndarray
        The raw 3D image
    noise_level : float
        The parameter to suppress the enhancement of the background noises
    filter_size : tuple, optional
        the window size to apply the normalization along x, y, and z axis. Default: (27, 27, 1)

    Returns
    -------
    norm : numpy.ndarray
        The normalized 3D image

    Notes
    -----
    The normalization in the edge regions used "reflect" padding, which is different with
    the lcn_gpu function (uses zero padding).
    """
    filter = np.ones(filter_size)
    filter = filter / filter.size
    avg = ndimage.convolve(img3d, filter, mode='reflect')
    diff_sqr = np.square(img3d - avg)
    std = np.sqrt(ndimage.convolve(diff_sqr, filter, mode='reflect'))
    norm = np.divide(img3d - avg, std + noise_level)
    return norm


def conv3d_keras(filter_size, img3d_siz):
    """
    Generate a keras model for applying 3D convolution

    Parameters
    ----------
    filter_size : tuple
    img3d_siz : tuple

    Returns
    -------
    tf.keras.Model
        The keras model to apply 3D convolution
    """
    inputs = Input((img3d_siz[0], img3d_siz[1], img3d_siz[2], 1))
    conv_3d = Conv3D(1, filter_size, kernel_initializer=keras.initializers.Ones(), padding='same')(inputs)
    return Model(inputs=inputs, outputs=conv_3d)


def lcn_gpu(img3d, noise_level=5, filter_size=(27, 27, 1)):
    """
    Local contrast normalization by gpu

    Parameters
    ----------
    img3d : numpy.ndarray
        The raw 3D image
    noise_level : float
        The parameter to suppress the enhancement of the background noises
    filter_size : tuple, optional
        the window size to apply the normalization along x, y, and z axis. Default: (27, 27, 1)

    Returns
    -------
    norm : numpy.ndarray
        The normalized 3D image

    Notes
    -----
    The normalization in the edge regions currently used zero padding based on keras.Conv3D function,
    which is different with the lcn_cpu function (uses "reflect" padding).
    """
    img3d_siz = img3d.shape
    volume = filter_size[0] * filter_size[1] * filter_size[2]
    conv3d_model = conv3d_keras(filter_size, img3d_siz)
    img3d = np.expand_dims(img3d, axis=(0,4))
    avg = conv3d_model.predict(img3d) / volume
    diff_sqr = np.square(img3d - avg)
    std = np.sqrt(conv3d_model.predict(diff_sqr) / volume)
    norm = np.divide(img3d - avg, std + noise_level)
    return norm[0, :, :, :, 0]


def _normalize_image(image, noise_level):
    """
    Normalize an 3D image by local contrast normalization

    Parameters
    ----------
    image : numpy.ndarray
        A 3D image to be normalized
    noise_level : float
        The parameter to suppress the enhancement of the background noises

    Returns
    -------
    numpy.ndarray
        The normalized image
    """
    image_norm = image - np.median(image)
    image_norm[image_norm < 0] = 0
    return lcn_gpu(image_norm, noise_level, filter_size=(27, 27, 1))


def _normalize_label(label_img):
    """
    Transform cell/non-cell image into binary (0/1)

    Parameters
    ----------
    label_img : numpy.ndarray
        Input image of cell/non-cell regions

    Returns
    -------
    numpy.ndarray
        The binarized image
    """
    return (label_img > 0).astype(int)