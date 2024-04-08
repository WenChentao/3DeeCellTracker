"""
Module Name: stardistwrapper.py
Author: Chentao Wen
Modification Description: This module is a wrapper of 3D StarDist modified according to 2_training.ipynb in GitHub StarDist repository
"""

from __future__ import print_function, unicode_literals, absolute_import, division, annotations

import os
import re
import sys
from glob import glob
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from csbdeep.utils import Path, normalize
from numpy import ndarray
import scipy.ndimage.measurements as ndm
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from stardist import Rays_GoldenSpiral, fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config3D
from stardist.utils import _normalize_grid
from tifffile import imread, imwrite
from tqdm import tqdm

from CellTracker.plot import custom_tab20_cmap
from CellTracker.utils import load_2d_slices_at_time
from CellTracker.stardist3dcustom import StarDist3DCustom

STARDIST_MODELS = "stardist_models"

UP_LIMIT = 400000
matplotlib.rcParams["image.interpolation"] = "none"

np.random.seed(42)
lbl_cmap = random_label_cmap()


def load_stardist_model(model_name: str = "stardist", basedir: str = STARDIST_MODELS):
    model = StarDist3DCustom(None, name=model_name, basedir=basedir)
    print(f"Load pretrained stardist model '{model_name}' from folder '{basedir}'")
    return model


def predict_and_save(images_path: dict, model: StarDist3DCustom, results_folder: str, t_start: int = None):
    """
    Load 2D slices of a 3D image stack obtained at time t and predict instance coordinates using a trained StarDist3DCustom model.
    Save the predicted coordinates as numpy arrays in a folder.

    Args:
        images_path (dict): The file path of the 3D image stack with 2D slices at each time point.
        model (StarDist3DCustom): A trained StarDist3DCustom model for instance segmentation of 3D image stacks.
        results_folder (str): The folder path to save the results.
        t_start (int): When provided, skip the previous time points that have been processed.
    """
    # Check if the folder exists and create it if necessary
    _results_folder = Path(results_folder)
    _results_folder.mkdir(parents=True, exist_ok=True)
    import h5py

    if isinstance(images_path, dict):
        file_extension = os.path.splitext(images_path["h5_file"])[1]
        with h5py.File(images_path["h5_file"], 'r+') as f_raw, h5py.File(str(_results_folder / "seg.h5"), 'a') as f_seg:
            raw_images = f_raw[images_path["dset"]]
            _t_start = 1 if t_start is None else t_start
            num_vol = raw_images.shape[0]
            with tqdm(total=num_vol, initial=_t_start, desc="Segmenting images", ncols=50) as pbar:
                for t in range(_t_start, num_vol + 1):
                    # Load 2D slices at time t. The raw_images should have the shape (time, channel, depth, height, width)
                    x = raw_images[t - 1, images_path["channel"], :, :, :]
                    x = normalize(x, 1, 99.8, axis=(0, 1, 2))

                    # Save predicted instance coordinates as numpy arrays
                    (labels, details), prob_map = model.predict_instances(x)
                    if t == 1:
                        f_seg.create_dataset('prob', shape=(num_vol, *prob_map.shape),
                                                         chunks=(1, *prob_map.shape), dtype="float16",
                                                         compression="lzf")
                    f_seg["prob"][t - 1, ...] = prob_map
                    f_seg.create_dataset(f'coords_{str(t - 1).zfill(6)}',
                                         data=details["points"][:, [1, 2, 0]])
                    pbar.update(1)
            print(f"All images from t={1} to t={num_vol} have been Segmented")
    else:
        raise ValueError("images_path should be of str or dict type")


def save_auto_seg(labels_xyz, results_folder: str, vol: int):
    _results_folder = Path(results_folder)
    _results_folder.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if labels_xyz.max() <= 255 else np.uint16
    imwrite(str(_results_folder / f"auto_vol_{vol}.tif"), labels_xyz.astype(dtype))


def load_training_images(path_train_images: str, path_train_labels: str, max_projection: bool):
    """Load images for training StarDist3DCustom"""
    X = sorted(glob(path_train_images))
    Y = sorted(glob(path_train_labels))
    assert len(X) > 0 and len(Y) > 0, "Error: No images found in either X or Y."
    assert all(Path(x).name == Path(y).name for x, y in zip(X, Y)), "Error: Filenames in X and Y do not match."
    X = list(map(imread, X))
    Y = list(map(imread, Y))
    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
    axis_norm = (0, 1, 2)  # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print(
            "Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x, axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]
    if len(X) == 1:
        print(
            "Warning: only one training data was provided! It will be used for both training and validation purposes!")
        X = [X[0], X[0]]
        Y = [Y[0], Y[0]]
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))
    print(f"X[0] shape:{X[0].shape}")
    i = 0
    img, lbl = X[i], Y[i]
    assert img.ndim in (3, 4)
    img = img if img.ndim == 3 else img[..., :3]
    if max_projection:
        plot_img_label_max_projection(img, lbl)
    else:
        plot_img_label_center_slice(img, lbl)

    return X, Y, X_trn, Y_trn, X_val, Y_val, n_channel


def configure(Y: List[ndarray], n_channel: int, up_limit: int = UP_LIMIT, model_name: str = 'stardist',
              basedir: str = STARDIST_MODELS):
    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    print('empirical anisotropy of labeled objects = %s' % str(anisotropy))

    # 96 is a good default choice (see 1_data.ipynb)
    n_rays = 96

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    # Set train_patch_size which should
    # 1. match anisotropy and under a predefined limitation
    a, b, c = anisotropy
    train_patch_size = np.cbrt(up_limit * a * b * c) / np.array([a, b, c])
    # 2. less than the image size
    up_limit_xyz = Y[0].shape[0], np.min(Y[0].shape[1:3]), np.min(Y[0].shape[1:3])
    scaling = np.min(np.asarray(up_limit_xyz) / train_patch_size)
    if scaling < 1:
        train_patch_size = train_patch_size * scaling
    # 3. can be divided by div_by (related to unet architecture)
    unet_n_depth = 2  #
    grid_norm = _normalize_grid(grid, 3)
    unet_pool = 2, 2, 2
    div_by = tuple(p ** unet_n_depth * g for p, g in zip(unet_pool, grid_norm))
    print(f"div_by: {div_by}")
    train_patch_size = [int(d * (i // d)) for i, d in zip(train_patch_size, div_by)]
    # 4. size of x and y should be the same (since augmentation will flip x-y axes)
    train_patch_size[1] = train_patch_size[2] = min(train_patch_size[1:])

    conf = Config3D(
        rays=rays,
        grid=grid,
        anisotropy=anisotropy,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size=train_patch_size,
        train_batch_size=2,
    )
    print_dict(vars(conf))
    assert conf.unet_n_depth == unet_n_depth
    assert conf.grid == grid_norm
    assert conf.unet_pool == unet_pool

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    model = StarDist3DCustom(conf, name=model_name, basedir=basedir)

    median_size = calculate_extents(Y, np.median)
    fov = np.array(model._axes_tile_overlap('ZYX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    return model


def print_dict(my_dict: dict):
    for key, value in my_dict.items():
        print(f"{key}: {value}")


def plot_img_label_center_slice(img, lbl, img_title="image (XY slice)", lbl_title="label (XY slice)", z=None):
    if z is None:
        z = img.shape[0] // 2
    fig, (ai, al) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw=dict(width_ratios=(1.25, 1)))
    im = ai.imshow(img[z], cmap='gray', clim=(0, 1))
    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl[z], cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()


def create_cmap(label_image_3d: ndarray, image_pars: dict , max_color=20, dist_type="2d_projection"):
    """
    Create a cmap for label_image_3d, so that the neighboring cells will have different colors

    Notes
    -----
    Both label_image_3d and image_pars["voxel_size"] should follow the order of (height, width, depth)
    """
    n = label_image_3d.max()
    assert n==len(np.unique(label_image_3d))-1
    assert max_color <= 20
    coords_nx3 = ndm.center_of_mass(
        label_image_3d > 0,
        label_image_3d,
        range(1, n + 1)
    )
    coords_nx3 = coords_nx3 * np.asarray(image_pars["voxel_size"]).reshape((1,3))

    if dist_type == "2d_projection":
        dist_nxk, indices_nxk = kneighbors_2d_projections(coords_nx3, max_color)
    else:
        dist_nxk, indices_nxk = kneighbors_3d(coords_nx3, max_color)
    dist_nxn = np.full((n, n), np.inf)
    for i, indices_i in enumerate(indices_nxk):
        dist_nxn[i, indices_i[1:]] = dist_nxk[i, 1:]

    colors = np.full((n), -1, dtype=int)
    colored_cell = []
    available_colors = np.arange(n, dtype=int)
    for i in range(n):
        ref_index, tgt_index = np.unravel_index(dist_nxn.argmin(), dist_nxn.shape)
        for ind in [ref_index, tgt_index]:
            if colors[ind] == -1:
                color_nbrs = colors[indices_nxk[ind]]
                colors[ind] = np.setdiff1d(available_colors, color_nbrs)[0]
                if len(colored_cell) >= 1:
                    dist_nxn[ind, np.asarray(colored_cell)] = np.inf
                    dist_nxn[np.asarray(colored_cell), ind] = np.inf
                colored_cell.append(ind)
        if len(colored_cell)==n:
            break

    return custom_tab20_cmap(colors.tolist())


def kneighbors_2d_projections(coords_nx3, k):
    dist_nxn_xy = cdist(coords_nx3[:, [0, 1]], coords_nx3[:, [0, 1]], metric='euclidean')
    dist_nxn_xz = cdist(coords_nx3[:, [2, 1]], coords_nx3[:, [2, 1]], metric='euclidean')
    dist_nxn = np.minimum(dist_nxn_xy, dist_nxn_xz)
    n = coords_nx3.shape[0]
    indices_nxk = np.zeros((n, k), dtype=int)
    dist_nxk = np.zeros((n, k), dtype=float)
    for i, dist_n in enumerate(dist_nxn):
        indices = np.argpartition(dist_n, k)[:k]
        sorted_indices = indices[np.argsort(dist_n[indices])]
        indices_nxk[i,:] = sorted_indices
        dist_nxk[i, :] = dist_n[sorted_indices]
    return dist_nxk, indices_nxk


def kneighbors_3d(coords_nx3, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(coords_nx3)
    dist_nxk, indices_nxk = nbrs.kneighbors(coords_nx3)
    return dist_nxk, indices_nxk


def plot_img_label_max_projection(img, lbl, cmap, img_title="image (max projection/x-y)", lbl_title="label (max projection)",
                                  fig_width_px=1200, dpi=96, gamma=5):
    fig_width_in = fig_width_px / dpi  # convert to inches assuming 96 dpi
    fig_height_in = fig_width_in / 1.618  # set height to golden ratio
    # Create a new figure with the calculated size
    fig, (ai, al) = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in))

    _img = (img - img.min()) ** (1 / gamma)
    bottom = np.percentile(_img[lbl == 0], 95)
    top = np.percentile(_img[lbl > 0], 99)
    ai.imshow(_img.max(axis=0), clim=(bottom, top))
    ai.set_title(img_title)
    #al.imshow(lbl.max(axis=0), cmap=lbl_cmap)
    al.imshow(lbl.max(axis=0), cmap=cmap)
    al.set_title(lbl_title)
    plt.tight_layout()


def plot_img_label_max_projection_xz(img, lbl, cmap, img_title="image (max projection/x-z)",
                                     lbl_title="label (max projection)",
                                     fig_width_px=1200, dpi=96, scale_z: int = 1, gamma=5):
    fig_width_in = fig_width_px / dpi  # convert to inches assuming 96 dpi
    fig_height_in = fig_width_in / 1.618  # set height to golden ratio
    # Create a new figure with the calculated size
    fig, (ai, al) = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in))

    _img = (img - img.min()) ** (1 / gamma)
    bottom = np.percentile(_img[lbl == 0], 95)
    top = np.percentile(_img[lbl > 0], 99)
    ai.imshow(_img.max(axis=1), clim=(bottom, top), aspect=scale_z)
    ai.set_title(img_title)
    al.imshow(lbl.max(axis=1), cmap=cmap, aspect=scale_z)
    al.set_title(lbl_title)
    plt.tight_layout()


def random_fliprot(img, mask, axis=None):
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)

    assert img.ndim >= mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1, 2))
    x = random_intensity_change(x)
    return x, y
