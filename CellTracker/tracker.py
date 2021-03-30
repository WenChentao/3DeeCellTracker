"""
tracker.py
by Chentao Wen
2021.March

Resolution of variables on z axis:
* layer-based (l_, extracted from 3D image: i/scaling, or r/ratio):
* interpolated-layer-based (i_, required by accurate correction: scaling x l, or scaling/ratio * r):
* real-resolution (r_, required by fnn + prgls: ratio x l, or ratio/scaling * i):
"""

import os
import time
from functools import reduce

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.measurements as snm
from PIL import Image
from numpy import unravel_index
from scipy.stats import trim_mean
from skimage.segmentation import relabel_sequential
from sklearn.neighbors import NearestNeighbors

from CellTracker.interpolate_labels import gaussian_filter
from CellTracker.preprocess import lcn_gpu
from CellTracker.track import tracking_plot, tracking_plot_zx
from CellTracker.unet3d import unet3_prediction
from CellTracker.watershed import watershed_2d, watershed_3d, watershed_2d_markers


def make_folders(par_path, ensemble, adjacent):
    """
    make folders for storing data and results
    """
    folder_path = par_path["folder_path"]
    make_folder(os.path.join(folder_path, "data/"), "raw_image_path", par_path)
    make_folder(os.path.join(folder_path, "auto_vol1/"), "auto_segmentation_vol1_path", par_path)
    make_folder(os.path.join(folder_path, "manual_vol1/"), "manual_segmentation_vol1_path", par_path)
    make_folder(os.path.join(folder_path, "track_information/"), "track_information_path", par_path)
    make_folder(os.path.join(folder_path, "models/"), "models_path", par_path)
    make_folder(os.path.join(folder_path, "unet/"), "unet_path", par_path)
    track_results_path = get_tracking_path(adjacent, ensemble, folder_path)
    make_folder(track_results_path, "track_results_path", par_path)


def make_folder(path_i, path_name, par_path):
    """
    make a folder
    """
    if not os.path.exists(path_i):
        os.makedirs(path_i)
    par_path[path_name] = path_i


def get_tracking_path(adjacent, ensemble, folder_path):
    """
    generate path for storing tracking results according to tracking mode
    """
    if not ensemble:
        track_results_path = os.path.join(folder_path, "track_results_SingleMode/")
    elif not adjacent:
        track_results_path = os.path.join(folder_path, "track_results_EnsembleDstrbtMode/")
    else:
        track_results_path = os.path.join(folder_path, "track_results_EnsembleAdjctMode/")
    return track_results_path


def read_image(vol, path, name, z_range):
    """
    Read a raw 3D image
    Input:
        vol: a specific volume
        path: path
        name: file name
        z_range: range of layers
    Return:
        an array of the image
    """
    image_raw = []
    for z in range(z_range[0], z_range[1]):
        image_raw.append(cv2.imread(path + name % (vol, z), -1))
    return np.array(image_raw)


def read_segmentation(path, name, z_range):
    """
    Read the image of segmentation
    Input:
        vol: a specific volume
    Return:
        an array of the image
    """
    segm = []
    for z in range(z_range[0], z_range[1]):
        segm.append(cv2.imread(path + name %z, -1))
    return np.array(segm).transpose(1,2,0)

def segmentation(vol, par_image, par_tracker, par_path, unet_model, method, neuron_num):
    """
    Make segmentation (unet + watershed)
    Input:
        vol: a specific volume
        method: used for watershed_3d(). "neuron_num" or "min_size"
        neuron_num: used for watershed_3d()
    Return:
        image_cell_bg: the cell/background regions obtained by unet.
        l_center_coordinates: center coordinates of segmented cells by watershed
        segmentation_auto: individual cells segmented by watershed
        image_gcn: raw image / 65536
    """
    t = time.time()
    image_norm = read_image(vol, par_path["raw_image_path"], par_path["files_name"], [1, par_image["z_siz"] + 1])
    image_gcn = (image_norm.copy() / 65536.0).transpose(1, 2, 0)  # image_gcn will be used to correct tracking results
    image_cell_bg = save_unet_result(image_norm, par_image, par_path, par_tracker, unet_model, vol)

    # segment connected cell-like regions using watershed
    segmentation_auto = watershed(image_cell_bg, method, neuron_num, par_image, par_tracker)

    # calculate coordinates of the centers of each segmented cell
    l_center_coordinates = snm.center_of_mass(segmentation_auto > 0, segmentation_auto,
                                              range(1, segmentation_auto.max() + 1))
    elapsed = time.time() - t
    print('segmentation took %.1f s' % elapsed)
    return image_cell_bg, l_center_coordinates, segmentation_auto, image_gcn


def save_unet_result(image_norm, par_image, par_path, par_tracker, unet_model, vol):
    """
    predict cell regions and save the results
    """
    try:
        image_cell_bg = np.load(par_path["unet_path"] + "t%04i.npy" % (vol), allow_pickle=True)
    except OSError:
        # pre-processing: local contrast normalization
        image_norm = normalize_image(image_norm, par_image, par_tracker)

        # predict cell-like regions using 3D U-net
        image_cell_bg = unet3_prediction(image_norm, unet_model, shrink=par_image["shrink"])
        np.save(par_path["unet_path"] + "t%04i.npy" % (vol), np.array(image_cell_bg, dtype="float16"))
    return image_cell_bg


def watershed(image_cell_bg, method, neuron_num, par_image, par_tracker):
    """
    segment cells by watershed
    """
    [image_watershed2d_wo_border, _] = watershed_2d(image_cell_bg[0, :, :, :, 0], z_range=par_image["z_siz"],
                                                    min_distance=7)
    [_, image_watershed3d_wi_border,
     min_size, neuron_num] = watershed_3d(image_watershed2d_wo_border,
                                          samplingrate=[1, 1, par_image["z_xy_ratio"]], method=method,
                                          min_size=par_tracker["min_size"], neuron_num=neuron_num, min_distance=3)
    segmentation_auto, fw, inv = relabel_sequential(image_watershed3d_wi_border)
    par_tracker["min_size"] = min_size
    if method=="min_size":
        par_tracker["neuron_num"] = neuron_num
    return segmentation_auto


def normalize_image(image_norm, par_image, par_tracker):
    """
    normalize an image by local contrast normalization
    """
    t = time.time()
    background_pixels = np.where(image_norm < np.median(image_norm))
    image_norm = image_norm - np.median(image_norm)
    image_norm[background_pixels] = 0
    image_norm = lcn_gpu(image_norm, par_tracker["noise_level"], filter_size=(1, 27, 27),
                         img3d_siz=(par_image["x_siz"], par_image["y_siz"], par_image["z_siz"]))
    image_norm = image_norm.reshape(1, par_image["z_siz"], par_image["x_siz"], par_image["y_siz"], 1)
    image_norm = image_norm.transpose(0, 2, 3, 1, 4)
    elapsed = time.time() - t
    print('pre-processing took %.1f s' % elapsed)
    return image_norm


def save_img3(z_siz_, img, path):
    """
    save a 3D image as 2D image series
    """
    for z in range(1, z_siz_ + 1):
        img2d = img[:, :, z - 1].astype(np.uint8)
        Image.fromarray(img2d).save(path % (1, z))


def save_img3ts(z_range, img, path, t):
    """
    save a 3D image at time t as 2D image series
    """
    for i, z in enumerate(z_range):
        img2d = (img[:, :, z - 1]).astype(np.uint8)
        Image.fromarray(img2d).save(path % (t, i + 1))

def a(segmentation_manual, par_image, par_path):
    # relabeling indexes of manually corrected neurons
    segmentation_manual_relabels, fw, inv = relabel_sequential(segmentation_manual)

    # interpolate layers in z axis
    seg_cells_interpolated_corrected = interpolate(par_image, segmentation_manual_relabels)

    # save labels in the first volume (interpolated)
    save_img3ts(range((par_image["z_scaling"] + 1) // 2,
                      seg_cells_interpolated_corrected.shape[2] + 1,
                      par_image["z_scaling"]),
                seg_cells_interpolated_corrected,
                par_path["track_results_path"] + "track_results_t%04i_z%04i.tif",1)

    # calculate coordinates of centers (the corrected coordinates of cells in the first volume)
    center_points0 = snm.center_of_mass(segmentation_manual_relabels > 0, segmentation_manual_relabels,
                                        range(1, segmentation_manual_relabels.max() + 1))
    coordinates_tracked = np.asarray(center_points0)
    r_coordinates_tracked_real = coordinates_tracked.copy()
    r_coordinates_tracked_real[:, 2] = coordinates_tracked[:, 2] * par_image["z_xy_ratio"]

    # save a copy of the coordinates in volume 1
    r_coordinates_tracked_real_vol1 = r_coordinates_tracked_real.copy()
    return r_coordinates_tracked_real


def interpolate(par_image, segmentation_manual_relabels):
    print("interpolating...")
    seg_cells_interpolated, seg_cell_or_bg = gaussian_filter(segmentation_manual_relabels,
                                                             z_scaling=par_image["z_scaling"],
                                                             smooth_sigma=2.5)
    seg_cells_interpolated_corrected = watershed_2d_markers(seg_cells_interpolated, seg_cell_or_bg,
                                                            z_range=par_image["z_siz"] * par_image["z_scaling"] + 10)
    seg_cells_interpolated_corrected = seg_cells_interpolated_corrected[5:par_image["x_siz"] + 5,
                                       5:par_image["y_siz"] + 5, 5:par_image["z_siz"] * par_image["z_scaling"] + 5]
    return seg_cells_interpolated_corrected


def get_subregions(label_image, num):
    """
    Get individual regions of segmented cells
    Input:
        label_image: image of segmented cells
        num: number of cells
    Return:
        region_list: list, cropped images of each cell
        region_width: list, width of each cell in x,y,and z axis
        region_coord_min: list, minimum coordinates of each element in region list
    """
    region_list = []
    region_width = []
    region_coord_min = []
    for label in range(1, num + 1):
        print(label, end=" ")
        x_max, x_min, y_max, y_min, z_max, z_min = get_coordinates(label, label_image)
        region_list.append(label_image[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] == label)
        region_width.append([x_max + 1 - x_min, y_max + 1 - y_min, z_max + 1 - z_min])
        region_coord_min.append([x_min, y_min, z_min])
    return region_list, region_width, region_coord_min


def get_coordinates(label, label_image):
    region = np.where(label_image == label)
    x_max, x_min = np.max(region[0]), np.min(region[0])
    y_max, y_min = np.max(region[1]), np.min(region[1])
    z_max, z_min = np.max(region[2]), np.min(region[2])
    return x_max, x_min, y_max, y_min, z_max, z_min


def fit_ffn_prgls(r_coordinates_segment_pre, r_coordinates_segment_post, rep, par_tracker, FFN_model, draw=True):
    """
    Appliy FFN + PR-GLS from t1 to t2 (multiple times) to get transformation
        parameters to predict cell coordinates
    Input:
        r_coordinates_segment_pre, r_coordinates_segment_post: segmented cell coordinates from two volumes
        rep: the number of repetitions of (FFN + max_iteration times of PR-GLS)
        draw: if True, draw predicted coordinates from t1 and segmented coordinates of cells at t2
    Return:
        C_t: list of C in each repetition (to predict the transformed coordinates)
        BETA_t: list of the parameter beta used in each repetition (to predict coordinates)
        coor_real_t: list of the pre-transformed coordinates of automatically
            segmented cells in each repetition (to predict coordinates)
    """
    pre_transformation = r_coordinates_segment_pre.copy()
    C_t = []
    BETA_t = []
    coor_real_t = []
    for i in range(rep):
        coor_real_t.append(pre_transformation)
        C, pre_transformation, pre_transformation_pre = ffn_prgls_once(FFN_model, i, par_tracker, pre_transformation,
                                                                       r_coordinates_segment_post)
        C_t.append(C)
        BETA_t.append(par_tracker["BETA"] * (0.8 ** i))
        if draw:
            draw_tracking_results(i, pre_transformation_pre, r_coordinates_segment_post, pre_transformation, rep)
    return C_t, BETA_t, coor_real_t


def draw_tracking_results(i, pre_transformation_pre, r_coordinates_post_real, pre_transformation, rep):
    plt.figure(figsize=(16,16))
    plt.subplot(rep, 2, i * 2 + 1)
    tracking_plot(pre_transformation_pre, r_coordinates_post_real, pre_transformation)
    plt.subplot(rep, 2, i * 2 + 2)
    tracking_plot_zx(pre_transformation_pre, r_coordinates_post_real, pre_transformation)


def ffn_prgls_once(ffn_model, i, par_tracker, pre_transformation, r_coordinates_post_real):
    init_match = initial_matching_quick(ffn_model, pre_transformation, r_coordinates_post_real, 20)
    pre_transformation_pre = pre_transformation.copy()
    P, pre_transformation, C = pr_gls_quick(pre_transformation,
                                            r_coordinates_post_real, init_match,
                                            BETA=par_tracker["BETA"] * (0.8 ** i),
                                            max_iteration=par_tracker["max_iteration"],
                                            LAMBDA=par_tracker["LAMBDA"])
    return C, pre_transformation, pre_transformation_pre


def predict_one_rep(r_coordinates_prgls, coor_pre_real_t, BETA_t, C_t, i, rep,
                    r_coordinates_post_real, draw=True):
    """
    Predict cell coordinates using one set of the transformation parameters
        from fnn_prgls()
    Input:
        r_coordinates_prgls: the coordinates before transformation
        coor_pre_real_t, BETA_t, C_t: one set of the transformation parameters
        i: the number of the repetition the set of parameters come from (used if draw==True)
        rep: the total number of repetition (used if draw==True)
        r_coordinates_segment_post: used when drawing is True
        draw: whether draw the intermediate results or not
    Return:
        r_coordinates_prgls_2: the coordinates after transformation
    """

    length_cells = np.size(r_coordinates_prgls, axis=0)
    length_auto_segmentation = np.size(coor_pre_real_t, axis=0)

    r_coordinates_prgls_tile = np.tile(r_coordinates_prgls, (length_auto_segmentation, 1, 1))
    coor_pre_real_t_tile = np.tile(coor_pre_real_t, (length_cells, 1, 1)).transpose(1, 0, 2)
    Gram_matrix = np.exp(-np.sum(np.square(r_coordinates_prgls_tile - coor_pre_real_t_tile),
                                 axis=2) / (2 * BETA_t * BETA_t))

    r_coordinates_prgls_2 = np.matrix.transpose(np.matrix.transpose(
        r_coordinates_prgls) + np.dot(C_t, Gram_matrix))

    if draw:
        draw_tracking_results(i, r_coordinates_prgls, r_coordinates_post_real, r_coordinates_prgls_2, rep)

    return r_coordinates_prgls_2


def predict_pos_once(r_coordinates_segment_pre, r_coordinates_segment_post, r_coordinates_tracked_pre, par_tracker,
                     FFN_model, draw=False):
    """
    Predict cell coordinates using the transformation parameters in all repetitions
        from fnn_prgls()
    Input:
        r_coordinates_segment_pre, r_coordinates_segment_post: coordinates of the automatically
            segmented cells at t1 and t2
        r_coordinates_tracked_pre: the coordinates of the confirmed cells tracked at t1 (from vol=1)
    Return:
        r_coordinates_prgls: the predicted coordinates of the confirmed cells at t2
    """

    C_t, BETA_t, coor_real_t = fit_ffn_prgls(r_coordinates_segment_pre, r_coordinates_segment_post, 5,
                                             par_tracker, FFN_model, draw=False)

    # apply the transformation function to calculate new coordinates of points set in previous volume (tracked coordinates)
    r_coordinates_prgls = r_coordinates_tracked_pre.copy()

    for i in range(len(C_t)):
        r_coordinates_prgls = predict_one_rep(r_coordinates_prgls, coor_real_t[i],
                                              BETA_t[i], C_t[i], i, len(C_t), r_coordinates_segment_post, draw=draw)

    return r_coordinates_prgls


def get_reference_vols(ensemble, vol, adjacent=False):
    """
    Get the reference volumes to calculate multiple prediction from which
    Input:
        ensemble: the maximum number of predictions
        vol: the current volume number at which the prediction was made
    Return:
        vols_list: the list of the reference volume numbers
    """
    if not ensemble:
        return [vol - 2]
    if vol - 1 < ensemble:
        vols_list = list(range(vol - 1))
    else:
        if adjacent:
            vols_list = list(range(vol - ensemble - 1, vol - 1))
        else:
            vols_list = get_remote_vols(ensemble, vol)
    return vols_list


def get_remote_vols(ensemble, vol):
    interval = (vol - 1) // ensemble
    start = np.mod(vol - 1, ensemble)
    vols_list = list(range(start, vol - interval, interval))
    return vols_list


def correction_once_interp(i_displacement_from_vol1, par_image, par_subregions, cell_on_bound,
                           r_coordinates_tracked_real_vol1, image_cell_bg, image_gcn, seg_cells_interpolated_corrected):
    """
    i_displacement_from_vol1: scale in interpolated image
    i_l_tracked_cells_prgls_0, i_l_overlap_prgls_0: transformed image, scale in interpolated image
    l_tracked_cells_prgls, l_overlap_prgls: transformed image, scale in raw image
    l_coordinates_prgls_int_move, l_centers_prgls: predicted (prgls) coordinates
    l_centers_unet_x_prgls : unet prgls overlapping centers
    r_displacement_correction: par_image["z_xy_ratio"] times

    """
    # generate current image of labels from the manually corrected segmentation in volume 1
    i_l_tracked_cells_prgls_0, i_l_overlap_prgls_0 = transform_cells_quick(par_subregions, i_displacement_from_vol1,
                                                                           print_seq=False)
    l_tracked_cells_prgls = i_l_tracked_cells_prgls_0[:, :,
                            par_image["z_scaling"] // 2:par_image["z_siz"] * par_image["z_scaling"]:par_image[
                                "z_scaling"]]
    l_overlap_prgls = i_l_overlap_prgls_0[:, :,
                      par_image["z_scaling"] // 2:par_image["z_siz"] * par_image["z_scaling"]:par_image["z_scaling"]]

    # overlapping regions of multiple cells are discarded before correction to avoid cells merging
    l_tracked_cells_prgls[np.where(l_overlap_prgls > 1)] = 0

    for i in np.where(cell_on_bound == 1)[0]:
        l_tracked_cells_prgls[np.where(l_tracked_cells_prgls == (i + 1))] = 0

    # accurate correction of displacement
    l_coordinates_prgls_int_move = r_coordinates_tracked_real_vol1 * np.array(
        [1, 1, 1 / par_image["z_xy_ratio"]]) + i_displacement_from_vol1 * np.array([1, 1, 1 / par_image["z_scaling"]])
    l_centers_unet_x_prgls = snm.center_of_mass(image_cell_bg[0, :, :, :, 0] + image_gcn, l_tracked_cells_prgls,
                                                range(1, seg_cells_interpolated_corrected.max() + 1))
    l_centers_unet_x_prgls = np.asarray(l_centers_unet_x_prgls)
    l_centers_prgls = np.asarray(l_coordinates_prgls_int_move)

    lost_cells = np.where(np.isnan(l_centers_unet_x_prgls)[:, 0])

    r_displacement_correction = l_centers_unet_x_prgls - l_centers_prgls
    r_displacement_correction[lost_cells, :] = 0
    r_displacement_correction[:, 2] = r_displacement_correction[:, 2] * par_image["z_xy_ratio"]

    # calculate the corrected displacement from vol #1
    r_displacement_from_vol1 = i_displacement_from_vol1 * np.array(
        [1, 1, par_image["z_xy_ratio"] / par_image["z_scaling"]]) + r_displacement_correction
    i_displacement_from_vol1_new = displacement_real_to_interpolatedimage(r_displacement_from_vol1, par_image)

    return r_displacement_from_vol1, i_displacement_from_vol1_new, r_displacement_correction


def transform_cells_quick(par_subregions, vectors3d, print_seq=True):
    """
    Move cells according to vectors3d
    Input:
        img3d_padding: padded cell image
        pad_x, pad_y, pad_z, region_list, region_width, region_xyz_min: pad setting and cell region information
        vectors3d: sequence (int), movement of each cell
    Return:
        output: transformed image
        mask: overlap between different labels (if value>1)
    """
    label_moved = par_subregions["label_padding"].copy() * 0
    mask = label_moved.copy()
    for label in range(0, len(par_subregions["region_list"])):
        if print_seq:
            print(label, end=" ")
        new_x_min = par_subregions["region_xyz_min"][label][0] + vectors3d[label, 0] + par_subregions["pad_x"]
        new_y_min = par_subregions["region_xyz_min"][label][1] + vectors3d[label, 1] + par_subregions["pad_y"]
        new_z_min = par_subregions["region_xyz_min"][label][2] + vectors3d[label, 2] + par_subregions["pad_z"]
        subregion_previous = label_moved[new_x_min:new_x_min + par_subregions["region_width"][label][0],
                             new_y_min:new_y_min + par_subregions["region_width"][label][1],
                             new_z_min:new_z_min + par_subregions["region_width"][label][2]]
        if len(subregion_previous.flatten()) == 0:
            continue
        subregion_new = subregion_previous * (1 - par_subregions["region_list"][label]) + par_subregions["region_list"][
            label] * (label + 1)
        label_moved[new_x_min:new_x_min + par_subregions["region_width"][label][0],
        new_y_min:new_y_min + par_subregions["region_width"][label][1],
        new_z_min:new_z_min + par_subregions["region_width"][label][2]] = subregion_new
        mask[new_x_min:new_x_min + par_subregions["region_width"][label][0],
        new_y_min:new_y_min + par_subregions["region_width"][label][1],
        new_z_min:new_z_min + par_subregions["region_width"][label][2]] += (
                    par_subregions["region_list"][label] > 0).astype("int8")
    output = label_moved[par_subregions["pad_x"]:-par_subregions["pad_x"],
             par_subregions["pad_y"]:-par_subregions["pad_y"], par_subregions["pad_z"]:-par_subregions["pad_z"]]
    mask = mask[par_subregions["pad_x"]:-par_subregions["pad_x"], par_subregions["pad_y"]:-par_subregions["pad_y"],
           par_subregions["pad_z"]:-par_subregions["pad_z"]]

    return [output, mask]


def displacement_real_to_image(real_disp, i_displacement_from_vol1, par_image):
    """
    Transform the coordinates from real to voxel
    Input:
        real_disp: coordinates in real scale
    Return:
        coordinates in voxel
    """
    l_displacement_from_vol1 = real_disp.copy()
    l_displacement_from_vol1[:, 2] = i_displacement_from_vol1[:, 2] / par_image["z_xy_ratio"]
    return np.rint(l_displacement_from_vol1).astype(int)


def displacement_image_to_real(voxel_disp, par_image):
    """
    Transform the coordinates from voxel to real
    Input:
        real_disp: coordinates in real scale
    Return:
        coordinates in voxel
    """
    real_disp = np.array(voxel_disp)
    real_disp[:, 2] = real_disp[:, 2] * par_image["z_xy_ratio"]
    return real_disp


def displacement_real_to_interpolatedimage(real_disp, par_image):
    """
    Transform the coordinates from real to voxel in the interpolated image
    Input:
        real_disp: coordinates in real scale
    Return:
        coordinates in voxel
    """
    i_displacement_from_vol1 = real_disp.copy()
    i_displacement_from_vol1[:, 2] = i_displacement_from_vol1[:, 2] * par_image["z_scaling"] / par_image["z_xy_ratio"]
    return np.rint(i_displacement_from_vol1).astype(int)


def match(volume1, volume2, par_image, par_tracker, par_path, par_subregions, r_coor_segment_pre, r_coor_tracked_pre,
          r_coor_confirmed_vol1, cells_on_boundary, unet_model, FFN_model, r_disp_from_vol1_input,
          seg_cells_interp, cell_t1, seg_t1, method="min_size"):
    """
    Match current volume and another volume2
    Input:
        volume1, volume2: the two volume to be tested for tracking
        r_coor_segment_pre, r_coor_tracked_pre: the coordinates of cells from segmentation or tracking in previous volume
        r_coor_confirmed_vol1: coordinates of cells in volume #1
        r_disp_from_vol1: displacement (from vol1) of cells in previous volume
        seg_cells_interp: segmentation (interpolated)
        cell_t1, seg_t1: cell-regions/segmentation in vol1
    """
    print('t=%i' % volume2)

    #######################################################
    # skip frames that cannot be tracked
    #######################################################
    if volume2 in par_image["miss_frame"]:
        print("volume2 is a miss_frame")
        return None

    ########################################################
    # generate automatic segmentation in current volume
    ########################################################
    image_cell_bg, l_center_coordinates, _, image_gcn = \
        segmentation(volume2, par_image, par_tracker, par_path, unet_model,
                     method=method, neuron_num=par_tracker["neuron_num"])

    t = time.time()
    r_coordinates_segment_post = displacement_image_to_real(l_center_coordinates, par_image)
    #######################################
    # track by fnn + prgls
    #######################################
    # calculate the mean predictions of each cell locations
    r_coor_prgls = predict_pos_once(r_coor_segment_pre, r_coordinates_segment_post,
                                    r_coor_tracked_pre, par_tracker, FFN_model, draw=True)
    print('fnn + pr-gls took %.1f s' % (time.time() - t))
    #####################
    # boundary cells
    #####################
    cells_bd = get_cells_onBoundary(r_coor_prgls, par_image)
    print("cells on boundary:", cells_bd[0] + 1)
    cells_on_boundary_local = cells_on_boundary.copy()
    cells_on_boundary_local[cells_bd] = 1

    ###################################
    # accurate correction
    ###################################
    t = time.time()
    # calculate r_displacements from the first volume
    # r_displacement_from_vol1: accurate displacement; i_disp_from_vol1: displacement using voxels numbers as unit
    r_disp_from_vol1 = r_disp_from_vol1_input + r_coor_prgls - r_coor_tracked_pre
    i_disp_from_vol1 = displacement_real_to_interpolatedimage(r_disp_from_vol1, par_image)

    i_cumulated_disp = i_disp_from_vol1 * 0.0

    print("FFN + PR-GLS: Left: x-y; Right: x-z")
    plt.pause(10)
    print("Accurate correction:")
    rep_correction = 5
    for i in range(rep_correction):
        # update positions (from vol1) by correction
        r_disp_from_vol1, i_disp_from_vol1, r_disp_correction = \
            correction_once_interp(
            i_disp_from_vol1, par_image, par_subregions, cells_on_boundary_local,
            r_coor_confirmed_vol1, image_cell_bg, image_gcn, seg_cells_interp
        )
        # stop the repetition if correction converged
        stop_flag = evaluate_correction(r_disp_correction, i_cumulated_disp, i, par_image)

        # draw correction
        if i == rep_correction-1 or stop_flag:
            r_coordinates_correction = r_coor_confirmed_vol1 + r_disp_from_vol1
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 2, 1)
            tracking_plot(r_coor_prgls, r_coordinates_segment_post, r_coordinates_correction)
            plt.subplot(1, 2, 2)
            tracking_plot_zx(r_coor_prgls, r_coordinates_segment_post, r_coordinates_correction)

            # generate current image of labels (more accurate)
            i_tracked_cells_corrected, i_overlap_corrected = transform_cells_quick(
                par_subregions, i_disp_from_vol1, print_seq=False)

            # re-calculate boundaries by watershed
            i_tracked_cells_corrected[np.where(i_overlap_corrected > 1)] = 0
            for i in np.where(cells_on_boundary_local == 1)[0]:
                i_tracked_cells_corrected[np.where(i_tracked_cells_corrected == (i + 1))] = 0

            z_range = range(par_image["z_scaling"] // 2, par_image["z_siz"] * par_image["z_scaling"],
                            par_image["z_scaling"])
            l_label_T_watershed = watershed_2d_markers(i_tracked_cells_corrected[:, :, z_range],
                                                       i_overlap_corrected[:, :, z_range],
                                                       z_range=par_image["z_siz"])

            plt.pause(10)
            print("current volume: t=",volume1)
            plt.figure(figsize=(16, 10))
            plt.subplot(1, 2, 1)
            fig = plt.imshow(cell_t1, cmap="gray")
            plt.subplot(1, 2, 2)
            fig = plt.imshow(seg_t1, cmap="tab20b")

            plt.pause(10)
            print("target volume: t=",volume2)
            plt.figure(figsize=(16, 10))
            plt.subplot(1, 2, 1)
            fig = plt.imshow(
                np.max(image_cell_bg[0, :, :, :, 0], axis=2) > 0.5, cmap="gray")
            plt.subplot(1, 2, 2)
            fig = plt.imshow(np.max(l_label_T_watershed, axis=2), cmap="tab20b")
            break

    return None


def pr_gls_quick(X, Y, corr, BETA=300, max_iteration=20, LAMBDA=0.1, vol=1E8):
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
        cc_max_idx = unravel_index(cc_ref_tgt_temp.argmax(), cc_ref_tgt_temp.shape)
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

        # avoid using too small values of sigma_square (the sample error should be
        # >=1 pixel)
        if sigma_square < 1:
            sigma_square = 1

    return P, T_X, C


def initial_matching_quick(fnn_model, ref, tgt, k_ptrs):
    """
    this function compute initial matching between all pairs of points
    in reference and target points set
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

    corr = np.reshape(fnn_model.predict([ref_x_flat_batch_meshgrid, tgt_x_flat_batch_meshgrid], batch_size=1024),
                      (tgt.shape[0], ref.shape[0]))

    return corr


def track_one_vol(volume, par_image, par_tracker, par_path, par_subregions, unet_model, FFN_model, r_segmented_list_pre,
                  r_tracked_list_pre, r_disp_from_vol1_input, r_coor_confirmed_vol1, seg_cells_interp,
                  cells_on_boundary, method="min_size", rep_correction=20):
    """
    Track on volume
    """
    r_coor_tracked_pre = r_tracked_list_pre[-1]
    ########################################################
    # generate automatic segmentation in current volume
    ########################################################
    image_cell_bg, l_center_coordinates, _, image_gcn = \
        segmentation(volume, par_image, par_tracker, par_path, unet_model,
                     method=method, neuron_num=par_tracker["neuron_num"])
    t = time.time()
    r_coor_segment_post = displacement_image_to_real(l_center_coordinates, par_image)
    #######################################
    # track by fnn + prgls
    #######################################
    # get a list of reference volumes to predict cell positions
    ref_list = get_reference_vols(par_tracker["ensemble"], volume, adjacent=par_tracker["adjacent"])
    # predict cell positions (single or ensemble)
    list_coor = []
    print("ref:", end=" ")
    for ref in ref_list:
        print(str(ref + 1), end=", ")
        r_coordinates_prgls = predict_pos_once(r_segmented_list_pre[ref],
                                               r_coor_segment_post,
                                               r_tracked_list_pre[ref],
                                               par_tracker, FFN_model, draw=False)
        list_coor.append(r_coordinates_prgls)
    # get mean prediction
    r_coordinates_prgls = trim_mean(list_coor, 0.1, axis=0)
    print("len of ref:%d" % (len(list_coor)))
    print('fnn + pr-gls took %.1f s' % (time.time() - t))
    ###########################################################
    # remove cells moved to the boundaries of the 3D image
    ###########################################################
    cells_bd = get_cells_onBoundary(r_coordinates_prgls, par_image)
    print("cells on boundary:", cells_bd[0] + 1)
    cells_on_boundary[cells_bd] = 1
    ###################################
    # accurate correction
    ###################################
    t = time.time()
    # get positions (from vol1) before correction
    i_cumulated_disp, i_disp_from_vol1, _ = \
        get_pos_before_correction(par_image, r_coordinates_prgls,
                                  r_coor_tracked_pre, r_disp_from_vol1_input)

    for i in range(rep_correction):
        # update positions (from vol1) by correction
        r_disp_from_vol1, i_disp_from_vol1, r_disp_correction = \
            correction_once_interp(
            i_disp_from_vol1, par_image, par_subregions, cells_on_boundary,
            r_coor_confirmed_vol1, image_cell_bg, image_gcn, seg_cells_interp
        )
        # stop the repetition if correction converged
        stop_flag = evaluate_correction(r_disp_correction, i_cumulated_disp, i, par_image)
        if stop_flag:
            break

    # generate current image of labels (more accurate)
    i_tracked_cells_corrected, i_overlap_corrected = transform_cells_quick(
        par_subregions, i_disp_from_vol1, print_seq=False)
    print('accurate correction of displacement took %.1f s' % (time.time() - t))

    # re-calculate boundaries by watershed
    i_tracked_cells_corrected[np.where(i_overlap_corrected > 1)] = 0
    for i in np.where(cells_on_boundary == 1)[0]:
        i_tracked_cells_corrected[np.where(i_tracked_cells_corrected == (i + 1))] = 0

    z_range = range(par_image["z_scaling"] // 2, par_image["z_siz"] * par_image["z_scaling"], par_image["z_scaling"])
    l_label_T_watershed = watershed_2d_markers(i_tracked_cells_corrected[:, :, z_range],
                                               i_overlap_corrected[:, :, z_range],
                                               z_range=par_image["z_siz"])
    ####################################################
    # save tracked labels
    ####################################################
    save_img3ts(range(1, par_image["z_siz"] + 1), l_label_T_watershed,
                par_path["track_results_path"] + "track_results_t%04i_z%04i.tif", volume)

    # update and save points locations
    r_coor_segment_pre = r_coor_segment_post.copy()
    r_coor_tracked_pre = r_coor_confirmed_vol1 + i_disp_from_vol1 * np.array(
        [1, 1, par_image["z_xy_ratio"] / par_image["z_scaling"]])
    return l_label_T_watershed, r_coor_segment_pre, r_coor_tracked_pre, r_disp_from_vol1


def get_pos_before_correction(par_image, r_coordinates_prgls, r_coordinates_tracked_pre, r_displacement_from_vol1):
    """
    get positions of cells before accurate correction is made
    """
    r_displacement_from_vol1 += r_coordinates_prgls - r_coordinates_tracked_pre
    i_displacement_from_vol1 = displacement_real_to_interpolatedimage(r_displacement_from_vol1, par_image)
    i_cumulated_disp = i_displacement_from_vol1 * 0.0
    return i_cumulated_disp, i_displacement_from_vol1, r_displacement_from_vol1


def evaluate_correction(r_displacement_correction, i_cumulated_disp, i, par_image):
    """
    evaluate if the accurate correction should be stopped
    """
    i_cumulated_disp, i_disp_test = get_accumulated_disp(i_cumulated_disp, par_image, r_displacement_correction)
    if i == 0:
        print("max correction:", end=" ")
    if min(np.nanmax(np.abs(i_disp_test)), np.nanmax(np.abs(i_cumulated_disp))) >= 0.5:
        print(np.nanmax(np.abs(i_disp_test)), end=",")
        return False
    else:
        print(np.nanmax(np.abs(i_disp_test)))
        return True


def get_accumulated_disp(i_cumulated_disp, par_image, r_displacement_correction):
    """
    get the accumulated displacements of cells during accurate correction
    """
    i_disp_test = r_displacement_correction.copy()
    i_disp_test[:, 2] *= par_image["z_scaling"] / par_image["z_xy_ratio"]
    i_cumulated_disp += i_disp_test
    return i_cumulated_disp, i_disp_test


def get_cells_onBoundary(r_coordinates_prgls, par_image):
    """
    get cell near the boundary of the image
    """
    cells_bd = np.where(reduce(
        np.logical_or,
        [r_coordinates_prgls[:, 0] < 6,
         r_coordinates_prgls[:, 1] < 6,
         r_coordinates_prgls[:, 0] > par_image["x_siz"] - 6,
         r_coordinates_prgls[:, 1] > par_image["y_siz"] - 6,
         r_coordinates_prgls[:, 2] / par_image["z_xy_ratio"] < 0,
         r_coordinates_prgls[:, 2] / par_image["z_xy_ratio"] > par_image["z_siz"]]))
    return cells_bd
