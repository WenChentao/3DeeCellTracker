"""
tracker.py
by Chentao Wen
2021.March

Resolution of variables on z axis:
* layer-based (l_, extracted from 3D image: = i_/z_scaling, or r_/z_xy_ratio):
* interpolated-layer-based (i_, required by accurate correction: = l_ * z_scaling, or r_ * z_scaling/z_xy_ratio):
* real-resolution (r_, required by fnn + prgls: = l_ * z_xy_ratio, or i_ * z_xy_ratio/z_scaling):
"""

import bz2
import os
import pickle
import time
from functools import reduce

import cv2
import matplotlib as mpl
import matplotlib.image as mgimg
import numpy as np
import scipy.ndimage.measurements as snm
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import trim_mean
from skimage.measure import label
from skimage.segmentation import relabel_sequential, find_boundaries

from .interpolate_labels import gaussian_filter
from .preprocess import _make_folder, _normalize_image, _normalize_label, load_image
from .track import pr_gls_quick, initial_matching_quick, \
    get_reference_vols, get_subregions, tracking_plot_xy, tracking_plot_zx
from .unet3d import unet3_prediction, _divide_img, _augmentation_generator
from .watershed import watershed_2d, watershed_3d, watershed_2d_markers

mpl.rcParams['image.interpolation'] = 'none'

TITLE_STYLE = {'fontsize': 16, 'verticalalignment': 'bottom'}

REP_NUM_PRGLS = 5
REP_NUM_CORRECTION = 20
BOUNDARY_XY = 6
ALPHA_BLEND = 0.5
# TODO: Write message when no cells were detected in segmentation (improper parameters)

def timer(func):
    """
    A decorators to display runtime of a function
    """

    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"{func.__name__} took {elapsed_time:0.2f} seconds")
        return value

    return wrapper_timer


def save_tracker(obj, filename, bz2_=True):
    """
    Save the tracker as a pickle file
    Args:
        obj: (obj, Tracker), the instance to be saved
        filename: (str), the location to save the obj
        bz2_:(bool), whether compress the file into a .bz2 file or not
    """
    if not bz2_:
        with open(filename, 'wb') as file_pi:
            pickle.dump(obj, file_pi)
    else:
        sfile = bz2.BZ2File(filename, 'wb')
        pickle.dump(obj, sfile)


def load_tracker(filename, bz2_=True):
    """
    Load the tracker from a pickle file
    Args:
        filename: (str), the location to save the obj
        bz2_:(bool), whether compress the file into a .bz2 file or not
    Returns:
        obj, Tracker: the saved file
    """
    if not bz2_:
        with open(filename, 'rb') as file_pi:
            return pickle.load(file_pi)
    else:
        sfile = bz2.BZ2File(filename, 'rb')
        return pickle.load(sfile)


def get_random_cmap(num, seed=1):
    """
    Generate a random cmap
    Args:
        num: (int), the number of colors to be generated
        seed: (int), use the same value to get the same cmap
    Returns:
        obj, matplotlib.colors.Colormap
    """
    vals = np.linspace(0, 1, num + 1)
    np.random.seed(seed)
    np.random.shuffle(vals)
    vals = np.concatenate(([0], vals[1:]))
    cmap = plt.cm.colors.ListedColormap(plt.cm.rainbow(vals))
    cmap.colors[0, :3] = 0
    return cmap


def get_tracking_path(adjacent, ensemble, folder_path):
    """
    Generate path for storing tracking results according to tracking mode
    Args:
        adjacent:
        ensemble:
        folder_path:

    Returns:

    """
    if not ensemble:
        track_results_path = os.path.join(folder_path, "track_results_SingleMode/")
    elif not adjacent:
        track_results_path = os.path.join(folder_path, "track_results_EnsembleDstrbtMode/")
    else:
        track_results_path = os.path.join(folder_path, "track_results_EnsembleAdjctMode/")
    return track_results_path


def read_image_ts(vol, path, name, z_range, print_=False):
    """
    Load a single volume of the 3D+T sub_images
    Input:
        vol: (int), a specific volume
        path: (str), folder path
        name: (str), file name
        z_range: (tuple), range of layers
    Return:
        obj, numpy.ndarray: an array of the image with shape (x, y, z)
    """
    image_raw = []
    for z in range(z_range[0], z_range[1]):
        image_raw.append(cv2.imread(path + name % (vol, z), -1))
    img_array = np.array(image_raw).transpose((1, 2, 0))
    if print_:
        print("Load images with shape:", img_array.shape)
    return img_array


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


class Draw:
    def __init__(self):
        self.history_r_tracked_coordinates = None
        self.r_coordinates_tracked_t0 = None
        self.segresult = None
        self.vol = None
        self.x_siz = None
        self.y_siz = None
        self.z_siz = None
        self.cell_num = None
        self.seg_cells_interpolated_corrected = None
        self.cell_num_t0 = None
        self.z_scaling = None
        self.z_xy_ratio = None
        self.Z_RANGE_INTERP = None
        self.tracked_labels = None

    def draw_segresult(self, percentile_high=99.9):
        axs, figs = self._make_3subplots()
        axs[0].set_title(f"Raw image at vol {self.vol}", fontdict=TITLE_STYLE)
        axs[1].set_title(f"Cell regions at vol {self.vol} by U-Net", fontdict=TITLE_STYLE)
        axs[2].set_title(f"Auto-_segment at vol {self.vol}", fontdict=TITLE_STYLE)
        anim_obj = []
        vmax = np.percentile(self.segresult.image_gcn, percentile_high)
        for z in range(self.z_siz):
            obj1 = axs[0].imshow(self.segresult.image_gcn[:, :, z],
                                 vmin=0, vmax=vmax, cmap="gray")
            obj2 = axs[1].imshow(self.segresult.image_cell_bg[0, :, :, z, 0] > 0.5, cmap="gray")
            obj3 = axs[2].imshow(self.segresult.segmentation_auto[:, :, z],
                                 vmin=0, vmax=self.cell_num, cmap=get_random_cmap(num=self.cell_num))
            anim_obj.append([obj1, obj2, obj3])
        anim = animation.ArtistAnimation(figs, anim_obj, interval=200).to_jshtml()

        axs[0].imshow(np.max(self.segresult.image_gcn, axis=2), vmin=0, vmax=vmax, cmap="gray")
        axs[1].imshow(np.max(self.segresult.image_cell_bg[0, :, :, :, 0] > 0.5, axis=2), cmap="gray")
        axs[2].imshow(np.max(self.segresult.segmentation_auto, axis=2),
                      cmap=get_random_cmap(num=self.cell_num))
        print("Segmentation results (max projection):")
        return anim

    def draw_manual_seg1(self):
        axm, figm = self._make_horizontal_2subplots()
        axm[0].imshow(np.max(self.segresult.image_cell_bg[0, :, :, :, 0], axis=2) > 0.5, cmap="gray")
        axm[0].set_title(f"Cell regions at vol {self.vol} by U-Net", fontdict=TITLE_STYLE)
        axm[1].imshow(np.max(self.seg_cells_interpolated_corrected, axis=2),
                      cmap=get_random_cmap(num=self.cell_num_t0))
        axm[1].set_title(f"Manual _segment at vol 1", fontdict=TITLE_STYLE)

    def _get_tracking_anim(self, ax, r_coordinates_predicted_pre, r_coordinates_segmented_post,
                           r_coordinates_predicted_post, layercoord, draw_point=True):
        element1 = tracking_plot_xy(
            ax[0], r_coordinates_predicted_pre, r_coordinates_segmented_post, r_coordinates_predicted_post,
            (self.y_siz, self.x_siz), draw_point, layercoord)
        element2 = tracking_plot_zx(
            ax[1], r_coordinates_predicted_pre, r_coordinates_segmented_post, r_coordinates_predicted_post,
            (self.y_siz, self.z_siz), draw_point, layercoord)
        if layercoord:
            ax[0].set_aspect('equal', 'box')
            ax[1].set_aspect('equal', 'box')
        return element1 + element2

    def draw_correction(self, i_disp_from_vol1_updated, r_coor_predicted):
        ax, fig = self._make_horizontal_2subplots()
        ax[0].set_title("Accurate Correction (y-x plane)", size=16)
        ax[1].set_title("Accurate Correction (y-z plane)", size=16)
        self._draw_correction(ax, r_coor_predicted, i_disp_from_vol1_updated)
        return None

    def _draw_correction(self, ax, r_coor_predicted, i_disp_from_vol1_updated):
        _ = self._get_tracking_anim(
            [ax[0], ax[1]],
            self._transform_real_to_layer(r_coor_predicted),
            self._transform_real_to_layer(self.segresult.r_coordinates_segment),
            self._transform_real_to_layer(self.r_coordinates_tracked_t0) +
            self._transform_interpolated_to_layer(i_disp_from_vol1_updated),
            layercoord=True, draw_point=False)
        ax[0].imshow(np.max(self.segresult.image_cell_bg[0, :, :, :, 0], axis=2) > 0.5, cmap="gray",
                     extent=(0, self.y_siz - 1, self.x_siz - 1, 0))
        ax[1].imshow(np.max(self.segresult.image_cell_bg[0, :, :, :, 0], axis=0).T > 0.5, aspect=self.z_xy_ratio,
                     cmap="gray", extent=(0, self.y_siz - 1, self.z_siz - 1, 0))

        return None

    def draw_overlapping(self, cells_on_boundary_local, volume2, i_disp_from_vol1_updated):
        # generate current image of labels (more accurate)
        self.tracked_labels = self._transform_motion_to_image(cells_on_boundary_local, i_disp_from_vol1_updated)
        self._draw_matching(volume2)
        plt.pause(0.1)
        return None

    def _draw_matching(self, volume2):
        axc, figc = self._make_4subplots()
        self._draw_before_matching(axc[0], axc[1], volume2)
        self._draw_after_matching(axc[2], axc[3], volume2)
        plt.tight_layout()
        return None

    def _draw_matching_6panel(self, target_volume, ax, r_coor_predicted_mean, i_disp_from_vol1_updated):
        for ax_i in ax:
            ax_i.cla()
        plt.suptitle(f"Tracking results at vol {target_volume}", size=16)

        _ = self._get_tracking_anim([ax[0], ax[1]], self.history_r_tracked_coordinates[target_volume - 2],
                                    self.segresult.r_coordinates_segment, r_coor_predicted_mean, layercoord=False)
        self._draw_correction([ax[2], ax[3]], r_coor_predicted_mean, i_disp_from_vol1_updated)
        self._draw_after_matching(ax[4], ax[5], target_volume, legend=False)
        self._set_layout_anim()
        for axi in ax:
            plt.setp(axi.get_xticklabels(), visible=False)
            plt.setp(axi.get_yticklabels(), visible=False)
            axi.tick_params(axis='both', which='both', length=0)
            axi.axis("off")
        return None

    def _draw_before_matching(self, ax1, ax2, volume2):
        ax1.imshow(np.max(self.segresult.image_cell_bg[0, :, :, :, 0], axis=2) > 0.5, cmap="gray")
        ax1.imshow(np.max(self.seg_cells_interpolated_corrected[:, :, self.Z_RANGE_INTERP], axis=2),
                   cmap=get_random_cmap(num=self.cell_num_t0), alpha=ALPHA_BLEND)

        ax2.imshow(np.max(self.segresult.image_cell_bg[0, :, :, :, 0], axis=0).T > 0.5, aspect=self.z_xy_ratio,
                   cmap="gray")
        ax2.imshow(np.max(self.seg_cells_interpolated_corrected[:, :, self.Z_RANGE_INTERP], axis=0).T,
                   cmap=get_random_cmap(num=self.cell_num_t0), aspect=self.z_xy_ratio, alpha=ALPHA_BLEND)
        ax1.set_title(f"Before matching: Cells at vol {volume2} + Labels at vol {self.vol} (y-x plane)",
                      fontdict=TITLE_STYLE)
        ax2.set_title(f"Before matching (y-z plane)",
                      fontdict=TITLE_STYLE)

    def _draw_after_matching(self, ax1, ax2, volume2, legend=True):
        ax1.imshow(np.max(self.segresult.image_cell_bg[0, :, :, :, 0], axis=2) > 0.5, cmap="gray")
        ax1.imshow(np.max(self.tracked_labels, axis=2),
                   cmap=get_random_cmap(num=self.cell_num_t0), alpha=ALPHA_BLEND)

        ax2.imshow(np.max(self.segresult.image_cell_bg[0, :, :, :, 0], axis=0).T > 0.5, aspect=self.z_xy_ratio,
                   cmap="gray")
        ax2.imshow(np.max(self.tracked_labels, axis=0).T, cmap=get_random_cmap(num=self.cell_num_t0),
                   aspect=self.z_xy_ratio, alpha=ALPHA_BLEND)
        if legend:
            ax1.set_title(f"After matching: Cells at vol {volume2} + Labels at vol {volume2} (y-x plane)",
                          fontdict=TITLE_STYLE)
            ax2.set_title(f"After matching (y-z plane)",
                          fontdict=TITLE_STYLE)
        return None

    def _prepare_tracking_animation(self):
        ax, fig = self._make_horizontal_2subplots()
        ax[0].set_title("Matching by FFN + PR-GLS (y-x plane)", fontdict=TITLE_STYLE)
        ax[1].set_title("Matching by FFN + PR-GLS (y-z plane)", fontdict=TITLE_STYLE)
        plt.tight_layout()
        plt.close(fig)
        return ax, fig

    def _make_horizontal_2subplots(self):
        fig, ax = plt.subplots(1, 2, figsize=(20, int(12 * self.x_siz / self.y_siz)))
        plt.tight_layout()
        return ax, fig

    def _make_vertical_2subplots(self):
        height_z = self.z_siz * self.z_xy_ratio
        fig1 = plt.figure(figsize=(10, int(12 * (self.x_siz / self.y_siz + height_z / self.y_siz))))
        gs = GridSpec(10, 1, figure=fig1)
        h1 = int(10 * (self.x_siz / (self.x_siz + height_z)))
        ax = fig1.add_subplot(gs[:h1]), fig1.add_subplot(gs[h1:, 0])
        plt.tight_layout()
        return ax, fig1

    def _make_3subplots(self):
        fig = plt.figure(figsize=(20, int(24 * self.x_siz / self.y_siz)))
        ax = plt.subplot(221), plt.subplot(222), plt.subplot(223)
        plt.tight_layout()
        return ax, fig

    def _make_4subplots(self):
        fig, axs = plt.subplots(2, 2, figsize=(20, int(24 * self.x_siz / self.y_siz)))
        ax = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
        plt.tight_layout()
        return ax, fig

    def _make_6subplots(self):
        fig, axs = plt.subplots(3, 2, figsize=(14, int(21 * self.x_siz / self.y_siz)))
        ax = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]
        return ax, fig

    @staticmethod
    def _set_layout_anim():
        plt.tight_layout()
        plt.subplots_adjust(right=0.9, bottom=0.1)

    def _transform_interpolated_to_layer(self, i_disp_from_vol1_updated):
        raise NotImplementedError("Must override this method")

    def _transform_motion_to_image(self, cells_on_boundary_local, i_disp_from_vol1_updated):
        raise NotImplementedError("Must override this method")

    def _transform_real_to_layer(self, r_coor_predicted):
        raise NotImplementedError("Must override this method")


class Segmentation:
    class SegResults:
        def __init__(self):
            self.image_cell_bg = None
            self.l_center_coordinates = None
            self.segmentation_auto = None
            self.image_gcn = None
            self.r_coordinates_segment = None

        def _update_results(self, image_cell_bg, l_center_coordinates, segmentation_auto,
                            image_gcn, r_coordinates_segment):
            self.image_cell_bg = image_cell_bg
            self.l_center_coordinates = l_center_coordinates
            self.segmentation_auto = segmentation_auto
            self.image_gcn = image_gcn
            self.r_coordinates_segment = r_coordinates_segment

    def __init__(self, volume_num, siz_xyz: tuple, z_xy_ratio, z_scaling,
                 image_name, unet_model_file, shrink):
        self.volume_num = volume_num  # number of volumes the 3D + T image
        self.x_siz = siz_xyz[0]  # size of each 3D image
        self.y_siz = siz_xyz[1]  # size of each 3D image
        self.z_siz = siz_xyz[2]  # size of each 3D image
        self.z_xy_ratio = z_xy_ratio  # the (rough) resolution ratio between the z axis and the x-y plane
        self.z_scaling = z_scaling  # (integer; >=1) for interpolating images along z. z_scaling = 1 makes no interpolation.
        self.shrink = shrink  # <(input sizes of u-net)/2, pad and shrink for u-net prediction
        self.image_name = image_name  # file names for the raw image files
        # file names for the manaul _segment files at t=1
        # weight file of the trained 3D U-Net. f2b:structure a; fS2a:b; fS2b:c
        self.unet_model_file = unet_model_file
        self.unet_path = None
        self.models_path = None
        self.auto_segmentation_vol1_path = None
        self.raw_image_path = None
        self.recalculate_unet = False
        self.segresult = self.SegResults()

    def set_segmentation(self, noise_level=None, min_size=None, reset_=False):
        """Reset the parameters and/or delete the caches of predictions by unet"""
        if self.noise_level == noise_level and self.min_size == min_size:
            print("Segmentation parameters were not modified")
        elif noise_level==None or min_size==None:
            print("Segmentation parameters were not modified")
        else:
            self.noise_level = noise_level
            self.min_size = min_size
            print(f"Parameters were modified: noise_level={self.noise_level}, min_size={self.min_size}")
            for f in os.listdir(self.unet_path):
                os.remove(os.path.join(self.unet_path, f))
            print(f"All files under /unet folder were deleted")
        if reset_:
            for f in os.listdir(self.unet_path):
                os.remove(os.path.join(self.unet_path, f))
            print(f"All files under /unet folder were deleted")

    @staticmethod
    def _transform_disps(disp, factor):
        new_disp = np.array(disp).copy()
        new_disp[:, 2] = new_disp[:, 2] * factor
        return new_disp

    def _transform_layer_to_real(self, voxel_disp):
        """Transform the coordinates from voxel to real"""
        return self._transform_disps(voxel_disp, self.z_xy_ratio)

    def _transform_real_to_interpolated(self, r_disp):
        """Transform the coordinates from real to voxel in the interpolated image"""
        return np.rint(self._transform_disps(r_disp, self.z_scaling / self.z_xy_ratio)).astype(int)

    def _transform_real_to_layer(self, r_disp):
        """Transform the coordinates from real to voxel in the raw image"""
        return np.rint(self._transform_disps(r_disp, 1 / self.z_xy_ratio)).astype(int)

    def _transform_interpolated_to_layer(self, r_disp):
        """Transform the coordinates from real to voxel in the raw image"""
        return np.rint(self._transform_disps(r_disp, 1 / self.z_scaling)).astype(int)

    def load_unet(self):
        """
        load weights of the trained unet model
        """
        self.unet_model = load_model(os.path.join(self.models_path, self.unet_model_file))
        self.unet_model.save_weights(os.path.join(self.unet_weights_path, 'weights_initial.h5'))
        print("Loaded the 3D U-Net model")

    def segment_vol1(self, method="min_size"):
        self.vol = 1
        self.segresult._update_results(*self._segment(self.vol, method=method, print_shape=True))
        self.r_coordinates_segment_t0 = self.segresult.r_coordinates_segment.copy()

        # save the segmented cells of volume #1
        save_img3(z_siz_=self.z_siz, img=self.segresult.segmentation_auto,
                  path=self.auto_segmentation_vol1_path + "auto_t%04i_z%04i.tif")
        print(f"Segmented volume 1 and saved it")

    def _segment(self, vol, method, print_shape=False):
        """
        Make _segment (unet + _watershed)
        Input:
            vol: a specific volume
            method: used for watershed_3d(). "cell_num" or "min_size"
        Return:
            image_cell_bg: the cell/background regions obtained by unet.
            l_center_coordinates: center coordinates of segmented cells by _watershed
            segmentation_auto: individual cells segmented by _watershed
            image_gcn: raw image / 65536
            r_coordinates_segment: l_center_coordinates transformed to real scale
        """
        image_raw = read_image_ts(vol, self.raw_image_path, self.image_name, (1, self.z_siz + 1), print_=print_shape)
        # image_gcn will be used to correct tracking results
        image_gcn = (image_raw.copy() / 65536.0)
        image_cell_bg = self._predict_cellregions(image_raw, vol)

        # segment connected cell-like regions using _watershed
        segmentation_auto = self._watershed(image_cell_bg, method)

        # calculate coordinates of the centers of each segmented cell
        l_center_coordinates = snm.center_of_mass(segmentation_auto > 0, segmentation_auto,
                                                  range(1, segmentation_auto.max() + 1))
        r_coordinates_segment = self._transform_layer_to_real(l_center_coordinates)

        return image_cell_bg, l_center_coordinates, segmentation_auto, image_gcn, r_coordinates_segment

    def _predict_cellregions(self, image_raw, vol):
        """
        predict cell regions and save the results
        """
        if self.recalculate_unet:
            image_cell_bg = self._save_unet_regions(image_raw, vol)
        else:
            try:
                image_cell_bg = np.load(self.unet_path + "t%04i.npy" % vol, allow_pickle=True)
            except OSError:
                image_cell_bg = self._save_unet_regions(image_raw, vol)
        return image_cell_bg

    def _save_unet_regions(self, image_raw, vol):
        # pre-processing: local contrast normalization
        image_norm = np.expand_dims(_normalize_image(image_raw, self.noise_level), axis=(0, 4))
        # predict cell-like regions using 3D U-net
        image_cell_bg = unet3_prediction(image_norm, self.unet_model, shrink=self.shrink)
        np.save(self.unet_path + "t%04i.npy" % vol, np.array(image_cell_bg, dtype="float16"))
        return image_cell_bg

    def _watershed(self, image_cell_bg, method):
        """
        segment cells by _watershed
        """
        image_watershed2d_wo_border, _ = watershed_2d(image_cell_bg[0, :, :, :, 0], z_range=self.z_siz,
                                                      min_distance=7)
        _, image_watershed3d_wi_border, min_size, cell_num = watershed_3d(
            image_watershed2d_wo_border, samplingrate=[1, 1, self.z_xy_ratio], method=method,
            min_size=self.min_size, cell_num=self.cell_num, min_distance=3)
        segmentation_auto, fw, inv = relabel_sequential(image_watershed3d_wi_border)
        self.min_size = min_size
        if method == "min_size":
            self.cell_num = cell_num
        return segmentation_auto


class Tracker(Segmentation, Draw):

    def __init__(self,
                 volume_num, siz_xyz: tuple, z_xy_ratio, z_scaling, noise_level, min_size, beta_tk,
                 lambda_tk, maxiter_tk, folder_path, image_name, unet_model_file,
                 ffn_model_file, cell_num=0, ensemble=False, adjacent=False,
                 shrink=(24, 24, 2), miss_frame=None
                 ):
        Segmentation.__init__(self, volume_num, siz_xyz, z_xy_ratio, z_scaling,
                              image_name, unet_model_file, shrink)

        if not miss_frame:
            self.miss_frame = []
        else:
            self.miss_frame = miss_frame
        self.noise_level = noise_level  # a threshold to discriminate noise/artifacts from cells
        self.min_size = min_size  # a threshold to remove small objects which may be noise/artifacts
        self.beta_tk = beta_tk  # control coherence using a weighted average of movements of nearby points;
        # larger BETA includes more points, thus generates more coherent movements
        self.lambda_tk = lambda_tk  # control coherence by adding a loss of incoherence, large LAMBDA
        # generates larger penalty for incoherence.
        # maximum number of iterations; large values generate more accurate tracking.
        self.max_iteration = maxiter_tk
        self.ensemble = ensemble
        # use ensemble mode (False: single mode; value: number of predictions to make)
        self.adjacent = adjacent  # irrelevant in single mode
        self.folder_path = folder_path  # path of the folder storing related files
        self.ffn_model_file = ffn_model_file  # weight file of the trained FFN model
        self.manual_segmentation_vol1_path = None
        self.track_results_path = None
        self.track_information_path = None
        self.cell_num = cell_num
        self.cell_num_t0 = None
        self.Z_RANGE_INTERP = None

    def set_tracking(self, beta_tk, lambda_tk, maxiter_tk):
        if self.beta_tk == beta_tk and self.lambda_tk == lambda_tk and self.max_iteration == maxiter_tk:
            print("Tracking parameters were not modified")
        else:
            self.beta_tk = beta_tk
            self.lambda_tk = lambda_tk
            self.max_iteration = maxiter_tk
            print(f"Parameters were modified: beta_tk={self.beta_tk}, "
                  f"lambda_tk={self.lambda_tk}, maxiter_tk={self.max_iteration}")

    def make_folders(self):
        """
        make folders for storing data and results
        """
        print("Following folders were made under:", os.getcwd())
        folder_path = self.folder_path
        self.raw_image_path = _make_folder(os.path.join(folder_path, "data/"))
        self.auto_segmentation_vol1_path = _make_folder(os.path.join(folder_path, "auto_vol1/"))
        self.manual_segmentation_vol1_path = _make_folder(os.path.join(folder_path, "manual_vol1/"))
        self.track_information_path = _make_folder(os.path.join(folder_path, "track_information/"))
        self.models_path = _make_folder(os.path.join(folder_path, "models/"))
        self.unet_path = _make_folder(os.path.join(folder_path, "unet/"))
        track_results_path = get_tracking_path(self.adjacent, self.ensemble, folder_path)
        self.track_results_path = _make_folder(track_results_path)
        self.anim_path = _make_folder(os.path.join(folder_path, "anim/"))
        self.unet_weights_path = _make_folder(os.path.join(self.models_path, "unet_weights/"))

    def load_manual_seg(self):
        # load manually corrected _segment
        segmentation_manual = load_image(self.manual_segmentation_vol1_path, print_=False)
        # relabel cells sequentially
        self.segmentation_manual_relabels, _, _ = relabel_sequential(segmentation_manual)
        print("Loaded manual _segment at vol 1")

    def _retrain_preprocess(self):
        self.image_raw_vol1 = read_image_ts(1, self.raw_image_path, self.image_name, (1, self.z_siz + 1))
        self.train_image_norm = _normalize_image(self.image_raw_vol1, self.noise_level)
        self.label_vol1 = self.remove_2d_boundary(self.segmentation_manual_relabels) > 0
        self.train_label_norm = _normalize_label(self.label_vol1)
        print("Images were normalized")

        self.train_subimage = _divide_img(self.train_image_norm, self.unet_model.input_shape[1:4])
        self.train_subcells = _divide_img(self.train_label_norm, self.unet_model.input_shape[1:4])
        print("Images were divided")

        image_gen = ImageDataGenerator(rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, horizontal_flip=True, fill_mode='reflect')

        self.train_generator = _augmentation_generator(self.train_subimage, self.train_subcells, image_gen, batch_siz=8)
        self.valid_data = (self.train_subimage, self.train_subcells)
        print("Data for training 3D U-Net were prepared")

    def remove_2d_boundary(self, labels3d):
        labels_new = labels3d.copy()
        for z in range(self.z_siz):
            labels = labels_new[:,:,z]
            labels[find_boundaries(labels, mode='outer') == 1] = 0
        return labels_new

    def retrain_unet(self, iteration=10, weights_name="unet_weights_retrain_"):
        self._retrain_preprocess()

        self.unet_model.compile(loss='binary_crossentropy', optimizer="adam")
        self.unet_model.load_weights(os.path.join(self.unet_weights_path, 'weights_initial.h5'))

        # evaluate model prediction before retraining
        val_loss = self.unet_model.evaluate(self.train_subimage, self.train_subcells)
        print("val_loss before retraining: ", val_loss)
        self.val_losses = [val_loss]
        self.draw_retrain(step="before retrain")

        for step in range(1, iteration + 1):
            self.unet_model.fit_generator(self.train_generator, validation_data=self.valid_data, epochs=1,
                                          steps_per_epoch=60)
            loss = self.unet_model.history.history["val_loss"]
            if loss < min(self.val_losses):
                print("val_loss updated from ", min(self.val_losses), " to ", loss)
                self.unet_model.save_weights(os.path.join(self.unet_weights_path, weights_name + f"step{step}.h5"))
                self.draw_retrain(step)
            self.val_losses.append(loss)

    def draw_retrain(self, step, percentile_top=99.9, percentile_bottom=10):
        train_prediction = np.squeeze(
            unet3_prediction(np.expand_dims(self.train_image_norm, axis=(0, 4)), self.unet_model))
        fig, axs = plt.subplots(1, 2, figsize=(20, int(12 * self.x_siz / self.y_siz)))
        axs[0].imshow(np.max(self.label_vol1, axis=2), cmap="gray")
        axs[1].imshow(np.max(train_prediction, axis=2) > 0.5, cmap="gray")
        axs[0].set_title("Cell regions from manual segmentation at vol 1", fontdict=TITLE_STYLE)
        axs[1].set_title(f"Cell prediction at step {step} at vol 1", fontdict=TITLE_STYLE)
        plt.tight_layout()
        plt.pause(0.1)

    def _select_weights(self, step, weights_name="unet_weights_retrain_"):
        if step==0:
            self.unet_model.load_weights(os.path.join(self.unet_weights_path, 'weights_initial.h5'))
        elif step > 0:
            self.unet_model.load_weights((os.path.join(self.unet_weights_path, weights_name + f"step{step}.h5")))
            self.unet_model.save(os.path.join(self.unet_weights_path, "unet3_retrained.h5"))
        else:
            raise ValueError("step should be an interger >= 0")

    def interpolate_seg(self):
        # _interpolate layers in z axis
        self.seg_cells_interpolated_corrected = self._interpolate()
        self.Z_RANGE_INTERP = range(self.z_scaling // 2, self.seg_cells_interpolated_corrected.shape[2],
                                    self.z_scaling)
        # save labels in the first volume (interpolated)
        save_img3ts(self.Z_RANGE_INTERP, self.seg_cells_interpolated_corrected,
                    self.track_results_path + "track_results_t%04i_z%04i.tif", t=1)

        # calculate coordinates of cell centers at t=1
        center_points_t1 = snm.center_of_mass(self.segmentation_manual_relabels > 0,
                                              self.segmentation_manual_relabels,
                                              range(1, self.segmentation_manual_relabels.max() + 1))
        r_coordinates_manual_vol1 = self._transform_layer_to_real(center_points_t1)
        self.r_coordinates_tracked_t0 = r_coordinates_manual_vol1.copy()
        self.cell_num_t0 = r_coordinates_manual_vol1.shape[0]

    def _interpolate(self):
        seg_cells_interpolated, seg_cell_or_bg = gaussian_filter(
            self.segmentation_manual_relabels, z_scaling=self.z_scaling, smooth_sigma=2.5)
        seg_cells_interpolated_corrected = watershed_2d_markers(
            seg_cells_interpolated, seg_cell_or_bg, z_range=self.z_siz * self.z_scaling + 10)
        return seg_cells_interpolated_corrected[5:self.x_siz + 5,
               5:self.y_siz + 5, 5:self.z_siz * self.z_scaling + 5]

    def cal_subregions(self):
        # Compute subregions of each cells for quick "accurate correction"
        seg_16 = self.seg_cells_interpolated_corrected.astype("int16")

        self.region_list, self.region_width, self.region_xyz_min = get_subregions(seg_16, seg_16.max())
        self.pad_x, self.pad_y, self.pad_z = np.max(self.region_width, axis=0)
        self.label_padding = np.pad(seg_16,
                                    pad_width=((self.pad_x, self.pad_x),
                                               (self.pad_y, self.pad_y),
                                               (self.pad_z, self.pad_z)),
                                    mode='constant') * 0

    def check_multicells(self):
        # test if there are multiple cells marked as a single region
        for i, region in enumerate(self.region_list):
            assert np.sum(np.unique(label(region))) == 1, f"more than one cell in region {i + 1}"
        print("Checked mistakes of multiple cells as one: Correct!")

    def load_ffn(self):
        # load FFN model
        self.ffn_model = load_model(os.path.join(self.models_path, self.ffn_model_file))
        print("Loaded the FFN model")

    def initiate_tracking(self):
        # initiate all coordinates from vol=1 (t0)
        self.cells_on_boundary = np.zeros(self.cell_num_t0).astype(int)
        self.history_r_displacements = []
        self.history_r_displacements.append(np.zeros((self.cell_num_t0, 3)))
        self.history_r_segmented_coordinates = []
        self.history_r_segmented_coordinates.append(self.r_coordinates_segment_t0)
        self.history_r_tracked_coordinates = []
        self.history_r_tracked_coordinates.append(self.r_coordinates_tracked_t0)
        self.history_anim = []
        print("Initiated coordinates for tracking (from vol 1)")

    def match(self, target_volume, method="min_size"):
        """
        Match current volume and another target_volume
        Input:
            target_volume: the target volume to be matched
        """
        # skip frames that cannot be tracked
        if target_volume in self.miss_frame:
            print("target_volume is a miss_frame")
            return None

        # generate automatic _segment in current volume
        self.segresult._update_results(*self._segment(target_volume, method=method))

        # track by fnn + prgls
        r_coor_predicted, anim = self._predict_pos_once(source_volume=1, draw=True)

        # boundary cells
        cells_bd = self._get_cells_onBoundary(r_coor_predicted, self.ensemble)
        cells_on_boundary_local = self.cells_on_boundary.copy()
        cells_on_boundary_local[cells_bd] = 1

        # accurate correction
        _, i_disp_from_vol1_updated = \
            self._accurate_correction(cells_on_boundary_local, r_coor_predicted)
        print(f"Matching between vol 1 and vol {target_volume} was computed")
        return anim, [cells_on_boundary_local, target_volume, i_disp_from_vol1_updated, r_coor_predicted]

    def _accurate_correction(self, cells_on_boundary_local, r_coor_predicted):
        r_disp_from_vol1_updated = self.history_r_displacements[-1] + \
                                   (r_coor_predicted - self.history_r_tracked_coordinates[-1])
        i_disp_from_vol1_updated = self._transform_real_to_interpolated(r_disp_from_vol1_updated)
        for i in range(REP_NUM_CORRECTION):
            # update positions (from vol1) by correction
            r_disp_from_vol1_updated, i_disp_from_vol1_updated, r_disp_correction = \
                self._correction_once_interp(i_disp_from_vol1_updated, cells_on_boundary_local)

            # stop the repetition if correction converged
            stop_flag = self._evaluate_correction(r_disp_correction)
            if i == REP_NUM_CORRECTION - 1 or stop_flag:
                break
        return r_disp_from_vol1_updated, i_disp_from_vol1_updated

    def _predict_pos_once(self, source_volume, draw=False):
        """
        Predict cell coordinates using the transformation parameters in all repetitions
            from fnn_prgls()
        """
        # fitting the parameters for transformation
        C_t, BETA_t, coor_intermediate_list = self._fit_ffn_prgls(
            REP_NUM_PRGLS, self.history_r_segmented_coordinates[source_volume - 1])

        # Transform the coordinates
        r_coordinates_predicted = self.history_r_tracked_coordinates[source_volume - 1].copy()

        if draw:
            ax, fig = self._prepare_tracking_animation()
            plt_objs = []
            for i in range(len(C_t)):
                r_coordinates_predicted, r_coordinates_predicted_pre = self._predict_one_rep(
                    r_coordinates_predicted, coor_intermediate_list[i], BETA_t[i], C_t[i])
                plt_obj = self._get_tracking_anim(
                    ax, r_coordinates_predicted_pre, self.segresult.r_coordinates_segment,
                    r_coordinates_predicted, layercoord=False)
                plt_objs.append(plt_obj)
            anim = animation.ArtistAnimation(fig, plt_objs, interval=200).to_jshtml()
        else:
            for i in range(len(C_t)):
                r_coordinates_predicted, r_coordinates_predicted_pre = self._predict_one_rep(
                    r_coordinates_predicted, coor_intermediate_list[i], BETA_t[i], C_t[i])
            anim = None

        return r_coordinates_predicted, anim

    def _fit_ffn_prgls(self, rep, r_coordinates_segment_pre):
        """
        Appliy FFN + PR-GLS from t1 to t2 (multiple times) to get transformation
            parameters to predict cell coordinates
        Input:
            rep: the number of repetitions of (FFN + max_iteration times of PR-GLS)
        Return:
            C_t: list of C in each repetition (to predict the transformed coordinates)
            BETA_t: list of the parameter beta used in each repetition (to predict coordinates)
            coor_intermediate_list: list of the pre-transformed coordinates of automatically
                segmented cells in each repetition (to predict coordinates)
        """
        corr_intermediate = r_coordinates_segment_pre.copy()
        C_t = []
        BETA_t = []
        coor_intermediate_list = []
        for i in range(rep):
            coor_intermediate_list.append(corr_intermediate)
            C, corr_intermediate = self._ffn_prgls_once(i, corr_intermediate)
            C_t.append(C)
            BETA_t.append(self.beta_tk * (0.8 ** i))
        return C_t, BETA_t, coor_intermediate_list

    def _ffn_prgls_once(self, i, r_coordinates_segment_pre):
        init_match = initial_matching_quick(self.ffn_model, r_coordinates_segment_pre,
                                            self.segresult.r_coordinates_segment, 20)
        pre_transformation_pre = r_coordinates_segment_pre.copy()
        P, r_coordinates_segment_post, C = pr_gls_quick(pre_transformation_pre,
                                                        self.segresult.r_coordinates_segment,
                                                        init_match,
                                                        BETA=self.beta_tk * (0.8 ** i),
                                                        max_iteration=self.max_iteration,
                                                        LAMBDA=self.lambda_tk)
        return C, r_coordinates_segment_post

    def _predict_one_rep(self, r_coordinates_predicted_pre, coor_intermediate_list, BETA_t, C_t):
        """
        Predict cell coordinates using one set of the transformation parameters
            from fnn_prgls()
        Input:
            r_coordinates_predicted_pre: the coordinates before transformation
            coor_intermediate_list, BETA_t, C_t: one set of the transformation parameters
        Return:
            r_coordinates_predicted_post: the coordinates after transformation
        """

        length_auto_segmentation = np.size(coor_intermediate_list, axis=0)

        r_coordinates_predicted_tile = np.tile(r_coordinates_predicted_pre, (length_auto_segmentation, 1, 1))
        coor_intermediate_tile = np.tile(coor_intermediate_list, (self.cell_num_t0, 1, 1)).transpose((1, 0, 2))
        Gram_matrix = np.exp(-np.sum(np.square(r_coordinates_predicted_tile - coor_intermediate_tile),
                                     axis=2) / (2 * BETA_t * BETA_t))

        r_coordinates_predicted_post = r_coordinates_predicted_pre + np.dot(C_t, Gram_matrix).T

        return r_coordinates_predicted_post, r_coordinates_predicted_pre

    def _get_cells_onBoundary(self, r_coordinates_prgls, ensemble):
        """
        get cell near the boundary of the image
        """
        if ensemble:
            boundary_xy = 0
        else:
            boundary_xy = BOUNDARY_XY
        cells_bd = np.where(reduce(
            np.logical_or,
            [r_coordinates_prgls[:, 0] < boundary_xy,
             r_coordinates_prgls[:, 1] < boundary_xy,
             r_coordinates_prgls[:, 0] > self.x_siz - boundary_xy,
             r_coordinates_prgls[:, 1] > self.y_siz - boundary_xy,
             r_coordinates_prgls[:, 2] / self.z_xy_ratio < 0,
             r_coordinates_prgls[:, 2] / self.z_xy_ratio > self.z_siz])
        )
        return cells_bd

    def _correction_once_interp(self, i_displacement_from_vol1, cell_on_bound):
        """
        Correct the tracking for once (in interpolated image)
        """
        # generate current image of labels from the manually corrected _segment in volume 1
        i_l_tracked_cells_prgls_0, i_l_overlap_prgls_0 = self._transform_cells_quick(i_displacement_from_vol1)
        l_tracked_cells_prgls = i_l_tracked_cells_prgls_0[:, :,
                                self.z_scaling // 2:self.z_siz * self.z_scaling:self.z_scaling]
        l_overlap_prgls = i_l_overlap_prgls_0[:, :,
                          self.z_scaling // 2:self.z_siz * self.z_scaling:self.z_scaling]

        # overlapping regions of multiple cells are discarded before correction to avoid cells merging
        l_tracked_cells_prgls[np.where(l_overlap_prgls > 1)] = 0

        for i in np.where(cell_on_bound == 1)[0]:
            l_tracked_cells_prgls[l_tracked_cells_prgls == (i + 1)] = 0

        # accurate correction of displacement
        l_coordinates_prgls_int_move = \
            self.r_coordinates_tracked_t0 * np.array([1, 1, 1 / self.z_xy_ratio]) + \
            i_displacement_from_vol1 * np.array([1, 1, 1 / self.z_scaling])
        l_centers_unet_x_prgls = snm.center_of_mass(
            self.segresult.image_cell_bg[0, :, :, :, 0] + self.segresult.image_gcn, l_tracked_cells_prgls,
            range(1, self.seg_cells_interpolated_corrected.max() + 1))
        l_centers_unet_x_prgls = np.asarray(l_centers_unet_x_prgls)
        l_centers_prgls = np.asarray(l_coordinates_prgls_int_move)

        lost_cells = np.where(np.isnan(l_centers_unet_x_prgls)[:, 0])

        r_displacement_correction = l_centers_unet_x_prgls - l_centers_prgls
        r_displacement_correction[lost_cells, :] = 0
        r_displacement_correction[:, 2] = r_displacement_correction[:, 2] * self.z_xy_ratio

        # calculate the corrected displacement from vol #1
        r_displacement_from_vol1 = i_displacement_from_vol1 * np.array(
            [1, 1, self.z_xy_ratio / self.z_scaling]) + r_displacement_correction
        i_displacement_from_vol1_new = self._transform_real_to_interpolated(r_displacement_from_vol1)

        return r_displacement_from_vol1, i_displacement_from_vol1_new, r_displacement_correction

    def _transform_cells_quick(self, vectors3d, print_seq=False):
        """
        Move cells according to vectors3d
        Input:
            vectors3d: sequence (int), movement of each cell
        Return:
            output: transformed image
            mask: overlap between different labels (if value>1)
        """
        label_moved = self.label_padding.copy() * 0
        mask = label_moved.copy()
        for label in range(0, len(self.region_list)):
            if print_seq:
                print(label, end=" ")
            new_x_min = self.region_xyz_min[label][0] + vectors3d[label, 0] + self.pad_x
            new_y_min = self.region_xyz_min[label][1] + vectors3d[label, 1] + self.pad_y
            new_z_min = self.region_xyz_min[label][2] + vectors3d[label, 2] + self.pad_z
            subregion_previous = label_moved[new_x_min:new_x_min + self.region_width[label][0],
                                 new_y_min:new_y_min + self.region_width[label][1],
                                 new_z_min:new_z_min + self.region_width[label][2]]
            if len(subregion_previous.flatten()) == 0:
                continue
            subregion_new = subregion_previous * (1 - self.region_list[label]) + \
                            self.region_list[label] * (label + 1)
            label_moved[new_x_min:new_x_min + self.region_width[label][0],
            new_y_min:new_y_min + self.region_width[label][1],
            new_z_min:new_z_min + self.region_width[label][2]] = subregion_new
            mask[new_x_min:new_x_min + self.region_width[label][0],
            new_y_min:new_y_min + self.region_width[label][1],
            new_z_min:new_z_min + self.region_width[label][2]] += \
                (self.region_list[label] > 0).astype("int8")
        output = label_moved[self.pad_x:-self.pad_x, self.pad_y:-self.pad_y, self.pad_z:-self.pad_z]
        mask = mask[self.pad_x:-self.pad_x, self.pad_y:-self.pad_y, self.pad_z:-self.pad_z]

        return output, mask

    def _transform_motion_to_image(self, cells_on_boundary_local, i_disp_from_vol1_updated):
        i_tracked_cells_corrected, i_overlap_corrected = self._transform_cells_quick(i_disp_from_vol1_updated)
        # re-calculate boundaries by _watershed
        i_tracked_cells_corrected[i_overlap_corrected > 1] = 0
        for i in np.where(cells_on_boundary_local == 1)[0]:
            i_tracked_cells_corrected[i_tracked_cells_corrected == (i + 1)] = 0
        z_range = range(self.z_scaling // 2, self.z_siz * self.z_scaling, self.z_scaling)
        tracked_labels = watershed_2d_markers(
            i_tracked_cells_corrected[:, :, z_range], i_overlap_corrected[:, :, z_range],
            z_range=self.z_siz)
        return tracked_labels

    def _evaluate_correction(self, r_displacement_correction):
        """
        evaluate if the accurate correction should be stopped
        """
        i_disp_test = r_displacement_correction.copy()
        i_disp_test[:, 2] *= self.z_scaling / self.z_xy_ratio
        if np.nanmax(np.abs(i_disp_test)) >= 0.5:
            # print(np.nanmax(np.abs(i_disp_test)), end=",")
            return False
        else:
            # print(np.nanmax(np.abs(i_disp_test)))
            return True

    def track_animation(self, from_volume):
        self._reset_tracking_state(from_volume)
        ax, fig = self._make_6subplots()
        return ax, fig

    def track_and_confirm(self, from_volume, fig, ax):
        for vol in range(from_volume, self.volume_num + 1):
            self.track_one_vol(vol, fig, ax)
        return None

    def track_animation_replay(self, from_volume):
        fig, ax = plt.subplots(figsize=(14, int(21 * self.x_siz / self.y_siz)), tight_layout=True)
        plt.close(fig)
        ax.axis('off')
        track_process_images = []
        for volume in range(from_volume, self.volume_num + 1):
            try:
                im = mgimg.imread(self.anim_path + "track_anim_t%04i.png" % volume)
            except FileNotFoundError:
                continue
            implot = ax.imshow(im)
            track_process_images.append([implot])

        track_anim = animation.ArtistAnimation(fig, track_process_images, interval=200, repeat=False).to_jshtml()
        return track_anim

    def _reset_tracking_state(self, from_volume):
        assert from_volume >= 2, "from_volume should >= 2"
        current_vol = len(self.history_r_displacements)
        del self.history_r_displacements[from_volume - 1:]
        del self.history_r_segmented_coordinates[from_volume - 1:]
        del self.history_r_tracked_coordinates[from_volume - 1:]
        assert len(self.history_r_displacements) == from_volume - 1, \
            f"Currently data has been tracked until vol {current_vol}, the program cannot start from {from_volume}"
        # print(f"Currently data has been tracked until vol {current_vol}, start from vol {from_volume}")

    def track_one_vol(self, target_volume, fig, axc6, method="min_size"):
        """
        Track on volume
        """
        # skip frames that cannot be tracked
        if target_volume in self.miss_frame:
            save_img3ts(range(1, self.z_siz + 1), self.tracked_labels,
                        self.track_results_path + "track_results_t%04i_z%04i.tif", target_volume)
            self.history_r_displacements.append(self.history_r_displacements[-1])
            self.history_r_segmented_coordinates.append(self.segresult.r_coordinates_segment)
            self.history_r_tracked_coordinates.append(
                self.r_coordinates_tracked_t0 + self.history_r_displacements[-1])
            return None

        # make _segment of target volume
        self.segresult._update_results(*self._segment(target_volume, method=method))

        # FFN + PR-GLS predictions (single or ensemble)
        source_vols_list = get_reference_vols(self.ensemble, target_volume, adjacent=self.adjacent)
        list_predictions = []
        for source_vol in source_vols_list:
            r_coor_predicted, _ = self._predict_pos_once(source_volume=source_vol, draw=False)
            list_predictions.append(r_coor_predicted)
        r_coor_predicted_mean = trim_mean(list_predictions, 0.1, axis=0)

        # remove cells moved to the boundaries of the 3D image
        cells_bd = self._get_cells_onBoundary(r_coor_predicted_mean, self.ensemble)
        self.cells_on_boundary[cells_bd] = 1

        # accurate correction to get more accurate positions
        r_disp_from_vol1_updated, i_disp_from_vol1_updated = \
            self._accurate_correction(self.cells_on_boundary, r_coor_predicted_mean)

        # transform positions into images
        self.tracked_labels = self._transform_motion_to_image(self.cells_on_boundary, i_disp_from_vol1_updated)

        # save tracked labels
        save_img3ts(range(1, self.z_siz + 1), self.tracked_labels,
                    self.track_results_path + "track_results_t%04i_z%04i.tif", target_volume)

        self._draw_matching_6panel(target_volume, axc6, r_coor_predicted_mean, i_disp_from_vol1_updated)
        fig.canvas.draw()
        plt.savefig(self.anim_path + "track_anim_t%04i.png" % target_volume, bbox_inches='tight')

        # update and save points locations
        if self.ensemble:
            self.cells_on_boundary = \
                np.zeros(self.cell_num_t0).astype(int)  # in ensemble mode, cells on boundary are not deleted forever
        self.history_r_displacements.append(r_disp_from_vol1_updated)
        self.history_r_segmented_coordinates.append(self.segresult.r_coordinates_segment)
        self.history_r_tracked_coordinates.append(self.r_coordinates_tracked_t0 + r_disp_from_vol1_updated)

        return None
