#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:37:20 2017

@author: wen
"""
import itertools
import math
import os
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv3D, LeakyReLU, Input, MaxPooling3D, \
    UpSampling3D, concatenate, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from .preprocess import load_image, _make_folder, _normalize_image, _normalize_label

warnings.filterwarnings('ignore')


def unet3_a():
    """
    Generate a 3D unet model used in figure 2-S1a (eLife 2021)
    Returns:
        obj: keras.Model
    """
    pool_size = up_size = (2, 2, 1)
    inputs = Input((160, 160, 16, 1))
    return _unet3_depth3(inputs, pool_size, up_size)


def unet3_b():
    """
    Generate a 3D unet model used in figure 2-S1b
        Returns:
        obj: keras.Model
    """
    pool_size = up_size = (2, 2, 1)
    inputs = Input((96, 96, 8, 1))

    downscale_p = partial(_downscale, transform=_conv3d_relu_bn)
    upscale_p = partial(_upscale, transform=_conv3d_relu_bn)

    conv_level0, pool_level1 = downscale_p(64, 64, pool_size, inputs)
    conv_level1, pool_level2 = downscale_p(128, 128, pool_size, pool_level1)

    up_level1 = upscale_p(256, 256, up_size, conv_level1, pool_level2)
    up_level0 = upscale_p(128, 128, up_size, conv_level0, up_level1)

    output_m2 = _conv3d_relu_bn(64, up_level0)
    output_m1 = _conv3d_relu_bn(64, output_m2)
    predictions = Conv3D(1, 1, padding='same', activation='sigmoid')(output_m1)

    unet_model = Model(inputs=inputs, outputs=predictions)

    return unet_model


def unet3_c():
    """
    Generate a 3D unet model used in figure 2-S1c
        Returns:
        obj: keras.Model
    """
    pool_size = up_size = (2, 2, 2)
    inputs = Input((64, 64, 64, 1))
    return _unet3_depth3(inputs, pool_size, up_size)


def _unet3_depth3(inputs, pool_size, up_size):
    downscale_p = partial(_downscale, transform=_conv3d_leakyrelu_bn)
    upscale_p = partial(_upscale, transform=_conv3d_leakyrelu_bn)
    conv_level0, pool_level1 = downscale_p(8, 16, pool_size, inputs)
    conv_level1, pool_level2 = downscale_p(16, 32, pool_size, pool_level1)
    conv_level2, pool_level3 = downscale_p(32, 64, pool_size, pool_level2)
    up_level2 = upscale_p(64, 64, up_size, conv_level2, pool_level3)
    up_level1 = upscale_p(32, 32, up_size, conv_level1, up_level2)
    up_level0 = upscale_p(16, 16, up_size, conv_level0, up_level1)
    output_m2 = _conv3d_leakyrelu_bn(8, up_level0)
    output_m1 = _conv3d_leakyrelu_bn(8, output_m2)
    predictions = Conv3D(1, 1, padding='same', activation='sigmoid')(output_m1)
    unet_model = Model(inputs=inputs, outputs=predictions)
    return unet_model


def _conv3d_leakyrelu_bn(filter_num, inputs):
    """
    Build a block to perform convolution (3d) + LeakyReLU + BatchNormalization
    Args:
        filter_num: (int), number of conv filters
        inputs: (obj, numpy.ndarray), shape: (sample, x, y, z, channel)

    Returns:
        obj, numpy.ndarray
    """
    outputs = Conv3D(filter_num, 3, padding='same')(inputs)
    outputs = LeakyReLU()(outputs)
    outputs = BatchNormalization()(outputs)
    return outputs


def _conv3d_relu_bn(filter_num, inputs):
    """
    Build a block to perform convolution (3d) + ReLU + BatchNormalization
    Args:
        filter_num: (int), number of conv filters
        inputs: (obj, numpy.ndarray), shape: (sample, x, y, z, channel)

    Returns:
        obj, numpy.ndarray
    """
    conv_2 = Conv3D(filter_num, 3, padding='same', activation='relu')(inputs)
    conv_2 = BatchNormalization()(conv_2)
    return conv_2


def _downscale(f1_num, f2_num, pool_size, inputs, transform):
    """
    Build a block to perform twice transformations (conv+...) followed by once max pooling
    Args:
        f1_num: (int), number of conv filters 1
        f2_num: (int), number of conv filters 2
        pool_size: (tuple), window size for max pooling
        inputs: (obj, numpy.ndarray), shape: (sample, x, y, z, channel)
        transform: (function), the transformation method

    Returns:
        im_output: (obj, numpy.ndarray), output at the save level
        im_downscaled: (obj, numpy.ndarray), output at the lower (downscaled) level
    """
    im_1 = transform(f1_num, inputs)
    im_output = transform(f2_num, im_1)
    im_downscaled = MaxPooling3D(pool_size=pool_size)(im_output)
    return im_output, im_downscaled


def _upscale(f1_num, f2_num, size, input_horiz, input_vertical, transform):
    """
    Build a block to perform twice transformations (conv+...) (on input1) followed by once upsampling,
    and then concatenated the results with input2
    Args:
        f1_num: (int), number of conv filters 1
        f2_num: (int), number of conv filters 1
        size: (tuple), window size for upsampling
        input_horiz: (obj, numpy.ndarray), shape: (sample, x, y, z, channel), previous input2
        input_vertical: (obj, numpy.ndarray), shape: (sample, x, y, z, channel), previous input1
        transform: (function), the transformation method

    Returns:
        (obj, numpy.ndarray), the concatenated output
    """
    im_1 = transform(f1_num, input_vertical)
    im_2 = transform(f2_num, im_1)
    im_up_concatenated = concatenate([UpSampling3D(size=size)(im_2), input_horiz])
    return im_up_concatenated


def unet3_prediction(img, model, shrink=(24, 24, 2)):
    """
    Predict cell/non-cell regions by applying 3D U-net on each sub-sub_images.
    Args:
        img: (obj, numpy.ndarry), shape: (sample, x, y, z, channel), the normalized images to be segmented.
        model: (obj, keras.Model), the pre-trained 3D U-Net model.
        shrink: (tuple), the surrounding voxels to make pad.
         it is also used to discard surrounding regions of each predicted sub-region.

    Returns:
        out_img: (obj, numpy.ndarry), predicted cell regions, shape: (sample, x, y, z, channel)

    Note:
        input images is expanded outside by "reflection" to predict the voxels near borders.
    """
    out_centr_siz1 = model.output_shape[1] - shrink[0] * 2  # size of the center part of the prediciton by unet
    out_centr_siz2 = model.output_shape[2] - shrink[1] * 2
    out_centr_siz3 = model.output_shape[3] - shrink[2] * 2

    x_siz, y_siz, z_siz = img.shape[1:4]  # size of the input sub_images

    _x_siz, _num_x = _get_sizes_padded_im(x_siz,
                                          out_centr_siz1)  # size of the expanded sub_images and number of subregions
    _y_siz, _num_y = _get_sizes_padded_im(y_siz, out_centr_siz2)
    _z_siz, _num_z = _get_sizes_padded_im(z_siz, out_centr_siz3)

    before1, before2, before3 = shrink  # "pad_width" for numpy.pad()
    after1, after2, after3 = before1 + (_x_siz - x_siz), before2 + (_y_siz - y_siz), before3 + (_z_siz - z_siz)

    img_padded = np.pad(img[0, :, :, :, 0], ((before1, after1), (before2, after2), (before3, after3)), 'reflect')
    img_padded = np.expand_dims(img_padded, axis=(0, 4))

    slice_prediction_center = np.s_[0, before1: before1 + out_centr_siz1,
                              before2: before2 + out_centr_siz2,
                              before3: before3 + out_centr_siz3, 0]

    unet_siz1, unet_siz2, unet_siz3 = model.input_shape[1:4]  # size of the input for the unet model

    # the expanded sub_images was predicted on each sub-sub_images
    expanded_img = np.zeros((1, _x_siz, _y_siz, _z_siz, 1), dtype='float32')
    for i, j, k in itertools.product(range(_num_x), range(_num_y), range(_num_z)):
        slice_prediction = np.s_[:, i * out_centr_siz1: i * out_centr_siz1 + unet_siz1,
                           j * out_centr_siz2: j * out_centr_siz2 + unet_siz2,
                           k * out_centr_siz3: k * out_centr_siz3 + unet_siz3, :]
        slice_write = np.s_[0, i * out_centr_siz1: (i + 1) * out_centr_siz1,
                      j * out_centr_siz2: (j + 1) * out_centr_siz2,
                      k * out_centr_siz3: (k + 1) * out_centr_siz3, 0]
        prediction_subregion = model.predict(img_padded[slice_prediction])
        expanded_img[slice_write] = prediction_subregion[slice_prediction_center]
    out_img = expanded_img[:, 0:x_siz, 0:y_siz, 0:z_siz, :]
    return out_img


def _get_sizes_padded_im(img_siz_i, out_centr_siz_i):
    """
    Calculate the sizes and number of subregions to prepare the padded sub_images
    Args:
        img_siz_i: (int) size of raw sub_images along axis i
        out_centr_siz_i: (int), size of the center of the prediction by unet, along axis i

    Returns:
        temp_siz_i: (int), size of the padded sub_images along axis i
        num_axis_i: (int), number of the subregions (as inputs for unet) along axis i
    """
    num_axis_i = int(math.ceil(img_siz_i * 1.0 / out_centr_siz_i))
    temp_siz_i = num_axis_i * out_centr_siz_i
    return temp_siz_i, num_axis_i


def _divide_img(img, unet_siz):
    """
    Divide an sub_images into multiple sub_images with the size used by the defined UNet
    Args:
        img: (obj, numpy.ndarray) shape (x, y, z), input sub_images
        unet_siz: (tuple), (x_siz, y_siz, z_siz) input size of the UNet

    Returns:
        obj, numpy.ndarray: shape (number_subimages, x, y, z, 1) sub_images
    """
    x_siz, y_siz, z_siz = img.shape
    x_input, y_input, z_input = unet_siz
    img_list = []
    for i, j, k in itertools.product(range(x_siz * 2 // x_input), range(y_siz * 2 // y_input),
                                     range(z_siz * 2 // z_input)):
        idx_x = i * x_input // 2 if i * x_input // 2 + x_input <= x_siz else x_siz - x_input
        idx_y = j * y_input // 2 if j * y_input // 2 + y_input <= y_siz else y_siz - y_input
        idx_z = k * z_input // 2 if k * z_input // 2 + z_input <= z_siz else z_siz - z_input
        img_list.append(img[idx_x:idx_x + x_input, idx_y:idx_y + y_input, idx_z:idx_z + z_input])
    return np.expand_dims(np.array(img_list), axis=4)


def _augmentation_generator(sub_images, sub_cells, img_gen, batch_siz):
    """
    This function generates the same style of augmentations for all 2D layers in both sub_images
    and its corresponding sub_cells.
    Args:
        sub_images: (obj, numpy.ndarray), shape (number_subimages, x, y, z, 1) sub_images
        sub_cells: (obj, numpy.ndarray), shape (number_subcells, x, y, z, 1) sub_cells
        img_gen: (obj. keras.preprocessing.image.ImageDataGenerator), A generator for 2D images
        batch_siz: (int), batch_siz used during training the U-Net.

    Returns:
        Generator: images and its corresponding labels, both with shape (batch_size, x, y, z, 1)
    """
    num_subimgs, x_siz, y_siz, z_siz = np.shape(sub_images)[0:4]
    while 1:
        seed_aug = np.random.randint(1, 100000)  # defined a random seed to apply the same augmentation
        image_gen = np.zeros((batch_siz, x_siz, y_siz, z_siz, 1), dtype='float32')
        cell_gen = np.zeros((batch_siz, x_siz, y_siz, z_siz, 1), dtype='int32')
        start = np.random.randint(0, num_subimgs - batch_siz)
        for z in range(0, z_siz):
            gx = img_gen.flow(sub_images[start:start + batch_siz, :, :, z, :], batch_size=batch_siz, seed=seed_aug)
            image_gen[:, :, :, z, :] = gx.next()
            gy = img_gen.flow(sub_cells[start:start + batch_siz, :, :, z, :], batch_size=batch_siz, seed=seed_aug)
            cell_gen[:, :, :, z, :] = gy.next()
        yield image_gen, cell_gen


class TrainingUNet3D:

    def __init__(self, noise_level, folder_path, model):
        self.x_siz, self.y_siz, self.z_siz = None, None, None
        self.noise_level = noise_level
        self.folder_path = folder_path
        self.train_image_path = None
        self.train_label_path = None
        self.valid_image_path = None
        self.valid_label_path = None
        self.models_path = ""
        self.make_folders()
        self.model = model
        self.model.compile(loss='binary_crossentropy', optimizer="adam")
        self.model.save_weights(os.path.join(self.models_path,'weights_initial.h5'))

    def make_folders(self):
        """
        make folders for storing data and results
        """
        print("Made folders under:", os.getcwd())
        folder_path = self.folder_path
        print("Following folders were made: ")
        self.train_image_path = _make_folder(os.path.join(folder_path, "train_image/"))
        self.train_label_path = _make_folder(os.path.join(folder_path, "train_label/"))
        self.valid_image_path = _make_folder(os.path.join(folder_path, "valid_image/"))
        self.valid_label_path = _make_folder(os.path.join(folder_path, "valid_label/"))
        self.models_path = _make_folder(os.path.join(folder_path, "models/"))

    def train(self):
        self._load_dataset()
        self._preprocess()
        self._train()

    def _load_dataset(self):
        self.train_image = load_image(self.train_image_path)
        self.x_siz, self.y_siz, self.z_size = self.train_image.shape
        self.train_label = load_image(self.train_label_path)
        self.valid_image = load_image(self.valid_image_path)
        self.valid_label = load_image(self.valid_label_path)

    def _preprocess(self):
        self.train_image_norm = _normalize_image(self.train_image, self.noise_level)
        self.valid_image_norm = _normalize_image(self.valid_image, self.noise_level)
        self.train_label_norm = _normalize_label(self.train_label)
        self.valid_label_norm = _normalize_label(self.valid_label)
        print("Images were normalized")

        self.train_subimage = _divide_img(self.train_image_norm, self.model.input_shape[1:4])
        self.valid_subimage = _divide_img(self.valid_image_norm, self.model.input_shape[1:4])
        self.train_subcells = _divide_img(self.train_label_norm, self.model.input_shape[1:4])
        self.valid_subcells = _divide_img(self.valid_label_norm, self.model.input_shape[1:4])
        print("Images were divided")

        image_gen = ImageDataGenerator(rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, horizontal_flip=True, fill_mode='reflect')

        self.train_generator = _augmentation_generator(self.train_subimage, self.train_subcells, image_gen, batch_siz=8)
        self.valid_data = (self.valid_subimage, self.valid_subcells)
        print("Data for training 3D U-Net were prepared")

    def draw_dataset(self, percentile_top=99.9, percentile_bottom=10):
        axs = self._show_4images(percentile_bottom, percentile_top,
                                 (self.train_image, self.train_label, self.valid_image, self.valid_label))
        axs[0, 0].set_title("Max projection of image (train)")
        axs[0, 1].set_title("Max projection of cell annotation (train)")
        axs[1, 0].set_title("Max projection of image (validation)")
        axs[1, 1].set_title("Max projection of cell annotation (validation)")
        plt.tight_layout()
        plt.pause(0.1)

    def draw_norm_dataset(self, percentile_top=99.9, percentile_bottom=10):
        axs = self._show_4images(percentile_bottom, percentile_top,
                                 (self.train_image_norm, self.train_label_norm, self.valid_image_norm,
                                  self.valid_label_norm))
        axs[0, 0].set_title("Max projection of normalized image (train)")
        axs[0, 1].set_title("Max projection of cell annotation (train)")
        axs[1, 0].set_title("Max projection of normalized image (validation)")
        axs[1, 1].set_title("Max projection of cell annotation (validation)")
        plt.tight_layout()
        plt.pause(0.1)

    def _show_4images(self, percentile_bottom, percentile_top, imgs):
        fig, axs = plt.subplots(2, 2, figsize=(20, int(24 * self.x_siz / self.y_siz)))
        vmax_train = np.percentile(imgs[0], percentile_top)
        vmax_valid = np.percentile(imgs[2], percentile_top)
        vmin_train = np.percentile(imgs[0], percentile_bottom)
        vmin_valid = np.percentile(imgs[2], percentile_bottom)
        axs[0, 0].imshow(np.max(imgs[0], axis=2), vmin=vmin_train, vmax=vmax_train, cmap="gray")
        axs[0, 1].imshow(np.max(imgs[1], axis=2), cmap="gray")
        axs[1, 0].imshow(np.max(imgs[2], axis=2), vmin=vmin_valid, vmax=vmax_valid, cmap="gray")
        axs[1, 1].imshow(np.max(imgs[3], axis=2), cmap="gray")
        return axs

    def draw_divided_train_data(self, percentile_top=99.9, percentile_bottom=10):
        vmax_train = np.percentile(self.train_image_norm, percentile_top)
        vmin_train = np.percentile(self.train_image_norm, percentile_bottom)
        fig, axs = plt.subplots(4, 8, figsize=(20, int(24 * self.x_siz / self.y_siz)))
        idx = np.random.randint(self.train_subimage.shape[0], size=16)
        for i, j in itertools.product(range(4), range(4)):
            axs[i, 2 * j].imshow(np.max(self.train_subimage[idx[i * 4 + j], :, :, :, 0], axis=2), vmin=vmin_train,
                                 vmax=vmax_train, cmap="gray")
            axs[i, 2 * j].axis("off")
        for i, j in itertools.product(range(4), range(4)):
            axs[i, 2 * j + 1].imshow(np.max(self.train_subcells[idx[i * 4 + j], :, :, :, 0], axis=2), cmap="gray")
            axs[i, 2 * j + 1].axis("off")
        plt.tight_layout()
        plt.pause(0.1)

    def _train(self, iteration=100, weights_name="weights_training_"):
        self.model.load_weights(os.path.join(self.models_path,'weights_initial.h5'))
        for step in range(1, iteration+1):
            self.model.fit_generator(self.train_generator, validation_data=self.valid_data, epochs=1, steps_per_epoch=60)
            if step == 1:
                self.val_losses = [self.model.history.history["val_loss"]]
                print("val_loss at step 1: ", min(self.val_losses))
                self.model.save_weights(os.path.join(self.models_path, weights_name + f"step{step}.h5"))
                self.draw_prediction(step)
            else:
                loss = self.model.history.history["val_loss"]
                if loss<min(self.val_losses):
                    print("val_loss updated from ", min(self.val_losses)," to ", loss)
                    self.model.save_weights(os.path.join(self.models_path, weights_name + f"step{step}.h5"))
                    self.draw_prediction(step)
                self.val_losses.append(loss)

    def _select_weights(self, step, weights_name="weights_training_"):
        self.model.load_weights(os.path.join(self.models_path, weights_name + f"step{step}.h5"))
        self.model.save(os.path.join(self.models_path, "unet3_pretrained.h5"))

    def draw_prediction(self, step, percentile_top=99.9, percentile_bottom=10):
        train_prediction = np.squeeze(unet3_prediction(np.expand_dims(self.train_image_norm,axis=(0,4)), self.model))
        valid_prediction = np.squeeze(unet3_prediction(np.expand_dims(self.valid_image_norm,axis=(0,4)), self.model))
        axs = self._show_4images(percentile_bottom, percentile_top,
                                 (self.train_image, train_prediction, self.valid_image, valid_prediction))
        axs[0, 0].set_title("Image (train)")
        axs[0, 1].set_title(f"Cell prediction at step {step} (train)")
        axs[1, 0].set_title("Max projection of image (validation)")
        axs[1, 1].set_title(f"Cell prediction at step {step} (validation)")
        plt.tight_layout()
        plt.pause(0.1)