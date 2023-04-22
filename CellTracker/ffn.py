import os
from glob import glob
from pathlib import Path
from typing import Tuple, Generator, Union

import numpy as np
import tensorflow as tf
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Concatenate
from tifffile import imread
from tqdm import tqdm

from CellTracker.synthesize import add_seg_errors, points_to_features

# parameters for feedforward network
RATIO_SEG_ERROR = 0.15
FFN_WEIGHTS_NAME = "weights_training_"
k_ptrs = 20  # number of neighbors for calculating relative coordinates

# parameters for generating data and for training
affine_level = 0.2
random_movement_level = 0.001
epochs = 30
steps_per_epoch = 60
batch_size = 128
NUMBER_FEATURES = 61


def affine_transform(points: ndarray, affine_level: float, rand_move_level: float) -> ndarray:
    """generate affine transformed points

    Notes
    -----
    points should have been normalized to have average of 0
    """
    random_transform = (np.random.rand(3, 3) - 0.5) * affine_level
    random_movements = (np.random.rand(*points.shape) - 0.5) * 4 * rand_move_level
    ptrs_affine = np.dot(points, np.eye(3) + random_transform) + random_movements
    return ptrs_affine


class DataGeneratorFFN:

    def __init__(self, points_normalized_nx3: ndarray):
        self.train_data_gen = self.generator_train_data(points_normalized_nx3)

    @staticmethod
    def generator_train_data(points_nx3: ndarray) -> Generator:
        n = points_nx3.shape[0]
        num_sets = 20
        sample_num_one_set = n * 2
        sample_num = sample_num_one_set * num_sets
        x_mxf = np.empty((sample_num, NUMBER_FEATURES * 2), dtype=np.float32)
        y_mx1 = np.empty((sample_num, 1), dtype=np.bool_)
        random_indexes = np.arange(sample_num)
        while True:
            for i in range(num_sets):
                points_tgt_nx3 = affine_transform(points_nx3, affine_level, random_movement_level)
                points_wi_seg_errors, replaced_indexes = add_seg_errors(points_tgt_nx3, ratio=RATIO_SEG_ERROR)
                s_ = slice(i * sample_num_one_set, (i + 1) * sample_num_one_set)
                points_to_features(x_mxf[s_, :], y_mx1[s_, 0], points_nx3, points_wi_seg_errors, replaced_indexes,
                                   method_features=features_of_points_ffn_quick, num_features=NUMBER_FEATURES,
                                   k_ptrs=k_ptrs)

            np.random.shuffle(random_indexes)
            for i in range(sample_num // batch_size):
                yield x_mxf[random_indexes[i * batch_size:(i + 1) * batch_size], :], \
                      y_mx1[random_indexes[i * batch_size:(i + 1) * batch_size], :]


def features_of_points_ffn_quick(points_nx3: ndarray, points_tgt_nx3: ndarray, k_neighbors: int, number_features: int,
                             knn_model) -> ndarray:
    distances_nxk, neighbors_indexes_nxk = knn_model.kneighbors(points_tgt_nx3)
    dist_mean_nx1x1 = np.mean(distances_nxk, axis=1)[:, None, None]
    neighbor_points_nxkx3 = points_nx3[neighbors_indexes_nxk[:, 1:k_neighbors + 1], :]
    coordinates_relative_nxkx3 = (neighbor_points_nxkx3 - points_tgt_nx3[:, None, :]) / dist_mean_nx1x1

    x_nxf = np.zeros((points_nx3.shape[0], number_features))
    x_nxf[:, 0:k_neighbors * 3] = coordinates_relative_nxkx3.reshape((-1, k_neighbors * 3))
    x_nxf[:, k_neighbors * 3] = dist_mean_nx1x1[:, 0, 0]
    return x_nxf


class TrainFFN:
    def __init__(self, model_name: str, points1_path: str = None, segmentation1_path: str = None, voxel_size: tuple = (1, 1, 1),
                 basedir: str = "./ffn_models"):
        """Set the model name and load/process a points set
        Notes
        -----
        segmentation1_path and voxel_size are used only when points1_path is None
        """
        self.path_model = Path(basedir)
        self.path_model.mkdir(exist_ok=True, parents=True)
        (self.path_model / "weights").mkdir(exist_ok=True, parents=True)
        self.model_name = model_name
        self.current_epoch = 1
        self.model = FFN()
        if points1_path is not None:
            self.points_t1 = normalize_points(np.loadtxt(points1_path))

        elif segmentation1_path is not None:
            slice_paths = sorted(glob(segmentation1_path))
            if len(slice_paths) == 0:
                # Raise an error if no image slices are found in the specified directory
                raise FileNotFoundError(f"No image in {segmentation1_path} was found")

            # Load the proofed segmentation and relabel it to sequential integers
            proofed_segmentation = imread(slice_paths).transpose((1, 2, 0))
            import scipy.ndimage.measurements as ndm
            points_t1 = np.asarray(
                ndm.center_of_mass(proofed_segmentation > 0,
                                   proofed_segmentation, range(1, proofed_segmentation.max() + 1)
                                   )
            )
            self.points_t1 = normalize_points(points_t1 * np.asarray(voxel_size)[None, :])
        else:
            raise ValueError("Either the segmentation1_path or the points1_path should be provided")
        self.optimizer = tf.keras.optimizers.Adam()
        self.points_generator = DataGeneratorFFN(self.points_t1)

    def train(self, num_epochs=10, iteration=5000, weights_name=FFN_WEIGHTS_NAME):
        train_loss_fn = tf.keras.losses.BinaryCrossentropy()
        train_loader = self.points_generator.train_data_gen

        start_epoch = self.current_epoch
        end_epoch = self.current_epoch + num_epochs
        for epoch in range(start_epoch, end_epoch):
            train_loss = 0
            n = 0
            with tqdm(total=iteration, desc=f'Epoch {epoch}/{end_epoch - 1}', ncols=50, unit='batch') as pbar:
                for X, y in train_loader:
                    with tf.GradientTape() as tape:
                        X_prediction = self.model(X)
                        X_loss = train_loss_fn(y, X_prediction)

                    gradients = tape.gradient(X_loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    train_loss += X_loss.numpy()
                    n += 1
                    pbar.update(1)
                    pbar.set_postfix(**{'Train loss': train_loss / n})

                    if n > iteration:
                        break

            self.model.save_weights(self.path_model / "weights" / f'{weights_name}_epoch{epoch}.h5')
            self.current_epoch += 1
        self.model.save_weights(self.path_model / (self.model_name + ".h5"))
        print(f"The trained models have been saved as: \n{str(self.path_model / (self.model_name + '.h5'))}")

    def select_ffn_weights(self, step, weights_name=FFN_WEIGHTS_NAME):
        """Load a trained ffn weight"""
        if step <= 0:
            raise ValueError("step should be an interger >= 0")
        self.model.load_weights(str(Path(self.path_model) / (weights_name + f"epoch{step}.h5")))

        print(f"Loaded the trained FFN model at step {step}")


class FFN(Model):
    def __init__(self):
        super().__init__()
        self.feat_layer1 = self._build_feat_layer1()
        self.combine_feat = Concatenate(axis=1)
        self.combine_feat2 = self._build_combine_feat2()
        self.pred = self._build_pred()

    def _build_feat_layer1(self):
        return tf.keras.Sequential([
            Dense(512, input_shape=(61,), use_bias=False),
            BatchNormalization(),
            LeakyReLU()
        ])

    def _build_combine_feat2(self):
        return tf.keras.Sequential([
            Dense(512, use_bias=False),
            BatchNormalization(),
            LeakyReLU()
        ])

    def _build_pred(self):
        return tf.keras.Sequential([
            Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        feat_x1 = self.feat_layer1(x[:, :61])
        feat_x2 = self.feat_layer1(x[:, 61:])
        combined_feat = self.combine_feat([feat_x1, feat_x2])
        combined_feat2 = self.combine_feat2(combined_feat)
        return self.pred(combined_feat2)


def initial_matching_ffn(ffn_model, ref: ndarray, tgt: ndarray, k_ptrs: int) -> ndarray:
    """
    This function compute initial matching between all pairs of points in reference and target points set.

    Parameters
    ----------
    ffn_model :
        The pretrained FFN model
    ref :
        The positions of the cells in the first volume
    tgt :
        The positions of the cells in the second volume
    k_ptrs :
        The number of neighboring points used for FFN

    Returns
    -------
    corr :
        The correspondence matrix between two point sets
    """
    nbors_ref = NearestNeighbors(n_neighbors=k_ptrs + 1).fit(ref)
    nbors_tgt = NearestNeighbors(n_neighbors=k_ptrs + 1).fit(tgt)

    ref_x_flat_batch = np.zeros((ref.shape[0], k_ptrs * 3 + 1), dtype='float32')
    tgt_x_flat_batch = np.zeros((tgt.shape[0], k_ptrs * 3 + 1), dtype='float32')

    for ref_i in range(ref.shape[0]):
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

    tgt_x_flat_batch_meshgrid = np.tile(tgt_x_flat_batch, (ref.shape[0], 1, 1)).transpose((1, 0, 2)).reshape(
        (ref.shape[0] * tgt.shape[0], k_ptrs * 3 + 1))

    corr = np.reshape(
        ffn_model.predict(
            np.concatenate((ref_x_flat_batch_meshgrid, tgt_x_flat_batch_meshgrid), axis=1), batch_size=1024),
        (tgt.shape[0], ref.shape[0]))
    return corr


def normalize_points(points: ndarray, return_para: bool = False) -> Union[ndarray, Tuple[ndarray, Tuple[any, any]]]:
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
