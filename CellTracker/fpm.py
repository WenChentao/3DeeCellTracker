from glob import glob
from pathlib import Path
from typing import Tuple, Generator, Union

import numpy as np
import tensorflow as tf
from keras.layers import MaxPooling2D
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Concatenate, Conv2D, Flatten, DepthwiseConv2D, Add
from tifffile import imread
from tqdm import tqdm

from CellTracker.utils import normalize_points
from CellTracker.synthesize_deformation import DataGeneratorFPN, K_NEIGHBORS, NUM_FEATURES, spherical_features_of_points

FPN_WEIGHTS_NAME = "weights_training_"
RANGE_ROT_TGT = 180
NUM_KERNELS = 128
NUM_FEATURES2 = 4


class TrainFPM:
    def __init__(self, model_name: str, match_model: Model, points1_path: str = None, segmentation1_path: str = None, voxel_size: tuple = (1, 1, 1),
                 range_rotation_tgt: float = RANGE_ROT_TGT,
                 basedir: str = "./fpm_models", move_factor: float = 0.2):

        """
        Set the model name and load/process a points set

        Notes
        -----
        segmentation1_path and voxel_size are used only when points1_path is None
        """
        # Create the model directory
        self.path_model = Path(basedir)
        self.path_model.mkdir(exist_ok=True, parents=True)
        (self.path_model / "weights").mkdir(exist_ok=True, parents=True)

        # Set the model name and current epoch
        self.model_name = model_name
        self.current_epoch = 1

        self.model = match_model

       # Load the points set
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
        self.points_generator = DataGeneratorFPN(self.points_t1, num_rotations=10, num_deformations=10,
                                                 range_rotation_tgt=range_rotation_tgt,
                                                 movement_factor=move_factor)

    def train(self, num_epochs=10, iteration=5000, weights_name=FPN_WEIGHTS_NAME):
        train_loss_fn = tf.keras.losses.BinaryCrossentropy()
        train_loader = self.points_generator.train_data_gen
        accuracy = tf.keras.metrics.BinaryAccuracy()

        start_epoch = self.current_epoch
        end_epoch = self.current_epoch + num_epochs
        for epoch in range(start_epoch, end_epoch):
            train_loss = 0
            train_acc = 0
            n = 0
            with tqdm(total=iteration, desc=f'Epoch {epoch}/{end_epoch - 1}', ncols=50, unit='batch') as pbar:
                for X, y in train_loader:
                    with tf.GradientTape() as tape:
                        # Calculate the loss
                        X_prediction = self.model(X)
                        X_loss = train_loss_fn(y, X_prediction)
                        # Calculate accuracy
                        accuracy.update_state(y, X_prediction)

                    gradients = tape.gradient(X_loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    train_loss += X_loss.numpy()
                    train_acc += accuracy.result().numpy()
                    n += 1
                    pbar.update(1)
                    pbar.set_postfix(**{'Train loss': train_loss / n, 'Train Accuracy': train_acc / n})

                    if n > iteration:
                        break

            self.model.save_weights(self.path_model / "weights" / f'{weights_name}_epoch{epoch}.h5')
            self.current_epoch += 1
        self.model.save_weights(self.path_model / (self.model_name + ".h5"))
        print(f"The trained models have been saved as: \n{str(self.path_model / (self.model_name + '.h5'))}")

    def select_ffn_weights(self, step, weights_name=FPN_WEIGHTS_NAME):
        """Load a trained ffn weight"""
        if step <= 0:
            raise ValueError("step should be an interger >= 0")
        self.model.load_weights(str(Path(self.path_model) / (weights_name + f"epoch{step}.h5")))

        print(f"Loaded the trained FFN model at step {step}")


class FlexiblePointMatcherOriginal(Model):
    """
    A class that defines a custom architecture for matching point sets of cells from two 3D images.
    """

    def __init__(self):
        super().__init__()
        self.feat_layer1 = self._build_feat_layer1()
        self.combine_feat = Concatenate(axis=1)
        self.combine_feat2 = self._build_combine_feat2()
        self.pred = self._build_pred()

    def _build_feat_layer1(self):
        return tf.keras.Sequential([
            Flatten(),
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

    def call(self, points_xy):
        feat_x1 = self.feat_layer1(points_xy[..., 0])
        feat_x2 = self.feat_layer1(points_xy[..., 1])
        combined_feat = self.combine_feat([feat_x1, feat_x2])
        combined_feat2 = self.combine_feat2(combined_feat)
        return self.pred(combined_feat2)


# class FlexiblePointMatcherCat(Model):
#     """
#     A class that defines a custom architecture for matching point sets of cells from two 3D images.
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.cnn = self.up_cnn()
#         self.mlp = self.down_mlp()
#         self.pred = self.classify()
#
#     @staticmethod
#     def up_cnn():
#         input_layer = Input(shape=(K_NEIGHBORS + 2, NUM_FEATURES, 1))
#
#         # First Convolutional block with BatchNorm and LeakyReLU
#         x = Conv2D(64, kernel_size=(K_NEIGHBORS + 2, 1), padding='valid')(input_layer)
#         x = BatchNormalization()(x)
#         x1 = LeakyReLU()(x)
#
#         # Second Convolutional block with BatchNorm and LeakyReLU
#         x = Conv2D(128, kernel_size=(1, 1))(x1)
#         x = BatchNormalization()(x)
#         x2 = LeakyReLU()(x)
#
#         # Add the skip connection from the first block to the second
#         x = Concatenate(axis=-1)([x1, x2])
#
#         # Third Convolutional block with BatchNorm and LeakyReLU
#         x = Conv2D(256, kernel_size=(1, 1))(x)
#         x = BatchNormalization()(x)
#         x3 = LeakyReLU()(x)
#
#         # Add the skip connection from the second block to the third
#         x = Concatenate(axis=-1)([x2, x3])
#         x = Flatten()(x)
#
#         return Model(inputs=input_layer, outputs=x)
#
#     @staticmethod
#     def down_mlp():
#         return tf.keras.Sequential([
#             Dense(256, use_bias=False),
#             BatchNormalization(),
#             LeakyReLU(),
#             Dense(64, use_bias=False),
#             BatchNormalization(),
#             LeakyReLU(),
#         ])
#
#     @staticmethod
#     def classify():
#         return tf.keras.Sequential([
#             Dense(1, activation='sigmoid')
#         ])
#
#     def call(self, points_xy):
#         expanded_feature_x = self.cnn(points_xy[..., 0])
#         expanded_feature_y = self.cnn(points_xy[..., 1])
#         expanded_feature = Concatenate(axis=1)([expanded_feature_x, expanded_feature_y])
#         contracted_feature = self.mlp(expanded_feature)
#         return self.pred(contracted_feature)


# class FlexiblePointMatcherAdd(Model):
#     """
#     A class that defines a custom architecture for matching point sets of cells from two 3D images.
#     """
#
#     def __init__(self, num_skip: int):
#         super().__init__()
#         self.cnn = self.up_cnn(num_skip)
#         self.cnn2 = self.down_cnn(num_skip)
#         self.pred = self.classify()
#
#     def up_cnn(self, num_skip: int):
#         input_layer = Input(shape=(K_NEIGHBORS + 2, NUM_FEATURES, 1))
#
#         x = self.conv_bn_lr(input_layer, kernel_size=(K_NEIGHBORS + 2, 1))
#
#         for i in range(num_skip):
#             x_conv = self.conv_bn_lr(x)
#             x = Add()([x, x_conv])
#
#         x = x[:, 0, :, :]
#         return Model(inputs=input_layer, outputs=x)
#
#     @staticmethod
#     def conv_bn_lr(x1, kernel_size=(1, 1), padding='valid'):
#         x = Conv2D(NUM_KERNELS, kernel_size=kernel_size, padding=padding)(x1)
#         x = BatchNormalization()(x)
#         x2 = LeakyReLU()(x)
#         return x2
#
#     def down_cnn(self, num_skip: int):
#         input_layer = Input(shape=(NUM_FEATURES, NUM_KERNELS, 2))
#         x = self.conv_bn_lr(input_layer, kernel_size=(4, 3), padding='same')
#
#         for i in range(num_skip):
#             x_conv = self.conv_bn_lr(x, kernel_size=(4, 3), padding='same')
#             x = Add()([x, x_conv])
#
#         return Model(inputs=input_layer, outputs=x)
#
#     @staticmethod
#     def classify():
#         return tf.keras.Sequential([
#             MaxPooling2D(pool_size=(4, 2)),
#             Flatten(),
#             Dense(128, use_bias=False),
#             BatchNormalization(),
#             LeakyReLU(),
#             Dense(1, activation='sigmoid')
#         ])
#
#     def call(self, points_xy):
#         expanded_feature_x = self.cnn(points_xy[..., 0])
#         expanded_feature_y = self.cnn(points_xy[..., 1])
#         expanded_feature = tf.stack([expanded_feature_x, expanded_feature_y], axis=-1)
#         contracted_feature = self.cnn2(expanded_feature)
#         return self.pred(contracted_feature)


class FlexiblePointMatcherConv(Model):
    """
    A class that defines a custom architecture for matching point sets of cells from two 3D images.
    """

    def __init__(self, num_skip: int):
        super().__init__()
        self.cnn = self.up_cnn(num_skip)
        self.cnn2 = self.down_cnn(num_skip)

    def up_cnn(self, num_skip: int):
        input_layer = Input(shape=(K_NEIGHBORS + 2, NUM_FEATURES, 1))

        x = self.conv_bn_lr(input_layer, kernel_size=(K_NEIGHBORS + 2, 1))

        for i in range(num_skip):
            x_conv = self.conv_bn_lr(x)
            x = Add()([x, x_conv])

        x = x[:, 0, :, :]
        return Model(inputs=input_layer, outputs=x)

    @staticmethod
    def conv_bn_lr(x1, n_kernels=NUM_KERNELS, kernel_size=(1, 1), padding='valid'):
        x = Conv2D(n_kernels, kernel_size=kernel_size, padding=padding)(x1)
        x = BatchNormalization()(x)
        x2 = LeakyReLU()(x)
        return x2

    def down_cnn(self, num_skip: int):
        input_layer = Input(shape=(NUM_FEATURES, NUM_KERNELS, 2))
        x = self.conv_bn_lr(input_layer, kernel_size=(NUM_FEATURES, 1), padding='valid')

        for i in range(num_skip):
            x_conv = self.conv_bn_lr(x, kernel_size=(1, 3), padding='same')
            x = Add()([x, x_conv])

        x = self.conv_bn_lr(x, n_kernels=8, kernel_size=(1, 1), padding='same')
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=input_layer, outputs=x)

    def call(self, points_xy):
        expanded_feature_x = self.cnn(points_xy[..., 0])
        expanded_feature_y = self.cnn(points_xy[..., 1])
        expanded_feature = tf.stack([expanded_feature_x, expanded_feature_y], axis=-1)
        return self.cnn2(expanded_feature)


def initial_matching_fpm(fpm_model, ptrs_ref_nx3: ndarray, ptrs_tgt_mx3: ndarray, k_neighbors: int) -> ndarray:
    """
    This function compute initial matching between all pairs of points in reference and target points set.

    Parameters
    ----------
    fpm_model :
        The pretrained FPM model
    ref :
        The positions of the cells in the first volume
    tgt :
        The positions of the cells in the second volume
    k_ptrs :
        The number of neighboring points used for FPM

    Returns
    -------
    similarity_scores :
        The correspondence matrix between two point sets
    """
    knn_model_ref = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(ptrs_ref_nx3)
    knn_model_tgt = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(ptrs_tgt_mx3)

    n = ptrs_ref_nx3.shape[0]
    m = ptrs_tgt_mx3.shape[0]

    features_ref_nxkp2x4 = spherical_features_of_points(ptrs_ref_nx3, ptrs_ref_nx3, k_neighbors, knn_model_ref).astype(np.float32)
    features_tgt_mxkp2x4 = spherical_features_of_points(ptrs_tgt_mx3, ptrs_tgt_mx3, k_neighbors, knn_model_tgt).astype(np.float32)

    ref_x_flat_batch_meshgrid = np.tile(features_ref_nxkp2x4, (m, 1, 1, 1)).reshape((n * m, k_neighbors + 2, NUM_FEATURES))
    tgt_x_flat_batch_meshgrid = np.tile(features_tgt_mxkp2x4, (n, 1, 1, 1)).transpose((1, 0, 2, 3)).reshape((n * m,  k_neighbors + 2, NUM_FEATURES))

    features_ref_tgt = np.concatenate((ref_x_flat_batch_meshgrid[:,:,:,None], tgt_x_flat_batch_meshgrid[:,:,:,None]), axis=3)
    similarity_scores = fpm_model.predict(features_ref_tgt, batch_size=1024).reshape((m, n))
    return similarity_scores