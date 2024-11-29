from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy import ndarray
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Concatenate, Conv2D, Flatten, Add
from tifffile import imread
from tqdm import tqdm

from CellTracker.utils import normalize_points
from CellTracker.synthesize_deformation import K_NEIGHBORS, NUM_FEATURES, \
    spherical_features_of_points, BATCH_SIZE, generator_train_data

FPN_WEIGHTS_NAME = "weights_training_"
RANGE_ROT_TGT = 180
NUM_KERNELS = 128
NUM_FEATURES2 = 4


class TrainFPM:
    def __init__(self, model_name: str, match_model: Model, points1_path: str = None, segmentation1_path: str = None, voxel_size: tuple = (1, 1, 1),
                 range_rotation_tgt: float = RANGE_ROT_TGT,
                 basedir: str = "./fpm_models", replacement=False, strength=0.4):

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

        self.strength = strength

       # Load the points set
        if points1_path is not None:
            self.points_t1, _ = normalize_points(np.loadtxt(points1_path))
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
            self.points_t1, _ = normalize_points(points_t1 * np.asarray(voxel_size)[None, :])
        else:
            raise ValueError("Either the segmentation1_path or the points1_path should be provided")
        self.optimizer = tf.keras.optimizers.Adam()
        self.range_rotation_tgt=range_rotation_tgt
        self.replacement = replacement

    def train(self, num_epochs=10, iteration=5000, weights_name=FPN_WEIGHTS_NAME):
        train_loss_fn = tf.keras.losses.BinaryCrossentropy()

        dataset = tf.data.Dataset.from_generator(
            generator_train_data,
            args=(self.points_t1, self.range_rotation_tgt, self.replacement, self.strength),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape([BATCH_SIZE, K_NEIGHBORS + 2, NUM_FEATURES, 2]),
                tf.TensorShape([BATCH_SIZE, 1])
            )
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        #train_loader = self.points_generator.train_data_gen
        accuracy = tf.keras.metrics.BinaryAccuracy()

        start_epoch = self.current_epoch
        end_epoch = self.current_epoch + num_epochs
        for epoch in range(start_epoch, end_epoch):
            train_loss = 0
            train_acc = 0
            n = 0
            with tqdm(total=iteration, desc=f'Epoch {epoch}/{end_epoch - 1}', ncols=50, unit='batch') as pbar:
                for X, y in dataset:
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


class FlexiblePointMatcherConv(Model):
    """
    A class that defines a custom architecture for matching point sets of cells from two 3D images.
    """

    def __init__(self, num_skip: int):
        super().__init__()
        self.encoder = up_cnn(num_skip)
        self.comparator = self.down_cnn(num_skip)

    def down_cnn(self, num_skip: int):
        input_layer = Input(shape=(NUM_FEATURES, NUM_KERNELS, 2))
        x = conv_bn_lr(input_layer, kernel_size=(NUM_FEATURES, 1), padding='valid')

        for i in range(num_skip):
            x_conv = conv_bn_lr(x, kernel_size=(1, 3), padding='same')
            x = Add()([x, x_conv])

        x = conv_bn_lr(x, n_kernels=8, kernel_size=(1, 1), padding='same')
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=input_layer, outputs=x)

    def call(self, points_xy):
        expanded_feature_x = self.encoder(points_xy[..., 0])
        expanded_feature_y = self.encoder(points_xy[..., 1])
        expanded_feature = tf.stack([expanded_feature_x, expanded_feature_y], axis=-1)
        return self.comparator(expanded_feature)

    def predict_feature(self, points_x, batch_size: int):
        num_samples = points_x.shape[0]
        predictions = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_points_xy = points_x[start_idx:end_idx]
            predictions.append(self.encoder(batch_points_xy))

        return tf.concat(predictions, axis=0)


def up_cnn(num_skip: int):
    input_layer = Input(shape=(K_NEIGHBORS + 2, NUM_FEATURES, 1))

    x = conv_bn_lr(input_layer, kernel_size=(K_NEIGHBORS + 2, 1))

    for i in range(num_skip):
        x_conv = conv_bn_lr(x)
        x = Add()([x, x_conv])

    x = x[:, 0, :, :]
    return Model(inputs=input_layer, outputs=x)
    
    
def conv_bn_lr(x1, n_kernels=NUM_KERNELS, kernel_size=(1, 1), padding='valid'):
    x = Conv2D(n_kernels, kernel_size=kernel_size, padding=padding)(x1)
    x = BatchNormalization()(x)
    x2 = LeakyReLU()(x)
    return x2
    
    
class FlexiblePointMatcherConvSimpler(Model):
    """
    A class that defines a custom architecture for matching point sets of cells from two 3D images.
    """

    def __init__(self, num_skip: int):
        super().__init__()
        self.encoder = up_cnn(num_skip)
        self.comparator = self.down_cnn()

    def down_cnn(self):
        input_layer = Input(shape=(NUM_FEATURES, NUM_KERNELS, 2))
        x = conv_bn_lr(input_layer, kernel_size=(NUM_FEATURES, 1), padding='valid')

        x = conv_bn_lr(x, n_kernels=1, kernel_size=(1, 1), padding='same')
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=input_layer, outputs=x)

    def call(self, points_xy):
        expanded_feature_x = self.encoder(points_xy[..., 0])
        expanded_feature_y = self.encoder(points_xy[..., 1])
        expanded_feature = tf.stack([expanded_feature_x, expanded_feature_y], axis=-1)
        return self.comparator(expanded_feature)

    def predict_feature(self, points_x, batch_size: int):
        num_samples = points_x.shape[0]
        predictions = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_points_xy = points_x[start_idx:end_idx]
            predictions.append(self.encoder(batch_points_xy))

        return tf.concat(predictions, axis=0)


class FPMPart2Model(Model):
    def __init__(self, comparator):
        super(FPMPart2Model, self).__init__()
        self.comparator = comparator

    def call(self, expanded_feature):
        return self.comparator(expanded_feature)


def initial_matching_fpm_(fpm_model, ptrs_ref_nx3: ndarray, ptrs_tgt_mx3: ndarray, k_neighbors: int) -> ndarray:
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


def initial_matching_fpm(fpm_models, ptrs_ref_nx3: ndarray, ptrs_tgt_mx3: ndarray, k_neighbors: int) -> ndarray:
    """
    This function compute initial matching between all pairs of points in reference and target points set.

    Parameters
    ----------
    fpm_models :
        The pretrained FPM model and FPM_PART2
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
    fpm_model, fpm_part2 = fpm_models

    knn_model_ref = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(ptrs_ref_nx3)
    knn_model_tgt = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(ptrs_tgt_mx3)

    n = ptrs_ref_nx3.shape[0]
    m = ptrs_tgt_mx3.shape[0]

    features_ref_nxkp2x4 = spherical_features_of_points(ptrs_ref_nx3, ptrs_ref_nx3, k_neighbors, knn_model_ref).astype(np.float32)
    features_tgt_mxkp2x4 = spherical_features_of_points(ptrs_tgt_mx3, ptrs_tgt_mx3, k_neighbors, knn_model_tgt).astype(np.float32)

    expanded_feature_ref = fpm_model.predict_feature(features_ref_nxkp2x4, batch_size=1024)
    expanded_feature_tgt = fpm_model.predict_feature(features_tgt_mxkp2x4, batch_size=1024)

    feature_shape = expanded_feature_tgt.shape
    axes = [1, 0] + list(range(2, 1 + len(feature_shape)))
    expanded_feature_ref_meshgrid = tf.reshape(tf.tile(tf.expand_dims(
        expanded_feature_ref, axis=0), [m, 1, 1, 1]), [n * m, *feature_shape[1:]])
    expanded_feature_tgt_meshgrid = tf.reshape(tf.transpose(
            tf.tile(tf.expand_dims(expanded_feature_tgt, axis=0), (n, 1, 1, 1)), perm=axes), (n * m, *feature_shape[1:]))
    features_ref_tgt = tf.stack((expanded_feature_ref_meshgrid, expanded_feature_tgt_meshgrid), axis=-1)

    similarity_scores = fpm_part2.predict(features_ref_tgt, batch_size=256).reshape((m, n))

    return similarity_scores


def initial_matching_fpm_local_search(fpm_model, ptrs_ref_nx3: ndarray, ptrs_tgt_mx3: ndarray, k_neighbors: int,
                                      prob_mxn_initial: ndarray, threshold: float = 1e-6) -> ndarray:
    """
    This function search for matches from paris that with higher probabilities.

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
    prob_mxn_initial:
        The pre-computed probability-matrix between points in reference and target volumes
    threshold:
        Only the points pairs with prob_mxn_initial>threshold will be calculated by the fpm_model

    Returns
    -------
    similarity_scores :
        The correspondence matrix between two point sets
    """
    knn_model_ref = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(ptrs_ref_nx3)
    knn_model_tgt = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(ptrs_tgt_mx3)

    n = ptrs_ref_nx3.shape[0]
    m = ptrs_tgt_mx3.shape[0]

    features_ref_nxkp2x4 = spherical_features_of_points(ptrs_ref_nx3, ptrs_ref_nx3, k_neighbors, knn_model_ref).astype(
        np.float32)
    features_tgt_mxkp2x4 = spherical_features_of_points(ptrs_tgt_mx3, ptrs_tgt_mx3, k_neighbors, knn_model_tgt).astype(
        np.float32)

    ref_x_flat_batch_meshgrid = np.tile(features_ref_nxkp2x4, (m, 1, 1, 1)).reshape(
        (n * m, k_neighbors + 2, NUM_FEATURES))
    tgt_x_flat_batch_meshgrid = np.tile(features_tgt_mxkp2x4, (n, 1, 1, 1)).transpose((1, 0, 2, 3)).reshape(
        (n * m, k_neighbors + 2, NUM_FEATURES))

    features_ref_tgt = np.concatenate(
        (ref_x_flat_batch_meshgrid[:, :, :, None], tgt_x_flat_batch_meshgrid[:, :, :, None]), axis=3)

    # Find the pairs to calculate similarity
    indices_cal = np.nonzero(prob_mxn_initial > threshold)
    one_dim_indices = np.ravel_multi_index(indices_cal, prob_mxn_initial.shape)
    similarity_scores = np.zeros((m*n))

    # Calculate the similarity in selected pairs
    features_ref_tgt_selected = features_ref_tgt[one_dim_indices]
    similarity_scores[one_dim_indices] = fpm_model.predict(features_ref_tgt_selected, batch_size=256)[:, 0]
    similarity_scores = similarity_scores.reshape((m, n))

    return similarity_scores
