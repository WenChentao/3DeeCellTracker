from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


def process_single_z_slice(z, segmentation_xyz, cell_overlaps_mask, sampling_xy):
    # 提取单个z切片的处理逻辑
    mask_image = np.logical_or(segmentation_xyz[:, :, z] > 0, cell_overlaps_mask[:, :, z] > 1)

    markers = segmentation_xyz[:, :, z].copy()
    markers[cell_overlaps_mask[:, :, z] > 1] = 0

    distance_map = distance_transform_edt(cell_overlaps_mask[:, :, z] > 1, sampling=sampling_xy)

    return watershed(distance_map, markers, mask=mask_image)


def _recalculate_cell_boundaries_joblib(segmentation_xyz: np.ndarray, cell_overlaps_mask: np.ndarray,
                                        sampling_xy: tuple = (1, 1), n_jobs=-1):
        """
        (Deprecated / runtime: 70% of _recalculate_cell_boundaries) 使用Joblib并行处理z切片

        Parameters:
        - n_jobs: 并行进程数，-1表示使用所有可用CPU核心
        """
        from joblib import Parallel, delayed
        recalculated_labels = np.zeros(segmentation_xyz.shape, dtype='int')

        # 并行处理每个z切片
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_z_slice)(
                z, segmentation_xyz, cell_overlaps_mask, sampling_xy
            ) for z in range(segmentation_xyz.shape[2])
        )

        # 将结果填入数组
        for z, result in enumerate(results):
            recalculated_labels[:, :, z] = result

        return recalculated_labels


def _recalculate_cell_boundaries_threadpool(segmentation_xyz, cell_overlaps_mask, sampling_xy=(1, 1), max_workers=None):
    # runtime: 40% of _recalculate_cell_boundaries
    recalculated_labels = np.zeros(segmentation_xyz.shape, dtype='int')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(
                process_single_z_slice,
                z, segmentation_xyz, cell_overlaps_mask, sampling_xy
            )
            for z in range(segmentation_xyz.shape[2])
        ]

        # 收集结果
        for z, future in zip(range(segmentation_xyz.shape[2]), futures):
            recalculated_labels[:, :, z] = future.result()

    return recalculated_labels


def tf_center_of_mass(prob_img, labels, label_range):
    """
    Compute centers of mass using TensorFlow for GPU acceleration.

    Parameters
    ----------
    prob_img : ndarray
        Probability image as a NumPy array.
    labels : ndarray
        Label image as a NumPy array.
    label_range : iterable
        Range of labels to compute centers of mass for.

    Returns
    -------
    centers_of_mass : ndarray
        Array of centers of mass for each label.
    """
    # Convert inputs to TensorFlow tensors
    import tensorflow as tf
    prob_img_tf = tf.convert_to_tensor(prob_img, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Flatten arrays
    prob_flat = tf.reshape(prob_img_tf, [-1])
    labels_flat = tf.reshape(labels_tf, [-1])

    # Create coordinate grids
    shape = prob_img_tf.shape
    coords = tf.stack(tf.meshgrid(
        tf.range(shape[0], dtype=tf.float32),
        tf.range(shape[1], dtype=tf.float32),
        tf.range(shape[2], dtype=tf.float32),
        indexing='ij'), axis=-1)
    coords_flat = tf.reshape(coords, [-1, 3])  # Flatten to (N, 3)

    # Compute weighted coordinate sums for each label
    weighted_coords = tf.expand_dims(prob_flat, -1) * coords_flat
    sum_weighted_coords = tf.math.unsorted_segment_sum(
        weighted_coords, labels_flat, num_segments=len(label_range) + 1)

    # Compute total weights for each label
    sum_weights = tf.math.unsorted_segment_sum(
        prob_flat, labels_flat, num_segments=len(label_range) + 1)

    # Avoid division by zero
    sum_weights = tf.where(sum_weights > 0, sum_weights, tf.ones_like(sum_weights))

    # Compute centers of mass
    centers_of_mass = sum_weighted_coords / tf.expand_dims(sum_weights, -1)

    # Extract only valid labels
    centers_of_mass = centers_of_mass[1:len(label_range) + 1]  # Skip background (label 0)

    return centers_of_mass.numpy()

def scipy_center_of_mass(prob_img, labels, label_range):
    """Deprecated, slow"""
    from scipy.ndimage import center_of_mass
    positions_of_new_centers = center_of_mass(prob_img, labels, label_range)
    positions_of_new_centers = np.asarray(positions_of_new_centers)
    return positions_of_new_centers
