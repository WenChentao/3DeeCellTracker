from typing import Tuple, Optional, List, Union, Generator, Callable

import numpy as np
from numpy import ndarray
from sklearn.neighbors import KernelDensity, NearestNeighbors

RATIO_SEG_ERROR = 0.15
K_NEIGHBORS = 20  # number of neighbors for calculating relative coordinates


def points_to_features(x_2nxk: ndarray, y_2nx1: ndarray, points_raw_nx3: ndarray, points_wi_errors_nx3: ndarray,
                       replaced_indexes: ndarray, method_features: Callable, num_features: int, k_ptrs: int = K_NEIGHBORS):
    knn_model_raw = NearestNeighbors(n_neighbors=k_ptrs + 1).fit(points_raw_nx3)
    knn_model_generated = NearestNeighbors(n_neighbors=k_ptrs + 1).fit(points_wi_errors_nx3)
    n = points_raw_nx3.shape[0]

    points_no_match_nx3 = no_match_points(n, points_wi_errors_nx3)

    x_a_nxf = method_features(points_raw_nx3, points_raw_nx3, k_ptrs, num_features, knn_model_raw)
    x_b_match_nxf = method_features(points_wi_errors_nx3, points_wi_errors_nx3, k_ptrs, num_features,
                                    knn_model_generated)
    x_b_no_match_nxf = method_features(points_wi_errors_nx3, points_no_match_nx3, k_ptrs, num_features,
                                       knn_model_generated)

    features_a = np.vstack((x_a_nxf, x_a_nxf))
    features_b = np.vstack((x_b_match_nxf, x_b_no_match_nxf))

    if np.random.rand() > 0.5:
        features_a, features_b = features_b, features_a

    x_2nxk[:, :num_features] = features_a
    x_2nxk[:, num_features:] = features_b

    y_2nx1[:n] = True
    y_2nx1[:n][replaced_indexes] = False
    y_2nx1[n:] = False


def no_match_points(n, points_wi_errors_nx3):
    random_indexes = np.arange(n)
    np.random.shuffle(random_indexes)
    points_no_match_nx3 = np.zeros_like(points_wi_errors_nx3)
    for i in range(n):
        if random_indexes[i] == i:
            no_match_index = random_indexes[i - 1]
        else:
            no_match_index = random_indexes[i]
        points_no_match_nx3[i, :] = points_wi_errors_nx3[no_match_index, :]
    return points_no_match_nx3


def add_seg_errors(points_normalized_nx3: ndarray, ratio: float = RATIO_SEG_ERROR, bandwidth: float = 0.1) -> Tuple[
    ndarray, ndarray]:
    if ratio <= 0 or ratio >= 1:
        raise ValueError(f"ratio should be set between 0 and 1 but = {ratio}")

    new_points_nx3 = points_normalized_nx3.copy()

    kde_model = KernelDensity(bandwidth=bandwidth)
    kde_model.fit(points_normalized_nx3)

    num_points = points_normalized_nx3.shape[0]
    num_replaced_points = int(np.ceil(num_points * ratio))

    points_indexes = np.arange(num_points)
    np.random.shuffle(points_indexes)

    replaced_indexes = points_indexes[:num_replaced_points]

    new_points_nx3[replaced_indexes, :] = kde_model.sample(num_replaced_points)

    return new_points_nx3, replaced_indexes


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

