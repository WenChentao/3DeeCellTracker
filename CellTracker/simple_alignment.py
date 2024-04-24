from typing import Callable

import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment
from scipy.special import erf
from skimage.transform import estimate_transform
from tensorflow.python.keras import Model

from CellTracker.robust_match import robust_distance, filter_matching_outliers_global
from CellTracker.utils import normalize_points

N_NEIGHBOR_LOCAL_ALIGNMENT = 30
K_POINTS = 20


def pre_alignment(coords_t1: ndarray, coords_t2: ndarray, match_model: Model, predict_method: Callable, match_method: str,
                  similarity_threshold: float, ttype: str = "euclidean"):
    print(f"Pre-alignment by FPM + {ttype} Transformation")
    coords_norm_t2, _ = normalize_points(coords_t2, return_para=True)
    coords_norm_t1, _ = normalize_points(coords_t1, return_para=True)

    similarity_scores = predict_method(match_model, coords_norm_t1, coords_norm_t2, K_POINTS)

    initial_matched_pairs = get_match_pairs(similarity_scores, coords_norm_t1, coords_norm_t2,
                                            threshold=similarity_threshold, method=match_method)
    src = coords_norm_t1[initial_matched_pairs[:, 0]]
    dst = coords_norm_t2[initial_matched_pairs[:, 1]]
    tform = estimate_transform(ttype, src, dst) # docstring said it's for 2D points but I have confirmed it also works correctly for 3D
    affine_aligned_coords_t1 = tform(coords_norm_t1) # the affine transformation matrix (without translation) can be extracted by tform.params.T[:3,:3]
    return affine_aligned_coords_t1, coords_norm_t2


def get_transform(coords_t1: ndarray, coords_t2: ndarray, match_model: Model, predict_method: Callable, match_method: str,
                  similarity_threshold: float, ttype: str = "affine"):
    coords_norm_t2, _ = normalize_points(coords_t2, return_para=True)
    coords_norm_t1, _ = normalize_points(coords_t1, return_para=True)

    similarity_scores = predict_method(match_model, coords_norm_t1, coords_norm_t2, K_POINTS)

    initial_matched_pairs = get_match_pairs(similarity_scores, coords_norm_t1, coords_norm_t2,
                                            threshold=similarity_threshold, method=match_method)
    src = coords_norm_t1[initial_matched_pairs[:, 0]]
    dst = coords_norm_t2[initial_matched_pairs[:, 1]]
    return estimate_transform(ttype, src, dst)


def align_by_control_points(coords_norm_t1: ndarray, coords_norm_t2: ndarray, initial_matched_pairs: ndarray, method="euclidean"):
    src = coords_norm_t1[initial_matched_pairs[:, 0]]
    dst = coords_norm_t2[initial_matched_pairs[:, 1]]
    tform = estimate_transform(method, src, dst)
    aligned_coords_t1 = tform(coords_norm_t1)
    return aligned_coords_t1, tform


def local_align_by_control_points(coords_norm_t1: ndarray, coords_norm_t2: ndarray, initial_matched_pairs: ndarray,
                                  method="euclidean", n_neighbors: int = N_NEIGHBOR_LOCAL_ALIGNMENT):
    src = coords_norm_t1[initial_matched_pairs[:, 0]]
    dst = coords_norm_t2[initial_matched_pairs[:, 1]]
    aligned_coords_t1 = np.zeros_like(coords_norm_t1)
    from sklearn.neighbors import NearestNeighbors

    knn_model = NearestNeighbors(n_neighbors=n_neighbors).fit(src)
    neighbors_inds = knn_model.kneighbors(coords_norm_t1, return_distance=False)
    for i, inds in enumerate(neighbors_inds):
        _src = src[inds, :]
        _dst = dst[inds, :]
        tform = estimate_transform(method, _src, _dst)
        aligned_coords_t1[i, :] = tform(coords_norm_t1[i:i+1,:])[0, :]
    return aligned_coords_t1, None


def coherence_match(updated_match_matrix: ndarray, segmented_coords_norm_t1, segmented_coords_norm_t2, threshold):
    matched_pairs = greedy_match(updated_match_matrix, threshold)
    for i in range(3):
        distances_matrix = robust_distance(matched_pairs, segmented_coords_norm_t1, segmented_coords_norm_t2)
        # map the ranks to a probability: rank = 0 -> 1.0, rank = +inf -> 0.0
        coherence = 1 - erf(distances_matrix / 3).T
        updated_match_matrix = np.sqrt(updated_match_matrix * coherence)
        matched_pairs = greedy_match(updated_match_matrix, threshold)
    return matched_pairs


def hungarian_match(match_score_matrix_: ndarray, match_score_matrix_updated: ndarray, similarity_threshold: float):
    row_indices, col_indices = linear_sum_assignment(match_score_matrix_updated, maximize=True)
    match_pairs = []
    for r, c in zip(row_indices, col_indices):
        if match_score_matrix_[r, c] > similarity_threshold and match_score_matrix_updated[r, c] > similarity_threshold:
            match_pairs.append((c, r))
    return np.asarray(match_pairs)


def greedy_match(updated_match_matrix: ndarray, threshold: float = 1e-6):
    """Return greedy match, and the updated matching matrix with matched rows, cols filled with -1"""
    working_match_score_matrix = updated_match_matrix.copy()
    match_pairs = []
    for pair_number in range(working_match_score_matrix.shape[1]):
        max_match_score = working_match_score_matrix.max()
        if max_match_score < threshold:
            break
        target_index, reference_index = np.unravel_index(working_match_score_matrix.argmax(),
                                                         working_match_score_matrix.shape)
        match_pairs.append((reference_index, target_index))

        working_match_score_matrix[target_index, :] = -1
        working_match_score_matrix[:, reference_index] = -1
    return np.asarray(match_pairs)


def get_match_pairs(updated_match_matrix: ndarray, segmented_coords_norm_t1, segmented_coords_norm_t2, threshold=0.5,
                    method="coherence") -> ndarray:
    """Match points from two point sets by simply choosing the pairs with the highest probability subsequently"""
    if method == "greedy":
        return greedy_match(updated_match_matrix, threshold)
    if method == "hungarian":
        return hungarian_match(updated_match_matrix, updated_match_matrix, threshold)
    if method == "coherence":
        return coherence_match(updated_match_matrix, segmented_coords_norm_t1, segmented_coords_norm_t2, threshold)
    raise ValueError("method should be 'greedy', 'hungarian' or 'coherence'")

