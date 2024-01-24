from typing import Callable

import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment
from skimage.transform import estimate_transform
from tensorflow.python.keras import Model

from CellTracker.robust_match import calc_min_path, filter_matching_outliers_global
from CellTracker.utils import normalize_points


def pre_alignment(coords_t1: ndarray, coords_t2: ndarray, match_model: Model, predict_method: Callable, match_method: str,
                  similarity_threshold: float, ttype: str = "euclidean"):
    print("Pre-alignment by FPM + Affine Transformation")
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


def rotation_align_by_control_points(coords_norm_t1: ndarray, coords_norm_t2: ndarray, initial_matched_pairs: ndarray):
    src = coords_norm_t1[initial_matched_pairs[:, 0]]
    dst = coords_norm_t2[initial_matched_pairs[:, 1]]
    tform = estimate_transform("euclidean", src, dst)
    aligned_coords_t1 = tform(coords_norm_t1)
    return aligned_coords_t1


def coherence_match(updated_match_matrix: ndarray, segmented_coords_norm_t1, segmented_coords_norm_t2, threshold):
    matched_pairs = greedy_match(updated_match_matrix, threshold)
    for i in range(5):
        coherence = calc_min_path(matched_pairs, segmented_coords_norm_t1, segmented_coords_norm_t2)
        updated_match_matrix = np.sqrt(updated_match_matrix * coherence)
        matched_pairs = greedy_match(updated_match_matrix, threshold)
    #matched_pairs = filter_matching_outliers(matched_pairs, segmented_coords_norm_t1, segmented_coords_norm_t2, neighbors=10)
    matched_pairs = filter_matching_outliers_global(matched_pairs, segmented_coords_norm_t1, segmented_coords_norm_t2,)
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


K_POINTS = 20
