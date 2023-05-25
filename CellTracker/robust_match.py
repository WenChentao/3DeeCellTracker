from typing import Tuple, List, Dict

import numpy as np
from numpy import ndarray
from sklearn.covariance import EllipticEnvelope


def find_robust_matches(matched_pairs: List[Tuple[int, int]], pairwise_differences_matrix: ndarray, reference_coords: ndarray):
    pair_distances = calculate_pair_differences(pairwise_differences_matrix, matched_pairs)
    filtered_pairs = exclude_distance_outliers(matched_pairs, pair_distances, reference_coords, outlier_ratio=0.03)
    return filtered_pairs
    # return estimate_coords_with_loess(reference_coords, filtered_pairs)


def estimate_coords_with_loess(reference_coords: ndarray, filtered_pairs: List[Tuple[int, int]]) -> ndarray:
    pass


def estimate_coords_with_knn_interpolation(reference_coords: ndarray, filtered_pairs: List[Tuple[int, int]]) -> ndarray:
    pass


def calculate_pair_differences(pairwise_differences_matrix: ndarray, matched_pairs: List[Tuple[int, int]]) -> List[float]:
    pair_differences = []
    for id_t1, id_t2 in matched_pairs:
        pair_differences.append(pairwise_differences_matrix[id_t2, id_t1, :])
    return pair_differences


def exclude_distance_outliers(matched_pairs: List[Tuple[int, int]], pair_differences: List[float], reference_coords, outlier_ratio=0.1) -> List[Tuple[int, int]]:
    elliptic_env = EllipticEnvelope(contamination=outlier_ratio)
    pair_differences = np.asarray(pair_differences)

    # Fit the model to your data
    elliptic_env.fit(pair_differences)

    # The predict function returns 1 for an inlier
    labels = elliptic_env.predict(pair_differences)
    inlier_indices = np.where(labels == 1)[0]

    inlier_pairs = [matched_pairs[i] for i in inlier_indices]
    return inlier_pairs


def find_greedy_matches(match_score_matrix_: ndarray, similarity_threshold: float) -> ndarray:
    working_match_score_matrix = match_score_matrix_.copy()
    match_pairs = []
    for pair_number in range(working_match_score_matrix.shape[1]):
        max_match_score = working_match_score_matrix.max()
        if max_match_score <= similarity_threshold:
            break
        target_index, reference_index = np.unravel_index(working_match_score_matrix.argmax(), working_match_score_matrix.shape)
        match_pairs.append((reference_index, target_index))

        working_match_score_matrix[target_index, :] = 0
        working_match_score_matrix[:, reference_index] = 0
    return np.asarray(match_pairs)