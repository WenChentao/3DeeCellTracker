from multiprocessing import Pool
from typing import Tuple, List

import numpy as np
from numpy import ndarray
from scipy.special import erf
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors

outlier_detector = MinCovDet()


def compute_mahalanobis(moves):
    outlier_detector.fit(moves)
    return outlier_detector.mahalanobis(moves)


def filter_outliers(matched_pairs: List[Tuple[int, int]], coordinates_nx3_t1: ndarray, coordinates_mx3_t2: ndarray,
                    neighbors: int = 10) -> ndarray:
    """
    This function filters the pairs with significantly different movements between two sets of coordinates (t1 and t2)
    by using a Mahalanobis distance threshold to determine outliers.
    """
    matched_pairs_array = np.asarray(matched_pairs)

    # Extract the coordinates of the matched pairs from the first set
    coordinates_matched_n1x3_t1 = coordinates_nx3_t1[matched_pairs_array[:, 0], :]

    # Calculate the nearest neighbors for each of the coordinates in the matched set
    knn = NearestNeighbors(n_neighbors=neighbors + 1).fit(coordinates_matched_n1x3_t1)
    _, nearest_indices_t1_nx1 = knn.kneighbors(coordinates_nx3_t1)

    # Map the indices of nearest neighbors back to original indices of matched pairs
    nearest_indices_original_nxkp1 = matched_pairs_array[nearest_indices_t1_nx1, 0]

    # Find the corresponding matched pairs for the k nearest neighbors of each point in the first set
    n = coordinates_nx3_t1.shape[0]
    neighbors_of_pairs_nxkp1x2 = np.zeros((n, neighbors + 1, 2), dtype=np.int_)
    for i in range(n):
        neighbors_of_pairs_nxkp1x2[i, :, :] = matched_pairs_array[
                                              np.isin(matched_pairs_array[:, 0], nearest_indices_original_nxkp1[i, :]),
                                              :]

    # Extract the movement vectors of each matched pairs and their neighbors
    movements_mxnx3 = coordinates_mx3_t2[:, None, :] - coordinates_nx3_t1[None, :, :]
    neighbors_movements_nxkp1x3 = movements_mxnx3[
        neighbors_of_pairs_nxkp1x2[..., 1].ravel(), neighbors_of_pairs_nxkp1x2[..., 0].ravel()].reshape(
        (n, neighbors + 1, 3))
    movements = neighbors_movements_nxkp1x3[matched_pairs_array[:, 0], ...]

    # Compute the Mahalanobis distance for each matched pair in a parallelized manner
    with Pool() as p:
        mahalanobis_distances = p.map(compute_mahalanobis, movements)  # Quick
    # mahalanobis_distances = [outlier_detector.fit(moves).mahalanobis(moves) for moves in movements] # Slow, deprecated

    # Filter out matched pairs where the Mahalanobis distance is greater than the threshold
    updated_pairs = []
    threshold_mdist = 20 ** 2
    for (ind_t1, ind_t2), mahalanobis_distance in zip(matched_pairs, mahalanobis_distances):
        if ind_t1 in neighbors_of_pairs_nxkp1x2[ind_t1, mahalanobis_distance <= threshold_mdist, 0]:
            updated_pairs.append((ind_t1, ind_t2))
    return np.asarray(updated_pairs)


def calc_min_path(pairs: List[Tuple[int, int]], coordinates_nx3_t1, coordinates_mx3_t2):
    n = coordinates_nx3_t1.shape[0]
    m = coordinates_mx3_t2.shape[0]

    min_path_length = np.full((n, m), np.inf)

    for x_i, y_i in pairs:
        dist_to_xi = coordinates_nx3_t1 - coordinates_nx3_t1[x_i]
        dist_to_y_from_yi = coordinates_mx3_t2[y_i] - coordinates_mx3_t2
        path_length = np.linalg.norm(dist_to_xi[:, None, :] + dist_to_y_from_yi[None, :, :], axis=-1)
        mask = np.linalg.norm(dist_to_xi, axis=-1) != 0
        min_path_length = np.where(mask[:, np.newaxis], np.minimum(min_path_length, path_length), min_path_length)
    min_path_length = convert_to_rank(min_path_length)

    knn = NearestNeighbors(n_neighbors=1).fit(coordinates_nx3_t1)
    dist, _ = knn.kneighbors()
    sigma = 2
    # map the orders to a probability according to Gaussian distribution: 0->1.0, +inf->0.0
    prob_result = 1 - erf(min_path_length / (sigma * np.sqrt(2)))
    return prob_result.T


def convert_to_rank(min_path_length):
    rank_matrix = np.argsort(min_path_length, axis=1)
    rank_result = np.zeros_like(rank_matrix)
    for i in range(rank_matrix.shape[0]):
        rank_result[i, rank_matrix[i]] = np.arange(0, rank_matrix.shape[1])
    return rank_result
