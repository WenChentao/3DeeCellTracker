from multiprocessing import Pool
from typing import Tuple, List

import numpy as np
import pandas as pd
import scipy.ndimage
from numpy import ndarray
from scipy.special import erf
from skimage import measure
from skimage.segmentation import relabel_sequential
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors

outlier_detector = MinCovDet(support_fraction=0.9)


def get_full_cell_candidates(coordinates_nx3: ndarray, prob_map_3d: ndarray, threshold: float = 0.3) -> ndarray:
    """
    This function returns the extra coordinates are the coordinates of the tiny cells in the prob_map_3d that are not
    included in the coordinates_nx3.
    """
    # Segment of the disconnected cell regions
    labels_from_prob_map = measure.label(prob_map_3d > threshold , connectivity=1)
    # Remove the labels if they contain the coordinates_nx3
    for i, j, k in coordinates_nx3:
        if labels_from_prob_map[i, j, k] != 0:
            labels_from_prob_map[labels_from_prob_map == labels_from_prob_map[i, j, k]] = 0
    # Relabel the remaining labels sequentially
    labels_from_prob_map, _, _ = relabel_sequential(labels_from_prob_map)
    coordinates_tiny = scipy.ndimage.center_of_mass(labels_from_prob_map, labels_from_prob_map, range(1, np.max(labels_from_prob_map) + 1))
    return np.asarray(coordinates_tiny)


def compute_mahalanobis(moves: ndarray) -> ndarray:
    """
    Compute the Mahalanobis distances for the given moves.

    Parameters:
        moves (numpy.ndarray): An array of shape (n_samples, n_features) representing the moves data.

    Returns:
        numpy.ndarray: An array of shape (n_samples,) containing the Mahalanobis distances.
    """
    try:
        outlier_detector.fit(moves)
        return outlier_detector.mahalanobis(moves)
    except ValueError:
        # Temporally set the support fraction to 1.0 to avoid singular matrix and retry fitting, then reset it back
        support_fraction = outlier_detector.support_fraction
        try:
            outlier_detector.support_fraction = 1.0
            outlier_detector.fit(moves)
            return outlier_detector.mahalanobis(moves)
        except ValueError:
            # If the matrix is still singular, return mahalanobis distance as 0
            return np.zeros_like(moves[:, 0])
        finally:
            outlier_detector.support_fraction = support_fraction


def filter_matching_outliers(matched_pairs: List[Tuple[int, int]], coordinates_nx3_t1: ndarray, coordinates_mx3_t2: ndarray,
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
    _, nearest_indices_t1_nxkp1 = knn.kneighbors(coordinates_nx3_t1)

    # Map the indices of nearest neighbors back to original indices of matched pairs
    nearest_indices_original_nxkp1 = matched_pairs_array[nearest_indices_t1_nxkp1, 0]

    # Find the corresponding matched pairs for the k nearest neighbors of each point in the first set
    n = coordinates_nx3_t1.shape[0]
    neighbors_of_pairs_nxkp1x2 = np.zeros((n, neighbors + 1, 2), dtype=np.int_)

    df_matched_pairs = pd.DataFrame(matched_pairs_array, columns=['key', 'value'])
    dict_matched_pairs = df_matched_pairs.set_index('key')['value'].to_dict()

    vfunc = np.vectorize(dict_matched_pairs.get)
    for i in range(n):
        neighbors_of_pairs_nxkp1x2[i, :, 0] = nearest_indices_original_nxkp1[i, :]
        neighbors_of_pairs_nxkp1x2[i, :, 1] = vfunc(nearest_indices_original_nxkp1[i, :])

    # Extract the movement vectors of each matched pairs and their neighbors
    movements_mxnx3 = coordinates_mx3_t2[:, None, :] - coordinates_nx3_t1[None, :, :]
    neighbors_movements_nxkp1x3 = movements_mxnx3[
        neighbors_of_pairs_nxkp1x2[..., 1].ravel(), neighbors_of_pairs_nxkp1x2[..., 0].ravel()].reshape(
        (n, neighbors + 1, 3))
    movements = neighbors_movements_nxkp1x3[matched_pairs_array[:, 0], ...]

    # Compute the Mahalanobis distance for each matched pair in a parallelized manner
    with Pool() as p:
        mahalanobis_distances = p.map(compute_mahalanobis, movements)  # Quick

    #print(mahalanobis_distances)

    # mahalanobis_distances = [outlier_detector.fit(moves).mahalanobis(moves) for moves in movements] # Slow, deprecated

    # Filter out matched pairs where the Mahalanobis distance is greater than the threshold
    updated_pairs = []
    threshold_mdist = 3 ** 2
    for (ind_t1, ind_t2), mahalanobis_distance in zip(matched_pairs, mahalanobis_distances):
        if mahalanobis_distance[0] <= threshold_mdist:
            updated_pairs.append((ind_t1, ind_t2))
    return np.asarray(updated_pairs)


def filter_matching_outliers_global(matched_pairs: List[Tuple[int, int]], coordinates_nx3_t1: ndarray, coordinates_mx3_t2: ndarray) -> ndarray:
    """
    This function filters the pairs with significantly different movements between two sets of coordinates (t1 and t2)
    by using a Mahalanobis distance threshold to determine outliers.
    """
    matched_pairs_array = np.asarray(matched_pairs)

    # Extract the movement vectors of each matched pairs and their neighbors
    movements_px3 = coordinates_mx3_t2[matched_pairs_array[:, 1], :] - coordinates_nx3_t1[matched_pairs_array[:, 0], :]
    mahalanobis_distances = outlier_detector.fit(movements_px3).mahalanobis(movements_px3)
    # Filter out matched pairs where the Mahalanobis distance is greater than the threshold
    updated_pairs = []
    threshold_mdist = 5 ** 2
    for (ind_t1, ind_t2), mahalanobis_distance in zip(matched_pairs, mahalanobis_distances):
        if mahalanobis_distance <= threshold_mdist:
            updated_pairs.append((ind_t1, ind_t2))
        # else:
        #     print(ind_t1, ind_t2, mahalanobis_distance)
    return np.asarray(updated_pairs)


def add_or_remove_points(predicted_coords_t1_to_t2: ndarray, predicted_coords_t2_to_t1: ndarray,
                         segmented_coords_norm_t1: ndarray, segmented_coords_norm_t2: ndarray,
                         matched_pairs: List[Tuple[int, int]]) -> \
        Tuple[ndarray, ndarray, Tuple[ndarray, ndarray]]:
    n, m = segmented_coords_norm_t1.shape[0], segmented_coords_norm_t2.shape[0]
    pairs = np.asarray(matched_pairs)
    unmatched_indice_t1 = np.setdiff1d(np.arange(n), pairs[:, 0])
    unmatched_indice_t2 = np.setdiff1d(np.arange(m), pairs[:, 1])

    # inliers_t1 = add_inliers_within_k_neighbors(k_neighbors, predicted_coords_t1_to_t2, segmented_coords_norm_t2, unmatched_indice_t2,
    #                          unmatched_indice_t1)
    # inliers_t2 = add_inliers_within_k_neighbors(k_neighbors, predicted_coords_t2_to_t1, segmented_coords_norm_t1, unmatched_indice_t1,
    #                          unmatched_indice_t2)
    inliers_t1 = add_inliers_within_a_radius(predicted_coords_t1_to_t2, segmented_coords_norm_t2,
                                             unmatched_indice_t1)
    inliers_t2 = add_inliers_within_a_radius(predicted_coords_t2_to_t1, segmented_coords_norm_t1,
                                             unmatched_indice_t2)

    all_inliers_t1 = np.concatenate((pairs[:, 0], inliers_t1)).astype(np.int_)
    all_inliers_t2 = np.concatenate((pairs[:, 1], inliers_t2)).astype(np.int_)
    inliers = (all_inliers_t1, all_inliers_t2)
    return predicted_coords_t1_to_t2[all_inliers_t1], segmented_coords_norm_t2[all_inliers_t2], inliers


def add_or_remove_points_with_prob_matrix(prob_mxn: ndarray, predicted_coords_t1_to_t2: ndarray,
                                          segmented_coords_norm_t2: ndarray):
    """
    Use a prob_mxn after applying prgls+greedy method to decide inliers
    """
    prob_t1, prob_t2 = prob_mxn.max(axis=0), prob_mxn.max(axis=1)
    all_inliers_t1 = np.nonzero(prob_t1 < -.5)[0]
    all_inliers_t2 = np.nonzero(prob_t2 < -.5)[0]
    inliers = (all_inliers_t1, all_inliers_t2)
    return predicted_coords_t1_to_t2[all_inliers_t1], segmented_coords_norm_t2[all_inliers_t2], inliers


def add_inliers_within_a_radius(predicted_coords_t1_to_t2, segmented_coords_norm_t2,
                                unmatched_indice_t1):
    """
    Add some points previously in the unmatched set to the inlier set using a dynamic radius criterion.

    Notes
    -----
    Suppose there is an unmatched point A at t2 and its prediction B at t1.
    Calculate the distance dist_A between A and its nearest neighbor at t2
    If B has its nearest neighbor point C within 1.2 * dist_A at t1, then A is added to the inlier set.
    Else, A is not added to the inlier set.
    """
    if unmatched_indice_t1.size == 0:
        return []
    knn_t1 = NearestNeighbors(n_neighbors=2).fit(predicted_coords_t1_to_t2)
    _, ids = knn_t1.kneighbors(segmented_coords_norm_t2)
    inliers_t2 = np.intersect1d(ids, unmatched_indice_t1)
    #print(inliers_t2+1)
    # for i in range(len(segmented_coords_norm_t2)):
    #     print(i+1, ids[i]+1)

    return np.asarray(inliers_t2)


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
