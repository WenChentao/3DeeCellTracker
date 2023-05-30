from typing import Tuple, List, Dict

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.interpolate import griddata
from scipy.optimize import linear_sum_assignment
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVR
from statsmodels.nonparametric.smoothers_lowess import lowess


def find_robust_matches(matched_pairs: List[Tuple[int, int]], pairwise_differences_matrix: ndarray, reference_coords: ndarray):
    pair_distances = calculate_pair_differences(pairwise_differences_matrix, matched_pairs)
    filtered_pairs = exclude_distance_outliers(matched_pairs, pair_distances, reference_coords, outlier_ratio=0.01)
    return filtered_pairs
    # return estimate_coords_with_loess(reference_coords, filtered_pairs)


def cal_coherence(matched_pairs: List[Tuple[int, int]], coordinates_nx3_t1: ndarray, coordinates_mx3_t2: ndarray, neighbors=2) -> ndarray:
    """
    This function calculates the coherence matrix between coordinates from two points in time.
    """
    # Extract k neighbors of each point in the t1 set
    matched_pairs_array = np.asarray(matched_pairs)
    coordinates_matched_n1x3_t1 = coordinates_nx3_t1[matched_pairs_array[:, 0], :]
    knn = NearestNeighbors(n_neighbors=neighbors + 1).fit(coordinates_matched_n1x3_t1)
    _, nearest_indices_t1_nx1 = knn.kneighbors(coordinates_nx3_t1)
    nearest_indices_original_nxkp1 = matched_pairs_array[nearest_indices_t1_nx1, 0]
    nearest_indices_nxk = nearest_indices_original_nxkp1[:, 1:].copy()

    # Calculate the movements from each point in t1 set to t2 set
    movements_mxnx3 = coordinates_mx3_t2[:, None, :] - coordinates_nx3_t1[None, :, :]

    # Calculate the movements from k neighbors of each point in t1 to its predicted target point in t2
    n = coordinates_nx3_t1.shape[0]
    neighbor_is_not_self = nearest_indices_original_nxkp1[:, 0] != np.arange(n)
    nearest_indices_nxk[neighbor_is_not_self] = nearest_indices_original_nxkp1[neighbor_is_not_self, :neighbors]
    neighbors_of_pairs_nxkx2 = np.zeros((n, neighbors, 2),dtype=np.int_)
    for i in range(n):
        neighbors_of_pairs_nxkx2[i, :, :] = matched_pairs_array[np.isin(matched_pairs_array[:, 0], nearest_indices_nxk[i, :]),:]
    neighbors_movements_kxnx3 = movements_mxnx3[neighbors_of_pairs_nxkx2[...,1].ravel(), neighbors_of_pairs_nxkx2[...,0].ravel()].reshape((n, neighbors, 3)).transpose(1,0,2)

    # Calculate the coherence as the difference between a movement from point i in t1 to point j in t2 and the predicted movements of the neighboring points of point i.
    coherence_mxn = np.mean(np.sum(np.square(movements_mxnx3[None, :, :, :] - neighbors_movements_kxnx3[:, None, :, :]), axis=-1), axis=0)
    coherence_mxn =  np.min(coherence_mxn, axis=0)[None, :] / (coherence_mxn + 1e-6)
    return coherence_mxn**4


def estimate_coords_with_rf(segmented_coords_t1: ndarray, confirmed_coords_t1: ndarray,
                                           filtered_pairs: List[Tuple[int, int]],
                                           pairwise_differences_matrix: ndarray) -> ndarray:
    # svr = GridSearchCV(
    #     SVR(kernel="rbf", gamma=0.1),
    #     param_grid={"C": [1e-1, 1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-10, -9, 5)},
    # )

    ref_indices = np.asarray(filtered_pairs)[:, 0]
    y_differences = calculate_pair_differences(pairwise_differences_matrix, filtered_pairs)
    y_differences = np.asarray(y_differences)
    x_ref_coords = segmented_coords_t1[ref_indices, :]
    assert y_differences.shape[1] == 3 and x_ref_coords.shape[1] == 3
    pred_differences = np.zeros_like(confirmed_coords_t1)

    resolution = 100  # Number of points in the height map grid
    x, y = confirmed_coords_t1[:,0], confirmed_coords_t1[:,1]
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi, yi = np.meshgrid(xi, yi)

    rf = RandomForestRegressor()
    # rf.fit(x_ref_coords[:, :2], y_differences)
    # pred_differences = rf.predict(confirmed_coords_t1[:, :2])
    for dim in range(3):
        rf.fit(x_ref_coords, y_differences[:, dim])
        pred_differences[:, dim] = rf.predict(confirmed_coords_t1)
        plt.figure(figsize=(10,5))
        #plt.scatter(x_ref_coords[:, 0], y_differences[:, dim])
        #plt.plot(confirmed_coords_t1[:, 0], pred_differences[:, dim])


        # Perform interpolation to estimate the height values on the grid
        zi = griddata((x, y), pred_differences[:, dim], (xi, yi), method='nearest')

        # Plot the height map
        plt.pcolormesh(xi, yi, zi, shading='auto', cmap='terrain')

    return confirmed_coords_t1 + pred_differences


def estimate_coords_with_knn_interpolation(segmented_coords_t1: ndarray, confirmed_coords_t1: ndarray,
                                           filtered_pairs: List[Tuple[int, int]],
                                           pairwise_differences_matrix: ndarray) -> ndarray:
    def squared_inverse_distance(distances):
        # Avoid division by zero
        eps = 1e-5
        weights = 1.0 / (distances + eps) ** 2
        return weights


    ref_indices = np.asarray(filtered_pairs)[:, 0]
    y_differences = calculate_pair_differences(pairwise_differences_matrix, filtered_pairs)
    y_differences = np.asarray(y_differences)
    x_ref_coords = segmented_coords_t1[ref_indices, :]
    assert y_differences.shape[1] == 3 and x_ref_coords.shape[1] == 3
    knn = KNeighborsRegressor(n_neighbors=3, weights=squared_inverse_distance)
    knn.fit(x_ref_coords, y_differences)
    pred_differences = knn.predict(confirmed_coords_t1)
    return confirmed_coords_t1 + pred_differences


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
        if max_match_score < similarity_threshold:
            break
        target_index, reference_index = np.unravel_index(working_match_score_matrix.argmax(), working_match_score_matrix.shape)
        match_pairs.append((reference_index, target_index))

        working_match_score_matrix[target_index, :] = -1
        working_match_score_matrix[:, reference_index] = -1
    return np.asarray(match_pairs)


def hungarian_match(match_score_matrix_: ndarray, similarity_threshold: float):
    row_indices, col_indices = linear_sum_assignment(match_score_matrix_, maximize=True)
    match_pairs = []
    for r, c in zip(row_indices, col_indices):
        if match_score_matrix_[r, c] > similarity_threshold:
            match_pairs.append((c, r))
    return np.asarray(match_pairs)

