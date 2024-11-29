from typing import Tuple

import numpy as np
from numpy import ndarray

from CellTracker.fpm import initial_matching_fpm, initial_matching_fpm_local_search
from CellTracker.robust_match import filter_matching_outliers_global
from CellTracker.simple_alignment import align_by_control_points, get_match_pairs, K_POINTS, greedy_match, \
    local_align_by_control_points, N_NEIGHBOR_LOCAL_ALIGNMENT
from CellTracker.utils import normalize_points
from CellTracker.v1_modules.ffn import initial_matching_ffn


BETA, LAMBDA, MAX_ITERATION = (3.0, 3.0, 2000)


def rotation_align_by_fpm(fpm_models_rot, coords_norm_t1, coords_norm_t2, similarity_threshold=0.4, threshold_mdist=3 ** 2):
    pairs_px2 = _match_pure_fpm(coords_norm_t1, coords_norm_t2, fpm_models_rot, similarity_threshold)
    return _transform_by_control_points(coords_norm_t1, coords_norm_t2, "euclidean", threshold_mdist, pairs_px2)


def affine_align_by_fpm(fpm_models, coords_norm_t1, coords_norm_t2, similarity_threshold=0.4, threshold_mdist=5 ** 2):
    pairs_px2, similarity_scores = match_by_fpm_prgls(fpm_models, coords_norm_t1, coords_norm_t2, similarity_threshold=similarity_threshold)
    return _transform_by_control_points(coords_norm_t1, coords_norm_t2, "affine", threshold_mdist, pairs_px2)


# def local_affine_align_by_fpm(fpm_model, coords_norm_t1, coords_norm_t2, similarity_threshold=0.3, threshold_mdist=5 ** 2):
#     pairs_px2 = match_by_fpm_prgls(fpm_model, coords_norm_t1, coords_norm_t2, similarity_threshold=similarity_threshold)
#     return _transform_by_control_points(coords_norm_t1, coords_norm_t2, "affine", threshold_mdist, pairs_px2, local_cal=True)


def _transform_by_control_points(coords_norm_t1, coords_norm_t2, method_transform: str, threshold_mdist:float, pairs_px2: ndarray,
                                 local_cal: bool=False):
    align_func = local_align_by_control_points if local_cal and pairs_px2.shape[0] > N_NEIGHBOR_LOCAL_ALIGNMENT else align_by_control_points
    assert not (local_cal and pairs_px2.shape[0] <= N_NEIGHBOR_LOCAL_ALIGNMENT), f"pairs_px2.shape={pairs_px2.shape}"
    aligned_coords_t1, _ = align_func(coords_norm_t1, coords_norm_t2, pairs_px2, method_transform)
    filtered_pairs = filter_matching_outliers_global(pairs_px2, aligned_coords_t1, coords_norm_t2, threshold_mdist)
    aligned_coords_t1_updated, tform = align_func(coords_norm_t1, coords_norm_t2, filtered_pairs, method_transform)
    sorted_pairs = _sort_pairs(filtered_pairs)
    return aligned_coords_t1_updated, coords_norm_t2, sorted_pairs, tform

def _sort_pairs(pairs_px2):
    sorted_indices = np.argsort(pairs_px2[:, 0])
    return np.column_stack((pairs_px2[sorted_indices, 0], pairs_px2[sorted_indices, 1]))


# def _align_by_fpm_affine(fpm_model, coords_norm_t1, coords_norm_t2, similarity_threshold=0.4):
#     # coords_norm_t1 = normalize_points(points1)
#     # coords_norm_t2 = normalize_points(points2)
#
#     pairs_px2 = _match_pure_fpm(coords_norm_t1, coords_norm_t2, fpm_model, similarity_threshold)
#     aligned_coords_t1, _ = align_by_control_points(coords_norm_t1, coords_norm_t2, pairs_px2, method="affine")
#     return pairs_px2, (aligned_coords_t1, coords_norm_t1, coords_norm_t2)


def _match_pure_fpm(coords_norm_t1: ndarray, coords_norm_t2:ndarray, fpm_models, similarity_threshold: float, prob_mxn_initial: ndarray=None):
    "Use fpm for matching and extract the pairs"
    if np.isnan(np.max(coords_norm_t1)):
        raise ValueError("coords_norm_t1 contains Nan value")
    if np.isnan(np.max(coords_norm_t2)):
        raise ValueError("coords_norm_t2 contains Nan value")

    if prob_mxn_initial is None:
        initial_matching = initial_matching_fpm(fpm_models, coords_norm_t1, coords_norm_t2, K_POINTS)
    else:
        initial_matching = initial_matching_fpm_local_search(fpm_models[0], coords_norm_t1, coords_norm_t2, K_POINTS, prob_mxn_initial)
    matching_copy = initial_matching.copy()
    pairs_px2 = get_match_pairs(matching_copy, coords_norm_t1, coords_norm_t2,
                                threshold=similarity_threshold, method="greedy")
    return pairs_px2


def match_by_fpm_prgls(fpm_models, coords_norm_t1, coords_norm_t2, similarity_threshold=0.4, similarity_threshold_final=0.3):
    pairs_px2 = _match_pure_fpm(coords_norm_t1, coords_norm_t2, fpm_models, similarity_threshold)

    n = coords_norm_t1.shape[0]
    m = coords_norm_t2.shape[0]
    predicted_coords_t1_to_t2, similarity_scores = predict_matching_prgls(pairs_px2,
                                                                          coords_norm_t1,
                                                                          coords_norm_t1,
                                                                          coords_norm_t2,
                                                                          (m, n), beta=BETA, lambda_=LAMBDA)

    pairs_final = greedy_match(similarity_scores, threshold=similarity_threshold_final)
    return pairs_final, similarity_scores


def match_by_ffn(ffn_model, points1, points2, similarity_threshold=0.4, match_method='coherence'):
    # Initialize the model
    coords_norm_t1, _ = normalize_points(points1)
    coords_norm_t2, _ = normalize_points(points2)
    initial_matching = initial_matching_ffn(ffn_model, coords_norm_t1, coords_norm_t2,
                                            K_POINTS)
    updated_matching = initial_matching.copy()
    pairs_px2 = get_match_pairs(updated_matching, coords_norm_t1, coords_norm_t2,
                                threshold=similarity_threshold, method=match_method)
    return pairs_px2



def load_fpm(fpm_model_path, match_model):
    fpm_model = match_model
    dummy_input = np.random.random((1, 22, 4, 2))
    try:
        _ = fpm_model(dummy_input)
        fpm_model.load_weights(fpm_model_path)
    except (OSError, ValueError) as e:
        raise ValueError(f"Failed to load the match model from {fpm_model_path}: {e}") from e
    return fpm_model


def ids_to_pairs(ids_1: np.ndarray, ids_2: np.ndarray) -> np.ndarray:
    """Generate ground truth of pairs from test_tracking data in https://osf.io/t7dzu/

    Notes
    -----
    The ids can be obtained from points[:, 3] loaded from the .npy files.
    The value -1 means the cell is not tracked. Other values (>=0) means the identified numeric id.
    """
    array1, array2 = ids_1.astype(int), ids_2.astype(int),
    filtered_array1 = array1[array1 != -1]
    filtered_array2 = array2[array2 != -1]

    # 找出相同元素及其在两个数组中的位置
    common_elements, common_idx1, common_idx2 = np.intersect1d(filtered_array1, filtered_array2, return_indices=True)

    # 对 array1, 2 进行排序，并记录原始索引
    sorted_indices_1 = np.argsort(array1)
    sorted_arr1 = array1[sorted_indices_1]
    sorted_indices_2 = np.argsort(array2)
    sorted_arr2 = array2[sorted_indices_2]

    # 使用 searchsorted 找到插入位置
    insert_positions_1 = np.searchsorted(sorted_arr1, common_elements)
    insert_positions_2 = np.searchsorted(sorted_arr2, common_elements)

    # 使用原始索引获取实际位置
    original_idx1 = sorted_indices_1[insert_positions_1]
    original_idx2 = sorted_indices_2[insert_positions_2]

    # 根据reference points的编号进行排序
    sorted_indices = np.argsort(original_idx1)
    pairs_gt = np.column_stack((original_idx1[sorted_indices], original_idx2[sorted_indices]))
    return pairs_gt


def accuracy(pairs_nx3: np.ndarray, pairs_gt_mx3: np.ndarray):
    if len(pairs_gt_mx3)==0:
        print("Warning: no ground truth data!")
        return

    common_elements, common_idx1, common_idx2 = np.intersect1d(pairs_nx3[:, 0], pairs_gt_mx3[:, 0],
                                                               return_indices=True)

    pairs_col_2 = pairs_nx3[common_idx1, 1]
    pairs_gt_col_2 = pairs_gt_mx3[common_idx2, 1]
    idx = np.nonzero(pairs_col_2==pairs_gt_col_2)[0]
    return len(idx)/len(pairs_gt_mx3)


def predict_matching_prgls(matched_pairs, confirmed_coords_norm_t1, segmented_coords_norm_t1, segmented_coords_norm_t2,
                           similarity_scores_shape: Tuple[int, int], beta=BETA, lambda_=LAMBDA):
    normalized_prob = cal_norm_prob(matched_pairs, similarity_scores_shape)
    tracked_coords_norm_t2, prob_mxn = prgls_with_two_ref(normalized_prob, segmented_coords_norm_t2,
                                                   segmented_coords_norm_t1, confirmed_coords_norm_t1,
                                                   beta=beta, lambda_=lambda_)
    return tracked_coords_norm_t2, prob_mxn


def cal_norm_prob(matched_pairs, shape):
    normalized_prob = np.full(shape, 0.1 / (shape[1] - 1))
    for ref, tgt in matched_pairs:
        normalized_prob[tgt, ref] = 0.9
    return normalized_prob


def prgls_with_two_ref(normalized_prob_mxn, ptrs_tgt_mx3: ndarray, ptrs_ref_nx3: ndarray, tracked_ref_lx3: ndarray,
                       beta: float, lambda_: float, max_iteration: int = MAX_ITERATION) \
        -> Tuple[ndarray, ndarray]:
    """
    Similar with prgls_quick, but use another ptrs_ref_nx3 to calculate the basis movements, and applied the movements
    to the tracked_ref_lx3
    """

    # Initiate parameters
    ratio_outliers = 0.05  # This is the gamma
    distance_weights_nxn = gaussian_kernel(ptrs_ref_nx3, ptrs_ref_nx3, beta ** 2)  # This is the Gram matrix
    distance_weights_nxl = gaussian_kernel(tracked_ref_lx3, ptrs_ref_nx3, beta ** 2)
    sigma_square = dist_squares(ptrs_ref_nx3, ptrs_tgt_mx3).mean() / 3  # This is the sigma^2
    predicted_coord_ref_nx3 = ptrs_ref_nx3.copy()  # This is the T(X)
    predicted_coord_ref_lx3 = tracked_ref_lx3.copy()

    ############################################################################
    # iteratively update predicted_ref_n1x3, ratio_outliers, sigma_square, and posterior_mxn. Plot and save results
    ############################################################################
    for iteration in range(1, max_iteration):
        # E-step: update posterior probability P_mxn
        posterior_mxn = estimate_posterior(normalized_prob_mxn, sigma_square, predicted_coord_ref_nx3, ptrs_tgt_mx3,
                                           ratio_outliers)

        # M-step: update predicted positions of reference set
        # movements_basis_3xn is the parameter C
        movements_basis_3xn = solve_movements_ref(sigma_square, lambda_, posterior_mxn, predicted_coord_ref_nx3,
                                                  ptrs_tgt_mx3, distance_weights_nxn)
        movements_ref_nx3 = np.dot(movements_basis_3xn, distance_weights_nxn).T
        movements_tracked_lx3 = np.dot(movements_basis_3xn, distance_weights_nxl).T
        if iteration > 1:
            predicted_coord_ref_nx3 += movements_ref_nx3  # The first estimation is not reliable thus is discarded
            predicted_coord_ref_lx3 += movements_tracked_lx3
        sum_posterior = np.sum(posterior_mxn)
        ratio_outliers = 1 - sum_posterior / ptrs_tgt_mx3.shape[0]

        # Sometimes this value could become minus due to the inaccurate float representation in computer.
        # Here I fixed this bug.
        if ratio_outliers < 1E-4:
            ratio_outliers = 1E-4

        sigma_square = np.sum(dist_squares(predicted_coord_ref_nx3, ptrs_tgt_mx3) * posterior_mxn) / (3 * sum_posterior)

        # Test convergence:
        dist_sqrt = np.sqrt(np.sum(np.square(movements_ref_nx3)))
        if dist_sqrt < 1E-2:
            # print(f"Converged at iteration = {iteration}")
            break

    return predicted_coord_ref_lx3, posterior_mxn


def dist_squares(ptrs_ref_nx3: ndarray, ptrs_tgt_mx3: ndarray) -> ndarray:
    ptrs_ref_mxnx3 = np.tile(ptrs_ref_nx3, (ptrs_tgt_mx3.shape[0], 1, 1))
    ptrs_tgt_mxnx3 = np.tile(ptrs_tgt_mx3, (ptrs_ref_nx3.shape[0], 1, 1)).transpose((1, 0, 2))
    dist2_mxn = np.sum(np.square(ptrs_ref_mxnx3 - ptrs_tgt_mxnx3), axis=2)
    return dist2_mxn


def gaussian_kernel(ptrs_ref_nx3, ptrs_tgt_mx3, sigma_square: float) -> ndarray:
    ptrs_ref_mxnx3 = np.tile(ptrs_ref_nx3, (ptrs_tgt_mx3.shape[0], 1, 1))
    ptrs_tgt_mxnx3 = np.tile(ptrs_tgt_mx3, (ptrs_ref_nx3.shape[0], 1, 1)).transpose((1, 0, 2))
    dist_square_sum_mxn = np.sum(np.square(ptrs_ref_mxnx3 - ptrs_tgt_mxnx3), axis=2)
    return np.exp(-dist_square_sum_mxn / (2 * sigma_square))


def estimate_posterior(prior_p_mxn: ndarray, initial_sigma_square: float, predicted_ref_nx3: ndarray,
                       ptrs_tgt_mx3: ndarray, ratio_outliers: float, vol: float = 1) -> ndarray:
    p_pos_j_when_j_match_i_mxn = gaussian_kernel(predicted_ref_nx3, ptrs_tgt_mx3, initial_sigma_square)
    p_pos_j_and_j_match_i_mxn = (1 - ratio_outliers) * prior_p_mxn * p_pos_j_when_j_match_i_mxn / (
            2 * np.pi * initial_sigma_square) ** 1.5
    posterior_sum_m = np.sum(p_pos_j_and_j_match_i_mxn, axis=1) + ratio_outliers / vol
    posterior_mxn = p_pos_j_and_j_match_i_mxn / posterior_sum_m[:, None]
    return posterior_mxn


def solve_movements_ref(initial_sigma_square, lambda_, posterior_mxn, ptrs_ref_nx3, ptrs_tgt_mx3,
                        scaling_factors_nxn):
    n = ptrs_ref_nx3.shape[0]
    posterior_sum_diag_nxn = np.diag(np.sum(posterior_mxn, axis=0))
    coefficient_nxn = np.dot(scaling_factors_nxn,
                             posterior_sum_diag_nxn) + lambda_ * initial_sigma_square * np.identity(n)
    dependent_3xn = np.dot(ptrs_tgt_mx3.T, posterior_mxn) - np.dot(ptrs_ref_nx3.T, posterior_sum_diag_nxn)
    movements_ref_3xn = np.linalg.solve(coefficient_nxn.T, dependent_3xn.T).T
    return movements_ref_3xn
