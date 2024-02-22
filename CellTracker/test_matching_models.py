import numpy as np

from CellTracker.fpm import initial_matching_fpm
from CellTracker.match_ids import predict_matching_prgls
from CellTracker.robust_match import filter_matching_outliers_global
from CellTracker.simple_alignment import align_by_control_points, get_match_pairs, K_POINTS, greedy_match, \
    local_align_by_control_points, N_NEIGHBOR_LOCAL_ALIGNMENT
from CellTracker.trackerlite import BETA, LAMBDA
from CellTracker.utils import normalize_points
from CellTracker.v1_modules.ffn import initial_matching_ffn


def rotation_align_by_fpm(fpm_model_rot, points1, points2, similarity_threshold=0.4, threshold_mdist=3**2):
    return _align_by_fpm(fpm_model_rot, points1, points2, "euclidean", threshold_mdist, similarity_threshold=similarity_threshold,
                         use_prgls=False)


def affine_align_by_fpm(fpm_model, points1, points2, similarity_threshold=0.4, threshold_mdist=5 ** 2):
    return _align_by_fpm(fpm_model, points1, points2, "affine", threshold_mdist, similarity_threshold=similarity_threshold)


def local_affine_align_by_fpm(fpm_model, points1, points2, similarity_threshold=0.3, threshold_mdist=5 ** 2):
    return _align_by_fpm(fpm_model, points1, points2, "affine", threshold_mdist, local_cal=True,
                         similarity_threshold=similarity_threshold)


def _align_by_fpm(fpm_model_rot, points1, points2, method_transform: str, threshold_mdist:float, similarity_threshold=0.4,
                  local_cal: bool=False, use_prgls=True):
    coords_norm_t1 = normalize_points(points1)
    coords_norm_t2 = normalize_points(points2)

    if use_prgls:
        pairs_px2 = match_by_fpm_prgls(fpm_model_rot, coords_norm_t1, coords_norm_t2, similarity_threshold=similarity_threshold)
    else:
        pairs_px2 = _match_fpm(coords_norm_t1, coords_norm_t2, fpm_model_rot, "greedy", similarity_threshold)

    align_func = local_align_by_control_points if local_cal and pairs_px2.shape[0] > N_NEIGHBOR_LOCAL_ALIGNMENT else align_by_control_points
    assert not (local_cal and pairs_px2.shape[0] <= N_NEIGHBOR_LOCAL_ALIGNMENT), f"pairs_px2.shape={pairs_px2.shape}"
    aligned_coords_t1 = align_func(coords_norm_t1, coords_norm_t2, pairs_px2, method_transform)
    filtered_pairs = filter_matching_outliers_global(pairs_px2, aligned_coords_t1, coords_norm_t2, threshold_mdist)
    aligned_coords_t1_updated = align_func(coords_norm_t1, coords_norm_t2, filtered_pairs, method_transform)
    sorted_pairs = _sort_pairs(filtered_pairs)

    return aligned_coords_t1_updated, coords_norm_t2, sorted_pairs

def _sort_pairs(pairs_px2):
    sorted_indices = np.argsort(pairs_px2[:, 0])
    return np.column_stack((pairs_px2[sorted_indices, 0], pairs_px2[sorted_indices, 1]))


def _align_by_fpm_simple(fpm_model, points1, points2, similarity_threshold=0.4, match_method='coherence'):
    # Initialize the model
    coords_norm_t1 = normalize_points(points1)
    coords_norm_t2 = normalize_points(points2)

    pairs_px2 = _match_fpm(coords_norm_t1, coords_norm_t2, fpm_model, 'coherence', similarity_threshold)
    aligned_coords_t1 = align_by_control_points(coords_norm_t1, coords_norm_t2, pairs_px2, method="affine")
    pairs_px2 = _match_fpm(aligned_coords_t1, coords_norm_t2, fpm_model, match_method, similarity_threshold)
    return pairs_px2, (aligned_coords_t1, coords_norm_t1, coords_norm_t2)


def _match_fpm(coords_norm_t1, coords_norm_t2, fpm_model, match_method, similarity_threshold):
    initial_matching = initial_matching_fpm(fpm_model, coords_norm_t1, coords_norm_t2,
                                            K_POINTS)
    matching_copy = initial_matching.copy()
    pairs_px2 = get_match_pairs(matching_copy, coords_norm_t1, coords_norm_t2,
                                threshold=similarity_threshold, method=match_method)
    return pairs_px2


def match_by_fpm_prgls(fpm_model, points1, points2, similarity_threshold=0.4, similarity_threshold_final=1E-6, match_method='coherence'):
    pairs_px2, (aligned_coords_t1, coords_norm_t1, coords_norm_t2) = \
        _align_by_fpm_simple(fpm_model, points1, points2, similarity_threshold, match_method)

    n = aligned_coords_t1.shape[0]
    m = points2.shape[0]
    predicted_coords_t1_to_t2, similarity_scores = predict_matching_prgls(pairs_px2,
                                                                          aligned_coords_t1,
                                                                          aligned_coords_t1,
                                                                          coords_norm_t2,
                                                                          (m, n), beta=BETA, lambda_=LAMBDA)

    match_seg_t1_seg_t2 = greedy_match(similarity_scores, threshold=similarity_threshold_final)
    return match_seg_t1_seg_t2


def match_by_ffn(ffn_model, points1, points2, similarity_threshold=0.4, match_method='coherence'):
    # Initialize the model
    coords_norm_t1 = normalize_points(points1)
    coords_norm_t2 = normalize_points(points2)
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