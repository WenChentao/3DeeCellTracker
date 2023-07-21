from numpy import ndarray
from skimage.transform import estimate_transform
from tensorflow.python.keras import Model

from CellTracker.fpm import initial_matching_fpm
from CellTracker.trackerlite import K_POINTS, get_match_pairs
from CellTracker.utils import normalize_points


def affine_align_and_normalize(coords_t1: ndarray, coords_t2: ndarray, fpm_model: Model, match_method: str,
                               similarity_threshold: float):
    coords_norm_t2, _ = normalize_points(coords_t2, return_para=True)
    coords_norm_t1, _ = normalize_points(coords_t1, return_para=True)
    # affine transformation
    similarity_scores = initial_matching_fpm(fpm_model, coords_norm_t1, coords_norm_t2, K_POINTS)
    initial_matched_pairs = get_match_pairs(similarity_scores, coords_norm_t1, coords_norm_t2,
                                            threshold=similarity_threshold, method=match_method)
    src = coords_norm_t1[initial_matched_pairs[:, 0]]
    dst = coords_norm_t2[initial_matched_pairs[:, 1]]
    tform = estimate_transform("affine", src, dst)
    affine_aligned_coords_t1 = tform(coords_norm_t1)
    return affine_aligned_coords_t1, coords_norm_t2


def rotation_align_by_control_points(coords_norm_t1: ndarray, coords_norm_t2: ndarray, initial_matched_pairs: ndarray):
    src = coords_norm_t1[initial_matched_pairs[:, 0]]
    dst = coords_norm_t2[initial_matched_pairs[:, 1]]
    tform = estimate_transform("euclidean", src, dst)
    aligned_coords_t1 = tform(coords_norm_t1)
    return aligned_coords_t1