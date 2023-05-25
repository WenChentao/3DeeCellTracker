import numpy as np
from numpy import ndarray

from CellTracker.trackerlite import get_best_pairs


def ransac_match(initial_match_matrix: ndarray, threshold=0.1) -> ndarray:
    """Match points from two point sets by simply choosing the pairs with the highest probability subsequently"""
    match_matrix = initial_match_matrix.copy()

    pairs_list = get_best_pairs(match_matrix, threshold)
    errors = cal_matching_errors(pair_list, ptrs_ref, ptrs_tgt)

    pairs_px2 = np.array(pairs_list)
    normalized_prob = np.full_like(match_matrix, 0.1 / (match_matrix.shape[1] - 1))
    for ref, tgt in pairs_list:
        normalized_prob[tgt, ref] = 0.9
    return normalized_prob, pairs_px2
