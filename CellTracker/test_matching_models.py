import numpy as np
from tensorflow.keras import Model

from CellTracker.fpm import initial_matching_fpm, FlexiblePointMatcherOriginal
from CellTracker.simple_alignment import affine_align_and_normalize, rotation_align_by_control_points
from CellTracker.trackerlite import K_POINTS, get_match_pairs
from CellTracker.plot import plot_initial_matching, plot_initial_matching_one_panel
from CellTracker.utils import normalize_points


def rotation_align_by_fpm(fpm_model_rot, points1, points2, similarity_threshold=0.4, match_method='greedy', ids_ref=None,
                          ids_tgt=None):
    # Initialize the model
    coords_norm_t1 = normalize_points(points1)
    coords_norm_t2 = normalize_points(points2)
    initial_matching = initial_matching_fpm(fpm_model_rot, coords_norm_t1, coords_norm_t2,
                                            K_POINTS)
    updated_matching = initial_matching.copy()
    pairs_px2 = get_match_pairs(updated_matching, coords_norm_t1, coords_norm_t2,
                                threshold=similarity_threshold, method=match_method)

    fig = plot_initial_matching(coords_norm_t1, coords_norm_t2, pairs_px2, 1, 2, ids_ref=ids_ref, ids_tgt=ids_tgt)
    aligned_coords_t1 = rotation_align_by_control_points(coords_norm_t1, coords_norm_t2, pairs_px2)
    fig = plot_initial_matching(aligned_coords_t1, coords_norm_t2, pairs_px2, 1, 2, ids_ref=ids_ref, ids_tgt=ids_tgt)
    return aligned_coords_t1, coords_norm_t2, pairs_px2


def match_by_fpm(fpm_model, points1, points2, similarity_threshold=0.4, match_method='coherence', ids_ref=None,
                 ids_tgt=None):
    # Initialize the model
    coords_norm_t1 = normalize_points(points1)
    coords_norm_t2 = normalize_points(points2)
    initial_matching = initial_matching_fpm(fpm_model, coords_norm_t1, coords_norm_t2,
                                            K_POINTS)
    updated_matching = initial_matching.copy()
    pairs_px2 = get_match_pairs(updated_matching, coords_norm_t1, coords_norm_t2,
                                threshold=similarity_threshold, method=match_method)

    fig = plot_initial_matching(coords_norm_t1, coords_norm_t2, pairs_px2, 1, 2, ids_ref=ids_ref, ids_tgt=ids_tgt)
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
