from typing import Tuple

import h5py
import numpy as np
from numpy import ndarray
import pandas as pd

from CellTracker.fpm import initial_matching_fpm
from CellTracker.simple_alignment import pre_alignment
from CellTracker.robust_match import add_or_remove_points_with_prob_matrix
from CellTracker.trackerlite import BETA, LAMBDA, K_POINTS, get_match_pairs, cal_norm_prob, \
    prgls_with_two_ref, greedy_match
from CellTracker.plot import plot_initial_matching, plot_predicted_movements


def match_coords_to_ids(fpm_model, coordinates_nx3: ndarray, path_to_neuropal_csv: str, skiprows: int = 0,
                        ignored_ids_wba: list = None, ignored_ids_neuropal: list = None, verbosity=4, hdf5_path=None) -> ndarray:
    """
    Match coordinates to ids from a csv file. The csv file is generated by Neuropal. The coordinates are matched to the
    ids by finding the closest coordinates in the csv file to the coordinates to be matched.

    Parameters
    ----------
    fpm_model: Model
        model to be used for matching
    coordinates_nx3: ndarray
        coordinates to be matched
    path_to_neuropal_csv: str
        path to csv file containing ids and coordinates
    skiprows: int
        number of rows to skip when reading csv file
    ignored_ids_wba: list
        ids in the wba coordinates to be ignored in the matching process
    ignored_ids_neuropal: list
        ids in the NeuroPAL coordinates to be ignored in the matching process
    verbosity: int
        verbosity level
    hdf5_path: str
        Save matching results in the path
    """
    # Read ids and coordinates from csv file. The first ? rows are information about the file. The 8th row is the header.
    # The ids are stored in the "User ID" column and the coordinates are stored in the "Real X (um)", "Real Y (um)" and "
    # Real Z (um)" columns.
    # Note that Real Y corresponds to Height and Real X corresponds to Width in the Neuropal GUI.
    if ignored_ids_wba is None:
        ignored_ids_wba = []
    if ignored_ids_neuropal is None:
        ignored_ids_neuropal = []
    df = pd.read_csv(path_to_neuropal_csv, skiprows=skiprows)
    s_and_ids = [(i, id) for i, id in enumerate(df["User ID"].values) if id not in ignored_ids_neuropal]
    ids_neuropal = np.array([id for (i, id) in s_and_ids], dtype="object")
    s_neuropal = np.asarray([i for (i, id) in s_and_ids])
    ids_wba = np.asarray([i for i in range(1, coordinates_nx3.shape[0]+1) if i not in ignored_ids_wba])

    neuropal_coordinates = np.asarray(df[["Real Y (um)", "Real X (um)", "Real Z (um)"]].values)
    return predict_cell_positions(fpm_model, neuropal_coordinates[s_neuropal], coordinates_nx3[ids_wba-1],
                                  ids_wba=ids_wba, ids_neuropal=ids_neuropal, verbosity=verbosity, hdf5_path=hdf5_path)


def predict_cell_positions(fpm_model, coords_neuropal: ndarray, coords_wba: ndarray,
                           beta: float = BETA, lambda_: float = LAMBDA, verbosity: int = 4,
                           hdf5_path: str = None, match_method="coherence", similarity_threshold: float = 0.4,
                           ids_wba = None, ids_neuropal = None,
                           learning_rate=0.5):
    """
    Predict
    """
    print_initial_info(match_method, similarity_threshold, verbosity)

    affine_aligned_coords_t1, neuropal_coords_norm_t2 = pre_alignment(coords_wba, coords_neuropal,
                                                                      fpm_model, match_method,
                                                                      similarity_threshold)

    filtered_coords_norm_t1 = affine_aligned_coords_t1.copy()
    filtered_coords_norm_t2 = neuropal_coords_norm_t2.copy()
    predicted_coords_set1 = affine_aligned_coords_t1.copy()
    n = affine_aligned_coords_t1.shape[0]
    m = neuropal_coords_norm_t2.shape[0]
    inliers_ori = (np.arange(n), np.arange(m))
    iter = 3
    for i in range(iter):
        inliers_pre = (inliers_ori[0], inliers_ori[1])
        similarity_scores = initial_matching_fpm(fpm_model, filtered_coords_norm_t1, filtered_coords_norm_t2, K_POINTS)
        updated_similarity_scores = similarity_scores.copy()
        updated_matched_pairs = get_match_pairs(updated_similarity_scores, filtered_coords_norm_t1,
                                        filtered_coords_norm_t2, threshold=similarity_threshold,
                                        method=match_method)

        if verbosity >= 2 and i == 0:
            print("FPM matching of pre-aligned points:")
            fig = plot_initial_matching(filtered_coords_norm_t1,
                                  filtered_coords_norm_t2,
                                  updated_matched_pairs, 1, -1, ids_tgt=ids_neuropal, ids_ref=ids_wba)

        if verbosity >= 3 and i > 0:
            print(f"FPM matching (iteration={i})):")
            fig = plot_initial_matching(filtered_coords_norm_t1,
                                        filtered_coords_norm_t2,
                                        updated_matched_pairs, 1, -1,
                                        ids_ref=ids_wba[inliers_ori[0]],
                                        ids_tgt=ids_neuropal[inliers_ori[1]])

        match_seg_t1_seg_t2 = np.column_stack(
            (inliers_pre[0][updated_matched_pairs[:, 0]], inliers_pre[1][updated_matched_pairs[:, 1]]))

        predicted_coords_t1_to_t2, similarity_scores = predict_matching_prgls(match_seg_t1_seg_t2, predicted_coords_set1,
                                                   predicted_coords_set1, neuropal_coords_norm_t2,
                                                   (m, n), beta, lambda_)

        if i == iter - 1:
            match_seg_t1_seg_t2, similarity_scores_post = greedy_match(similarity_scores,
                                                                       threshold=0.5)
            break
        else:
            match_seg_t1_seg_t2, similarity_scores_post = greedy_match(similarity_scores, threshold=similarity_threshold)
        filtered_coords_norm_t1, filtered_coords_norm_t2, inliers_ori = \
            add_or_remove_points_with_prob_matrix(similarity_scores_post, predicted_coords_t1_to_t2, neuropal_coords_norm_t2)

        if verbosity >= 4:
            print(f"prgls transformation (iteration={i}); beta={beta}):")
            fig = plot_predicted_movements(predicted_coords_set1, neuropal_coords_norm_t2, predicted_coords_t1_to_t2, -1, 1)

        beta *= 0.7
        predicted_coords_set1 = (predicted_coords_set1 + (predicted_coords_t1_to_t2 - predicted_coords_set1) * learning_rate)
        filtered_coords_norm_t1 = predicted_coords_set1[inliers_ori[0]]

    if verbosity >= 1:
        plot_final_matching_results(affine_aligned_coords_t1, ids_neuropal, ids_wba, match_seg_t1_seg_t2,
                                    neuropal_coords_norm_t2)

    if hdf5_path is not None:
        save_to_hdf5(affine_aligned_coords_t1, hdf5_path, ids_neuropal, ids_wba, match_seg_t1_seg_t2,
                     neuropal_coords_norm_t2)

    return np.asarray([(ids_wba[i], ids_neuropal[j]) for i, j in match_seg_t1_seg_t2])


def save_to_hdf5(affine_aligned_coords_t1, hdf5_path, ids_neuropal, ids_wba, match_seg_t1_seg_t2,
                 neuropal_coords_norm_t2):
    with h5py.File(hdf5_path, 'w') as new_file:
        dataset_t1 = new_file.create_dataset("affine_aligned_coords_t1", data=affine_aligned_coords_t1)
        dataset_t2 = new_file.create_dataset("neuropal_coords_norm_t2", data=neuropal_coords_norm_t2)
        dataset_match = new_file.create_dataset("match_seg_t1_seg_t2", data=match_seg_t1_seg_t2)
        dataset_t2.attrs["ids_neuropal"] = ids_neuropal.tolist()
        dataset_t1.attrs["ids_wba"] = ids_wba.tolist()


def plot_final_matching_results(affine_aligned_coords_t1, ids_neuropal, ids_wba, match_seg_t1_seg_t2,
                                neuropal_coords_norm_t2):
    print("Final matching 2D x-y:")
    fig = plot_initial_matching(affine_aligned_coords_t1,
                                neuropal_coords_norm_t2,
                                match_seg_t1_seg_t2,
                                t1=1, t2=-1,
                                fig_width_px=2400,
                                ids_tgt=ids_neuropal, ids_ref=ids_wba)
    print("Final matching 2D x-z:")
    fig = plot_initial_matching(affine_aligned_coords_t1[:, [2, 1, 0]],
                                neuropal_coords_norm_t2[:, [2, 1, 0]],
                                match_seg_t1_seg_t2,
                                t1=1, t2=-1,
                                fig_width_px=2400,
                                ids_tgt=ids_neuropal, ids_ref=ids_wba)
    # shift = (0.5, 0, 0)
    # fig = plot_matching_2d_with_plotly(neuropal_coords_norm_t2, affine_aligned_coords_t1,
    #                               match_seg_t1_seg_t2[:, [1, 0]],
    #                                    ids_ref=ids_neuropal, ids_tgt=ids_wba, shift=shift)
    # fig.update_layout(width=1500, height=1000)
    # fig.show()
    print("Final matching 3D:")
    fig = plot_initial_matching(affine_aligned_coords_t1,
                                neuropal_coords_norm_t2,
                                match_seg_t1_seg_t2,
                                t1=1, t2=-1,
                                fig_width_px=2400,
                                ids_tgt=ids_neuropal, ids_ref=ids_wba, show_3d=True)


def print_initial_info(match_method, similarity_threshold, verbosity):
    if verbosity >= 0:
        print(f"Matching method: {match_method}")
        print(f"Threshold for similarity: {similarity_threshold}")
        print(f"Post processing method: prgls")


def predict_matching_prgls(matched_pairs, confirmed_coords_norm_t1, segmented_coords_norm_t1, segmented_coords_norm_t2,
                           similarity_scores_shape: Tuple[int, int], beta=BETA, lambda_=LAMBDA):
    normalized_prob = cal_norm_prob(matched_pairs, similarity_scores_shape)
    tracked_coords_norm_t2, prob_mxn = prgls_with_two_ref(normalized_prob, segmented_coords_norm_t2,
                                                   segmented_coords_norm_t1, confirmed_coords_norm_t1,
                                                   beta=beta, lambda_=lambda_)

    return tracked_coords_norm_t2, prob_mxn

