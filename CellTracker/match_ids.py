from typing import List

import h5py
import numpy as np
from numpy import ndarray
import pandas as pd

from CellTracker.fpm import initial_matching_fpm
from CellTracker.simple_alignment import pre_alignment, greedy_match, get_match_pairs, K_POINTS
from CellTracker.robust_match import update_inliers_points, add_or_remove_points
from CellTracker.test_matching_models import predict_matching_prgls, affine_align_by_fpm
from CellTracker.trackerlite import BETA, LAMBDA, predict_new_positions, rotate_for_visualization, predict_by_prgls
from CellTracker.plot import plot_initial_matching, plot_predicted_movements
from CellTracker.utils import normalize_points


def match_coords_to_ids(fpm_model, coordinates_nx3: ndarray, path_to_neuropal_csv: str, skiprows: int = 0,
                        ignored_ids_wba: List[int] = None, ignored_ids_neuropal: List[str] = None, verbosity=4, hdf5_path=None) -> ndarray:
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

    ids_neuropal, indices_neuropal, neuropal_coordinates = read_neuropal_csv(ignored_ids_neuropal, path_to_neuropal_csv,
                                                                             skiprows)
    ids_wba = np.asarray([i for i in range(1, coordinates_nx3.shape[0]+1) if i not in ignored_ids_wba])

    return predict_cell_links(fpm_model, neuropal_coordinates[indices_neuropal], coordinates_nx3[ids_wba - 1],
                              ids_wba=ids_wba, ids_neuropal=ids_neuropal, verbosity=verbosity, hdf5_path=hdf5_path)


def read_neuropal_csv(ignored_ids_neuropal: List[str], path_to_neuropal_csv: str, skiprows: int):
    df = pd.read_csv(path_to_neuropal_csv, skiprows=skiprows)
    s_and_ids = [(i, id) for i, id in enumerate(df["User ID"].values) if id not in ignored_ids_neuropal]
    ids_neuropal = np.array([id for (i, id) in s_and_ids], dtype="object")
    indices_neuropal = np.asarray([i for (i, id) in s_and_ids])
    neuropal_coordinates = np.asarray(df[["Real Y (um)", "Real X (um)", "Real Z (um)"]].values)
    return ids_neuropal, indices_neuropal, neuropal_coordinates


def predict_cell_links(fpm_model, coords_neuropal: ndarray, coords_wba: ndarray,
                       beta: float = BETA, lambda_: float = LAMBDA, verbosity: int = 4,
                       hdf5_path: str = None, match_method="coherence", similarity_threshold: float = 0.4,
                       ids_wba = None, ids_neuropal = None,
                       learning_rate=0.5):
    """
    Predict
    """
    affine_aligned_coords_t1, match_seg_t1_seg_t2, neuropal_coords_norm_t2 = \
        _predict_cell_matchings(coords_wba, coords_neuropal, fpm_model, ids_wba, ids_neuropal, match_method, lambda_,
                                learning_rate, similarity_threshold, beta, verbosity)

    if verbosity >= 1:
        plot_final_matching_results(affine_aligned_coords_t1, ids_neuropal, ids_wba, match_seg_t1_seg_t2,
                                    neuropal_coords_norm_t2)

    if hdf5_path is not None:
        save_to_hdf5(affine_aligned_coords_t1, hdf5_path, ids_neuropal, ids_wba, match_seg_t1_seg_t2,
                     neuropal_coords_norm_t2)

    return np.asarray([(ids_wba[i], ids_neuropal[j]) for i, j in match_seg_t1_seg_t2])


# def predict_cell_positions(coords_t1, coords_t2, fpm_model, ids_t1, ids_t2, match_method, lambda_, learning_rate,
#                            similarity_threshold, beta, verbosity):
#     print_initial_info(match_method, similarity_threshold, verbosity)
#     affine_aligned_coords_t1, neuropal_coords_norm_t2 = pre_alignment(coords_t1, coords_t2,
#                                                                       match_model=fpm_model, predict_method=initial_matching_fpm,
#                                                                       match_method=match_method,
#                                                                       similarity_threshold=similarity_threshold,
#                                                                       ttype="affine")
#     filtered_coords_norm_t1 = affine_aligned_coords_t1.copy()
#     filtered_coords_norm_t2 = neuropal_coords_norm_t2.copy()
#     predicted_coords_set1 = affine_aligned_coords_t1.copy()
#     n = affine_aligned_coords_t1.shape[0]
#     m = neuropal_coords_norm_t2.shape[0]
#     inliers_updated = (np.arange(n), np.arange(m))
#     iter = 3
#     similarity_scores = initial_matching_fpm(fpm_model, filtered_coords_norm_t1, filtered_coords_norm_t2, K_POINTS)
#     for i in range(iter):
#         inliers_pre = (inliers_updated[0], inliers_updated[1])
#         updated_similarity_scores = similarity_scores[np.ix_(inliers_updated[1], inliers_updated[0])].copy()
#         updated_matched_pairs = get_match_pairs(updated_similarity_scores, filtered_coords_norm_t1,
#                                                 filtered_coords_norm_t2, threshold=similarity_threshold,
#                                                 method=match_method)
#
#         print_and_plot_matching(filtered_coords_norm_t1, filtered_coords_norm_t2, i, ids_t2, ids_t1,
#                                 inliers_updated, updated_matched_pairs, verbosity)
#
#         match_seg_t1_seg_t2 = np.column_stack(
#             (inliers_pre[0][updated_matched_pairs[:, 0]], inliers_pre[1][updated_matched_pairs[:, 1]]))
#
#         predicted_coords_t1_to_t2, similarity_scores = predict_matching_prgls(match_seg_t1_seg_t2,
#                                                                               predicted_coords_set1,
#                                                                               predicted_coords_set1,
#                                                                               neuropal_coords_norm_t2,
#                                                                               (m, n), beta, lambda_)
#         if verbosity >= 4:
#             print(f"prgls transformation (iteration={i}); beta={beta}):")
#             fig = plot_predicted_movements(predicted_coords_set1, neuropal_coords_norm_t2, predicted_coords_t1_to_t2,
#                                            -1, 1)
#
#         if i == iter - 1:
#             match_seg_t1_seg_t2 = greedy_match(similarity_scores, threshold=0.5)
#             print(similarity_scores.shape)
#             break
#         else:
#             match_seg_t1_seg_t2 = greedy_match(similarity_scores, threshold=similarity_threshold)
#         filtered_coords_norm_t1, filtered_coords_norm_t2, inliers_updated = \
#             update_inliers_points(match_seg_t1_seg_t2, predicted_coords_t1_to_t2, neuropal_coords_norm_t2)
#
#         predicted_coords_set1 = (
#                     predicted_coords_set1 + (predicted_coords_t1_to_t2 - predicted_coords_set1) * learning_rate)
#         filtered_coords_norm_t1 = predicted_coords_set1[inliers_updated[0]]
#     return affine_aligned_coords_t1, match_seg_t1_seg_t2, neuropal_coords_norm_t2


def _predict_cell_matchings(coords_t1, coords_t2, fpm_model, ids_t1, ids_t2, match_method, lambda_, learning_rate,
                            similarity_threshold, beta, verbosity, filter_points: bool = True):
    """
    Return predicted matchings and the aligned/normalized coordinates
    """
    # Load normalized coordinates at t1 and t2: segmented_pos is un-rotated, while other coords are rotated
    segmented_coords_norm_t1 = normalize_points(coords_t1)
    segmented_coords_norm_t2 = normalize_points(coords_t2)
    subset_t1, subset_t2 = np.arange(len(coords_t1)), np.arange(len(coords_t2))
    subset = (subset_t1, subset_t2)

    n, m = segmented_coords_norm_t1.shape[0], segmented_coords_norm_t2.shape[0]
    aligned_coords_subset_norm_t1, coords_subset_norm_t2, _, affine_tform = affine_align_by_fpm(fpm_model,
                                                                      coords_norm_t1=segmented_coords_norm_t1,
                                                                      coords_norm_t2=segmented_coords_norm_t2)
    aligned_segmented_coords_norm_t1 = affine_tform(segmented_coords_norm_t1)
    moved_seg_coords_t1 = aligned_segmented_coords_norm_t1.copy()

    iter = 3
    for i in range(iter):
        similarity_scores = initial_matching_fpm(fpm_model, aligned_coords_subset_norm_t1, coords_subset_norm_t2, K_POINTS)
        _matched_pairs_subset = get_match_pairs(similarity_scores, aligned_coords_subset_norm_t1, coords_subset_norm_t2,
                                        threshold=similarity_threshold, method=match_method)
        matched_pairs = np.column_stack(
            (subset[0][_matched_pairs_subset[:, 0]], subset[1][_matched_pairs_subset[:, 1]]))

        print_and_plot_matching(aligned_coords_subset_norm_t1, coords_subset_norm_t2, i, ids_t2, ids_t1,
                                subset, _matched_pairs_subset, verbosity)

        if i == iter - 1:
            tracked_coords_t1_to_t2, posterior_mxn = predict_by_prgls(
                matched_pairs, aligned_segmented_coords_norm_t1, aligned_segmented_coords_norm_t1,
                segmented_coords_norm_t2,(m, n), beta, lambda_)

            tracked_coords_norm_t2 = tracked_coords_t1_to_t2.copy()
            pairs_seg_t1_seg_t2 = greedy_match(posterior_mxn, threshold=0.5)
            pairs_in_confirmed_subset = np.asarray(
                [(np.nonzero(subset_t1==i)[0][0], j) for i, j in pairs_seg_t1_seg_t2 if i in subset_t1])
            tracked_coords_norm_t2[pairs_in_confirmed_subset[:, 0], :] = segmented_coords_norm_t2[pairs_in_confirmed_subset[:, 1], :]
        else:
            # Predict the corresponding positions in t2 of all the segmented cells in t1
            predicted_coords_t1_to_t2, posterior_mxn = predict_by_prgls(
                matched_pairs, aligned_segmented_coords_norm_t1, aligned_segmented_coords_norm_t1,
                segmented_coords_norm_t2, (m, n), beta, lambda_)

            if verbosity >= 4:
                print(f"prgls transformation (iteration={i}); beta={beta}):")
                fig = plot_predicted_movements(moved_seg_coords_t1, segmented_coords_norm_t2,
                                               predicted_coords_t1_to_t2,
                                               -1, 1)

            if filter_points:
                # Predict the corresponding positions in t1 of all the segmented cells in t2
                predicted_coords_t2_to_t1, _ = predict_by_prgls(
                    matched_pairs[:, [1, 0]], segmented_coords_norm_t2, segmented_coords_norm_t2, aligned_segmented_coords_norm_t1,
                    (n, m), beta, lambda_)

                aligned_coords_subset_norm_t1, coords_subset_norm_t2, subset = \
                    add_or_remove_points(
                        predicted_coords_t1_to_t2, predicted_coords_t2_to_t1,
                        aligned_segmented_coords_norm_t1, segmented_coords_norm_t2,
                        matched_pairs)

            moved_seg_coords_t1 += (predicted_coords_t1_to_t2 - moved_seg_coords_t1) * learning_rate
            aligned_coords_subset_norm_t1 = moved_seg_coords_t1[subset[0]]

    return aligned_segmented_coords_norm_t1, pairs_seg_t1_seg_t2, segmented_coords_norm_t2


def print_and_plot_matching(filtered_coords_norm_t1, filtered_coords_norm_t2, i, ids_t2, ids_t1, inliers_updated,
                            updated_matched_pairs, verbosity):
    if verbosity >= 2 and i == 0:
        print("FPM matching of pre-aligned points:")
        fig = plot_initial_matching(filtered_coords_norm_t1,
                                    filtered_coords_norm_t2,
                                    updated_matched_pairs, 1, -1, ids_tgt=ids_t2, ids_ref=ids_t1)
    if verbosity >= 3 and i > 0:
        print(f"FPM matching (iteration={i})):")
        fig = plot_initial_matching(filtered_coords_norm_t1,
                                    filtered_coords_norm_t2,
                                    updated_matched_pairs, 1, -1,
                                    ids_ref=ids_t1[inliers_updated[0]],
                                    ids_tgt=ids_t2[inliers_updated[1]])


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
                                fig_height_px=2400,
                                ids_tgt=ids_neuropal, ids_ref=ids_wba)
    print("Final matching 2D x-z:")
    fig = plot_initial_matching(affine_aligned_coords_t1[:, [2, 1, 0]],
                                neuropal_coords_norm_t2[:, [2, 1, 0]],
                                match_seg_t1_seg_t2,
                                t1=1, t2=-1,
                                fig_height_px=2400,
                                ids_tgt=ids_neuropal, ids_ref=ids_wba)
    print("Final matching 3D:")
    fig = plot_initial_matching(affine_aligned_coords_t1,
                                neuropal_coords_norm_t2,
                                match_seg_t1_seg_t2,
                                t1=1, t2=-1,
                                fig_height_px=2400,
                                ids_tgt=ids_neuropal, ids_ref=ids_wba, show_3d=True)


def print_initial_info(match_method, similarity_threshold, verbosity):
    if verbosity >= 0:
        print(f"Matching method: {match_method}")
        print(f"Threshold for similarity: {similarity_threshold}")
        print(f"Post processing method: prgls")
