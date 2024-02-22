from typing import List

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS

from CellTracker.simple_alignment import align_by_control_points
from CellTracker.test_matching_models import rotation_align_by_fpm, affine_align_by_fpm, local_affine_align_by_fpm, \
    match_by_fpm_prgls
from CellTracker.utils import simple_progress_bar


def match_initial_x_volumes(coords_initialx: List[ndarray], path_pairwise_match_initialx_h5: str, x=20):
    f_matches = h5py.File(path_pairwise_match_initialx_h5, "r")

    updated_matches = np.zeros((x), dtype=object)

    for iteration in range(5):
        no_update = True
        for target_vol in range(x, 1, -1):
            print(f"{iteration}-{target_vol}", end="\r")
            matches_matrix = np.zeros((coords_initialx[0].shape[0], coords_initialx[target_vol - 1].shape[0]), dtype=int)
            for ref, tgt in f_matches[f"1_{target_vol}"][:]:
                matches_matrix[ref, tgt] = 2

            for mid_vol in range(2, x+1):
                if mid_vol == target_vol:
                    continue
                if iteration == 0:
                    pairs_1_mid = f_matches[f"1_{mid_vol}"][:]
                else:
                    pairs_1_mid = updated_matches[mid_vol - 1].copy()
                pairs_mid_tgt = f_matches[f"{mid_vol}_{target_vol}"][:]
                common_mid_values = np.intersect1d(pairs_1_mid[:, 1], pairs_mid_tgt[:, 0])
                for value in common_mid_values:
                    pos1 = np.where(pairs_1_mid[:, 1] == value)[0][0]
                    pos2 = np.where(pairs_mid_tgt[:, 0] == value)[0][0]
                    matches_matrix[pairs_1_mid[pos1, 0], pairs_mid_tgt[pos2, 1]] += 1

            updated_pairs = []
            matches_matrix_copy = matches_matrix.copy()
            for i in range(coords_initialx[0].shape[0]):
                ref, tgt = np.unravel_index(np.argmax(matches_matrix_copy, axis=None), matches_matrix_copy.shape)
                if matches_matrix_copy[ref, tgt] < 2:
                    break
                updated_pairs.append((ref, tgt))
                matches_matrix_copy[ref, :] = 0
                matches_matrix_copy[:, tgt] = 0
            updated_pairs = np.asarray(updated_pairs)

            if iteration==0:
                sorted_original_pairs = sort_array_2d(f_matches[f"1_{target_vol}"][:])
            else:
                sorted_original_pairs = updated_matches[target_vol-1].copy()
            sorted_updated_pairs = sort_array_2d(updated_pairs)
            if len(sorted_updated_pairs) != len(sorted_original_pairs):
                no_update = False
            else:
                no_update = np.allclose(sorted_updated_pairs, sorted_original_pairs)

            updated_matches[target_vol-1] = sorted_updated_pairs
        if no_update:
            print(f"Quit loop when iteration={iteration}")
            break

    pairs_t1_to_tx_list = updated_matches.tolist()
    return pairs_t1_to_tx_list


def sort_array_2d(pairs_nx2):
    sorted_indices = np.argsort(pairs_nx2[:, 0])
    return pairs_nx2[sorted_indices]

def initial_match(coords_tgt_nx3: ndarray, fpm_model, tgt_vol = 1, coords_h5_file: h5py.File = None, coords_npy_files: List[str] = None,
                  fpm_model_rot = None, path_result: str = "./initial_match.h5"):
    if coords_h5_file is None and coords_npy_files is None:
        raise ValueError("Either coords_h5_file or coords_npy_files should be provided")

    if coords_h5_file is not None:
        t = coords_npy_files["seg"].shape[0]
    else:
        t = len(coords_npy_files)

    rigid_aligned_coords_txnx3 = np.full((t, *coords_tgt_nx3.shape), np.nan)
    tform_tx3x4 = np.zeros((t, 3, 4))
    tform_inv_tx3x4 = np.zeros((t, 3, 4))
    points_tgt = coords_tgt_nx3.copy()
    for i in range(t):
        if coords_h5_file is not None:
            points_ref = coords_h5_file[f'coords_{str(i + 1).zfill(6)}'][:]
        else:
            points_ref = np.load(coords_npy_files[i])[:, :3]

        if i+1 == tgt_vol:
            rigid_aligned_coords_txnx3[i,...] = points_tgt.copy()
            tform_tx3x4[i, :,:3] = np.eye(3)
            tform_inv_tx3x4[i, :,:3] = np.eye(3)
            continue
        else:
            pairs_3 = _match_fpm_prgls(fpm_model, fpm_model_rot, points_ref, points_tgt)
            rigid_aligned_coords_mx3, tform = align_by_control_points(points_ref, points_tgt, pairs_3, method="euclidean", return_tform=True)
            for id_ref, id_tgt in pairs_3:
                rigid_aligned_coords_txnx3[i, id_tgt, :] = rigid_aligned_coords_mx3[id_ref, :]
            tform_tx3x4[i, ...] = tform.params[:3, :]
            tform_inv_tx3x4[i, ...] = np.linalg.inv(tform.params)[:3, :]
        simple_progress_bar(i+1, t)

    with h5py.File(path_result, "w") as f:
        f.create_dataset("rigid_aligned_coords", data=rigid_aligned_coords_txnx3)
        f.create_dataset("tform", data=tform_tx3x4)
        f.create_dataset("tform_inv", data=tform_inv_tx3x4)
        print(f"Saved initial match results in {path_result}")


def _match_fpm_prgls(fpm_model, fpm_model_rot, points_ref, points_tgt):
    if fpm_model_rot is None:
        aligned_t1_rot = points_ref
    else:
        aligned_t1_rot, _, pairs_rot = rotation_align_by_fpm(fpm_model_rot, points1=points_ref,
                                                             points2=points_tgt)
    aligned_t1_1, _, pairs_1 = affine_align_by_fpm(fpm_model, points1=aligned_t1_rot,
                                                   points2=points_tgt)
    aligned_t1_2, _, pairs_2 = local_affine_align_by_fpm(fpm_model, points1=aligned_t1_1,
                                                         points2=points_tgt)
    pairs_3 = match_by_fpm_prgls(fpm_model, aligned_t1_2, points_tgt, match_method="coherence")
    return pairs_3


def pairwise_pointsets_distances(corresponding_coords_txnx3: ndarray):
    """
    Computes the pairwise distances between sets of corresponding points.

    Parameters:
    - corresponding_coords_txnx3 (ndarray): A 3-dimensional numpy array with shape (t, n, 3),
      where 't' is the number of point sets, 'n' is the number of points in each set,
      and the last dimension represents the coordinates (x, y, z) of each point.

    Returns:
    - ndarray: A 2-dimensional numpy array with shape (t, t), where each element represents
      the pairwise distances between the corresponding point sets.
    """
    t, n, _ = corresponding_coords_txnx3.shape
    pairwise_dist_txt = np.zeros((t, t))
    for i in range(n):
        coords_i = corresponding_coords_txnx3[:,i,:]
        distances_txt = cdist(coords_i, coords_i, metric='euclidean')
        # Replaces any NaN values in the distance matrix with the mean of the non-NaN values
        distances_txt[np.isnan(distances_txt)] = np.nanmean(distances_txt)
        pairwise_dist_txt += distances_txt
        simple_progress_bar(i + 1, n)
    # Fills the diagonal elements with zeros before returning the final pairwise distance matrix
    np.fill_diagonal(pairwise_dist_txt, 0)
    return pairwise_dist_txt


def get_reference_target_vols_list(distances_txt: ndarray, initial_ref: int, max_num_refs: int = 20):
    t = distances_txt.shape[0]
    reference_taget_vols_list = []
    current_ref_pool = [initial_ref]
    entire_timings = np.arange(1, t+1)
    for i in range(t-1):
        current_refs = np.asarray(current_ref_pool)
        potential_tgts = np.setdiff1d(entire_timings, current_refs)
        ref0_midpoint_tgts_dists_cxp = distances_txt[np.ix_(current_refs - 1, potential_tgts - 1)] + distances_txt[current_refs - 1, initial_ref - 1][:, None]
        ref0_midpoint_tgts_dists_nxp = find_bottom_n_per_column(ref0_midpoint_tgts_dists_cxp, n=max_num_refs)
        sum_ref0_tgts_dists_p = np.sum(ref0_midpoint_tgts_dists_nxp, axis=0)
        next_tgt_local = np.argmin(sum_ref0_tgts_dists_p)
        next_tgt = potential_tgts[next_tgt_local]
        current_ref_pool.append(next_tgt)
        if i+1 <= max_num_refs:
            refs = current_refs
        else:
            refs = current_refs[np.argpartition(ref0_midpoint_tgts_dists_cxp[:, next_tgt_local], max_num_refs)[:max_num_refs]]
        reference_taget_vols_list.append((refs, next_tgt))
        simple_progress_bar(i, t-1)
    return reference_taget_vols_list


def find_bottom_n_per_column(arr_kxt: ndarray, n: int):
    """
    Find the bottom N values in each column of a 2D numpy array.

    Parameters:
    arr: 2D numpy array
    n: Number of bottom values to find

    Returns:
    A 2D numpy array containing the bottom N values from each column, NOT sorted.
    """
    if arr_kxt.shape[0] <= n:
        return arr_kxt

    # Use np.partition to find the n smallest values
    bottom_n_values = np.partition(arr_kxt, n, axis=0)[:n]

    return bottom_n_values


def match_stepwise(coords_ref_nx3: ndarray, initial_ref: int, coords_h5_file: h5py.File, pairwise_dist_txt: ndarray, fpm_model, max_num_neighbors=20):
    matched_pointsets_list = [initial_ref]
    t = pairwise_dist_txt.shape[0]
    predicted_coords_txnx3 = np.full((t, *coords_ref_nx3.shape), np.nan)
    predicted_coords_txnx3[initial_ref-1] = coords_ref_nx3

    for i in range(t-1):
        next_tgt, ref_list = (pairwise_dist_txt, matched_pointsets_list, max_num_neighbors)
        predicted_coords_txnx3[next_tgt-1] = (ref_list, coords_h5_file, predicted_coords_txnx3, next_tgt, fpm_model)
        matched_pointsets_list.append(next_tgt)

    assert np.isnan(predicted_coords_txnx3).any(), "Some volumes are not tracked!"

    return predicted_coords_txnx3


def rigid_transform(tform_3x4: ndarray, coords_nx3: ndarray):
    return np.dot(coords_nx3, tform_3x4[:, :3].T) + tform_3x4[:, 3]


def visualize_pairwise_distances(points_tx2: ndarray):
    plt.figure(figsize=(30, 30))
    plt.scatter(points_tx2[:, 0], points_tx2[:, 1])
    plt.title('2D Visualization of Points Based on Distance Matrix')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    for i, (x, y) in enumerate(points_tx2):
        plt.text(x, y, f'{i + 1}', fontsize=8)


def get_mds_2d_projection(distances_txt):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    points_2d = mds.fit_transform(distances_txt)
    return points_2d
