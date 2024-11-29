from dataclasses import dataclass
from typing import List, Tuple

import h5py
import numpy as np
from numpy import ndarray

from CellTracker.coord_image_transformer import cal_interp_factor
from CellTracker.fpm import initial_matching_fpm, FPMPart2Model
from CellTracker.plot import plot_initial_matching, plot_predicted_movements
from CellTracker.robust_match import add_or_remove_points
from CellTracker.simple_alignment import greedy_match, get_match_pairs, K_POINTS
from CellTracker.test_matching_models import affine_align_by_fpm
from CellTracker.trackerlite import BETA, LAMBDA, predict_by_prgls
from CellTracker.utils import normalize_points


@dataclass
class NeuroPALData:
    neuron_ids_n: ndarray  # dtype=int
    neuron_names_n: ndarray # dtype="object"
    neuron_coordinates_nx3: ndarray # dtype=float
    neuropal_rfp_img_yxz: ndarray # shape=(y, x, z)
    neuropal_rgb_img_yxzc: ndarray # shape=(y, x, z, c)


def extract_coordinates_wba(tracking_results_h5: str) -> Tuple[ndarray, float]:
    with h5py.File(tracking_results_h5, 'r') as f:
        interp = cal_interp_factor(f.attrs["voxel_size_yxz"][:])
        center_vol = f.attrs["t_initial"]
        coordinates_zyx_nx3 = f['coords_txnx3'][center_vol - 1, ...]
        print(f"Extracted coordinates from WBA tracking results at volume {center_vol}.")
    return coordinates_zyx_nx3[:, [1, 2, 0]], interp

def extract_data_from_neuropal_nwb(path_to_neuropal_nwb: str) -> NeuroPALData:
    with h5py.File(path_to_neuropal_nwb, 'r') as f:
        channel_RFP = f["acquisition/NeuroPALImageRaw/RGBW_channels"][3]
        channel_RGB = f["acquisition/NeuroPALImageRaw/RGBW_channels"][0:3]
        print(f"channel_RFP: {channel_RFP}")
        neuropal_img = f["acquisition/NeuroPALImageRaw/data"]
        print(f"neuropal_img.shape: {neuropal_img.shape}")
        neuropal_rfp_img_yxz = neuropal_img[..., channel_RFP]
        neuropal_rgb_img_yxzc = neuropal_img[..., channel_RGB]

        id_labels_index = f["processing/NeuroPAL/NeuroPALSegmentation/NeuroPALNeurons/ID_labels_index"]
        id_labels = f["processing/NeuroPAL/NeuroPALSegmentation/NeuroPALNeurons/ID_labels"]
        print(f"id_labels_index.shape: {id_labels_index.shape}")

        neuron_names_n = []
        i_start = 0
        for i in id_labels_index[:]:
            neuron_names_n.append((
                b"".join(id_labels[i_start:i].tolist()).decode('utf-8')
            ))
            i_start = i
        dtype = h5py.string_dtype(encoding='utf-8')
        neuron_names_n = np.asarray(neuron_names_n, dtype=dtype)
        print(f"Number of identified neurons: {len(neuron_names_n)}")

        cell_positions = f["processing/NeuroPAL/NeuroPALSegmentation/NeuroPALNeurons/voxel_mask"][:]
        cell_positions_yxz_n = np.asarray([(pos["x"], pos["y"], pos["z"]) for pos in cell_positions]) # unit: voxel
        print(f"cell_positions.shape: {cell_positions_yxz_n.shape}")

        cell_ids_n = f["processing/NeuroPAL/NeuroPALSegmentation/NeuroPALNeurons/voxel_mask_index"][:]
        assert len(cell_ids_n) == len(neuron_names_n), "Number of cell_ids and neuron_names should be the same."

    return NeuroPALData(cell_ids_n, neuron_names_n, cell_positions_yxz_n, neuropal_rfp_img_yxz, neuropal_rgb_img_yxzc)


def extract_data_from_wba(raw_wba_file, tracking_results_h5: str, dset_name="default"):
    with h5py.File(tracking_results_h5, 'r') as f:
        rfp_channel = f.attrs["raw_channel_nuclei"]
        t0_index = f.attrs["t_initial"] - 1

    with h5py.File(raw_wba_file, 'r') as f:
        rfp_img = f[dset_name][t0_index, :, rfp_channel, :, :]
    return rfp_img


def match_coords_to_ids_by_neuropaldata(neuropal_data: NeuroPALData, tracking_results_h5: str, fpm_model,
                                        fpm_model_rot=None):
    # Extract neuropal coordinates, and wba coordinates at t_initial
    neuropal_coords_voxel_yxz_nx3 = neuropal_data.neuron_coordinates_nx3
    neuronal_names_neuropal = neuropal_data.neuron_names_n
    coords_voxel_wba_yxz_nx3, interp_factor = extract_coordinates_wba(tracking_results_h5)

    coords_real_wba_yxz_nx3 = coords_voxel_wba_yxz_nx3 * np.asarray([[1, 1, interp_factor]])
    neuropal_coords_real_yxz_nx3 = neuropal_coords_voxel_yxz_nx3 * np.asarray([[1, 1, interp_factor]])

    # pre-matching
    fpm_models = fpm_model, FPMPart2Model(fpm_model.comparator)
    ids_wba = np.asarray([i for i in range(1, coords_real_wba_yxz_nx3.shape[0] + 1)])
    affine_aligned_coords_t1, match_seg_t1_seg_t2, neuropal_coords_norm_t2 = \
        _predict_cell_matchings(coords_real_wba_yxz_nx3, neuropal_coords_real_yxz_nx3, fpm_models, ids_wba, neuronal_names_neuropal,
                                match_method="coherence", lambda_=LAMBDA, beta=BETA,
                                learning_rate=0.5, similarity_threshold=0.4, verbosity=0)

    plot_final_matching_results(affine_aligned_coords_t1, neuronal_names_neuropal,
                                ids_wba, match_seg_t1_seg_t2,
                                neuropal_coords_norm_t2)

    with h5py.File(tracking_results_h5, 'a') as f:
        if "link_with_Neuropal" in f:
            del f["link_with_Neuropal"]
        if "confirmed_wba_cells" in f.attrs:
            del f.attrs["confirmed_wba_cells"]
        if "confirmed_neuropal_cells" in f.attrs:
            del f.attrs["confirmed_neuropal_cells"]
        group = f.create_group("link_with_Neuropal")
        group.create_dataset("neuron_ids_n", data=neuropal_data.neuron_ids_n)
        group.create_dataset("neuron_names_n", data=neuropal_data.neuron_names_n)
        group.create_dataset("neuron_coordinates_yxz_nx3", data=neuropal_coords_voxel_yxz_nx3)
        group.create_dataset("neuropal_rfp_img_yxz", data=neuropal_data.neuropal_rfp_img_yxz)
        group.create_dataset("match_wba_neuropal_mx2", data=match_seg_t1_seg_t2)
        group.create_dataset("neuropal_rgb_img_yxzc", data=neuropal_data.neuropal_rgb_img_yxzc)


def match_coords_to_ids_by_csv(fpm_model, coordinates_nx3: ndarray, path_to_neuropal_csv: str, skiprows: int = 0,
                               ignored_ids_wba: List[int] = None, ignored_ids_neuropal: List[str] = None, verbosity=4, tracking_results_h5=None) -> ndarray:
    """
    Match coordinates to ids from a csv file. The csv file is generated by Neuropal. The coordinates are matched to the
    ids by finding the closest coordinates in the csv file to the coordinates to be matched.

    Parameters
    ----------
    fpm_models: Model
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
    tracking_results_h5: str
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
    return predict_cell_links(fpm_model, neuropal_coordinates[indices_neuropal],
                              coordinates_nx3[ids_wba - 1],
                              ids_wba=ids_wba, ids_neuropal=ids_neuropal, verbosity=verbosity,
                              tracking_results_h5=tracking_results_h5)


def read_neuropal_csv(ignored_ids_neuropal: List[str], path_to_neuropal_csv: str, skiprows: int):
    # Read ids and coordinates from csv file.
    data = np.genfromtxt(path_to_neuropal_csv, delimiter=',', skip_header=skiprows, dtype=None, encoding=None)
    column_names = data[0]
    data_dict = {column: data[1:, idx] for idx, column in enumerate(column_names)}

    # Extract ids
    user_ids = data_dict["User ID"]
    s_and_ids = [(i, user_id) for i, user_id in enumerate(user_ids) if user_id not in ignored_ids_neuropal]
    ids_neuropal = np.array([str(id) for (i, id) in s_and_ids], dtype='object')
    indices_neuropal = np.asarray([i for (i, id) in s_and_ids])

    # Extract coordinates
    y_coords = data_dict["Real Y (um)"].astype(float)
    x_coords = data_dict["Real X (um)"].astype(float)
    z_coords = data_dict["Real Z (um)"].astype(float)
    neuropal_coordinates = np.stack((y_coords, x_coords, z_coords), axis=-1)

    return ids_neuropal, indices_neuropal, neuropal_coordinates


def predict_cell_links(fpm_model, coords_neuropal: ndarray, coords_wba: ndarray, tracking_results_h5: str,
                       beta: float = BETA, lambda_: float = LAMBDA, verbosity: int = 4,
                       match_method="coherence", similarity_threshold: float = 0.4,
                       ids_wba = None, ids_neuropal = None,
                       learning_rate=0.5):
    fpm_models = fpm_model, FPMPart2Model(fpm_model.comparator)
    affine_aligned_coords_t1, match_seg_t1_seg_t2, neuropal_coords_norm_t2 = \
        _predict_cell_matchings(coords_wba, coords_neuropal, fpm_models, ids_wba, ids_neuropal, match_method, lambda_,
                                learning_rate, similarity_threshold, beta, verbosity)

    if verbosity >= 1:
        plot_final_matching_results(affine_aligned_coords_t1, ids_neuropal, ids_wba, match_seg_t1_seg_t2,
                                    neuropal_coords_norm_t2)

    save_to_tracking_results_h5(affine_aligned_coords_t1, tracking_results_h5, ids_neuropal, ids_wba, match_seg_t1_seg_t2,
                                neuropal_coords_norm_t2)

    return np.asarray([(ids_wba[i], ids_neuropal[j]) for i, j in match_seg_t1_seg_t2])


def _predict_cell_matchings(coords_t1, coords_t2, fpm_models, ids_t1, ids_t2, match_method, lambda_, learning_rate,
                            similarity_threshold, beta, verbosity, filter_points: bool = True):
    """
    Return predicted matchings and the aligned/normalized coordinates
    """
    # Load normalized coordinates at t1 and t2: segmented_pos is un-rotated, while other coords are rotated
    segmented_coords_norm_t1, _ = normalize_points(coords_t1)
    segmented_coords_norm_t2, _ = normalize_points(coords_t2)
    subset_t1, subset_t2 = np.arange(len(coords_t1)), np.arange(len(coords_t2))
    subset = (subset_t1, subset_t2)

    n, m = segmented_coords_norm_t1.shape[0], segmented_coords_norm_t2.shape[0]
    aligned_coords_subset_norm_t1, coords_subset_norm_t2, _, affine_tform = affine_align_by_fpm(fpm_models,
                                                                                                coords_norm_t1=segmented_coords_norm_t1,
                                                                                                coords_norm_t2=segmented_coords_norm_t2)
    aligned_segmented_coords_norm_t1 = affine_tform(segmented_coords_norm_t1)
    moved_seg_coords_t1 = aligned_segmented_coords_norm_t1.copy()

    iter = 3
    for i in range(iter):
        similarity_scores = initial_matching_fpm(fpm_models, aligned_coords_subset_norm_t1, coords_subset_norm_t2, K_POINTS)
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


def save_to_tracking_results_h5(affine_aligned_coords_t1, hdf5_path, ids_neuropal, ids_wba, match_seg_t1_seg_t2,
                                neuropal_coords_norm_t2):
    with h5py.File(hdf5_path, 'a') as new_file:
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
                                t1_name="wba_t0", t2_name="neuropal",
                                fig_height_px=2400,
                                ids_tgt=ids_neuropal, ids_ref=ids_wba)
    print("Final matching 2D x-z:")
    fig = plot_initial_matching(affine_aligned_coords_t1[:, [2, 1, 0]],
                                neuropal_coords_norm_t2[:, [2, 1, 0]],
                                match_seg_t1_seg_t2,
                                t1_name="wba_t0", t2_name="neuropal",
                                fig_height_px=2400,
                                ids_tgt=ids_neuropal, ids_ref=ids_wba)
    print("Final matching 3D:")
    fig = plot_initial_matching(affine_aligned_coords_t1,
                                neuropal_coords_norm_t2,
                                match_seg_t1_seg_t2,
                                t1_name="wba_t0", t2_name="neuropal",
                                fig_height_px=2400,
                                ids_tgt=ids_neuropal, ids_ref=ids_wba, show_3d=True)


def print_initial_info(match_method, similarity_threshold, verbosity):
    if verbosity >= 0:
        print(f"Matching method: {match_method}")
        print(f"Threshold for similarity: {similarity_threshold}")
        print(f"Post processing method: prgls")
