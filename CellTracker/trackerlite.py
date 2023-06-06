import re
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib.patches import ConnectionPatch
from numpy import ndarray
from scipy.interpolate import RBFInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.stats import trim_mean

from CellTracker.coord_image_transformer import Coordinates, plot_predicted_movements, plot_two_pointset_scatters
from CellTracker.ffn import initial_matching_ffn, normalize_points, FFN
from CellTracker.robust_match import filter_outliers, calc_min_path
from CellTracker.stardistwrapper import load_2d_slices_at_time

FIGURE = "figure"
COORDS_REAL = "coords_real"
LABELS = "labels"
TRACK_RESULTS = "track_results"
SEG = "seg"
BETA, LAMBDA, MAX_ITERATION = (2.0, 2.0, 2000)
K_POINTS = 20


class TrackerLite:
    """
    A class that tracks cells in 3D time-lapse images using a trained FFN model.
    """

    def __init__(self, results_dir: str, ffn_model_name: str,
                 proofed_coords_vol1: Coordinates, miss_frame: List[int] = None, basedir: str = "ffn_models"):
        """
        Initialize a new instance of the TrackerLite class.

        Args:
            results_dir: The path to the directory containing the results.
            ffn_model_name: The filename without extension to the FFN model file.
            proofed_coords_vol1: The confirmed cell positions at time step t1.
            miss_frame: A list of missing frames.
        """
        if miss_frame is not None and not isinstance(miss_frame, List):
            raise TypeError(f"miss_frame should be a list or None, but got {type(miss_frame)}")

        self.results_dir = Path(results_dir)
        (self.results_dir / TRACK_RESULTS / FIGURE).mkdir(parents=True, exist_ok=True)
        (self.results_dir / TRACK_RESULTS / COORDS_REAL).mkdir(parents=True, exist_ok=True)
        (self.results_dir / TRACK_RESULTS / LABELS).mkdir(parents=True, exist_ok=True)

        self.ffn_model_path = Path(basedir) / (ffn_model_name + ".h5")
        self.ffn_model = FFN()

        try:
            dummy_input = np.random.random((1, 122))
            _ = self.ffn_model(dummy_input)
            self.ffn_model.load_weights(str(self.ffn_model_path))
        except (OSError, ValueError) as e:
            raise ValueError(f"Failed to load the FFN model from {self.ffn_model_path}: {e}") from e

        self.proofed_coords_vol1 = proofed_coords_vol1
        self.miss_frame = [] if miss_frame is None else miss_frame

    def predict_cell_positions(self, t1: int, t2: int, confirmed_coord_t1: Coordinates = None,
                               beta: float = BETA, lambda_: float = LAMBDA, smoothing=0.01, draw_fig: bool = False,
                               post_processing: str = "tps"):
        """
        Predicts the positions of cells in a 3D image at time step t2, based on their positions at time step t1.

        Args:
            t1: The time step for the first set of cell positions.
            t2: The time step for the second set of cell positions.
            confirmed_coord_t1: The confirmed cell positions at time step t1.

        Returns:
            The predicted cell positions at time step t2.
        """
        assert t2 not in self.miss_frame
        segmented_pos_t1 = self._get_segmented_pos(t1)
        segmented_pos_t2 = self._get_segmented_pos(t2)

        if confirmed_coord_t1 is None:
            confirmed_coord_t1 = segmented_pos_t1

        # Normalize point sets
        confirmed_coords_norm_t1, (mean_t1, scale_t1) = normalize_points(confirmed_coord_t1.real, return_para=True)
        segmented_coords_norm_t2 = (segmented_pos_t2.real - mean_t1) / scale_t1
        segmented_coords_norm_t1 = (segmented_pos_t1.real - mean_t1) / scale_t1

        updated_segmented_coords_norm_t1 = segmented_coords_norm_t1.copy()
        for i in range(3):
            initial_matching = initial_matching_ffn(self.ffn_model, updated_segmented_coords_norm_t1,
                                                    segmented_coords_norm_t2,
                                                    K_POINTS)
            updated_matching = initial_matching.copy()
            matched_pairs = get_match_pairs(updated_matching, segmented_coords_norm_t2,
                                            updated_segmented_coords_norm_t1,
                                            method=self.match_method, threshold=self.similarity_threshold)
            normalized_prob = cal_norm_prob(matched_pairs, updated_matching.shape)
            plot_initial_matching(updated_segmented_coords_norm_t1 * scale_t1 + mean_t1, segmented_pos_t2.real,
                                  matched_pairs, t1, t2)
            updated_segmented_coords_norm_t1 = self.predict_new_positions(segmented_coords_norm_t1, matched_pairs,
                                                                          normalized_prob, post_processing,
                                                                          segmented_coords_norm_t1,
                                                                          segmented_coords_norm_t2,
                                                                          smoothing, beta, lambda_)
        tracked_coords_norm_t2 = updated_segmented_coords_norm_t1
        # tracked_coords_norm_t2 = self.predict_new_positions(confirmed_coords_norm_t1, matched_pairs,
        #                                                     normalized_prob, "tps",
        #                                                     segmented_coords_norm_t1, segmented_coords_norm_t2,
        #                                                     smoothing, beta, lambda_)
        print(f"Matching method: {self.match_method}")
        print(f"Threshold for similarity: {self.similarity_threshold}")
        print(f"Post processing method: {post_processing}")
        tracked_coords_t2 = tracked_coords_norm_t2 * scale_t1 + mean_t1
        if draw_fig:
            fig = plot_predicted_movements(confirmed_coord_t1.real, segmented_pos_t2.real, tracked_coords_t2, t1, t2)

        return Coordinates(tracked_coords_t2,
                           interpolation_factor=self.proofed_coords_vol1.interpolation_factor,
                           voxel_size=self.proofed_coords_vol1.voxel_size,
                           dtype="real")

    def predict_new_positions(self, confirmed_coords_norm_t1, matched_pairs, normalized_prob, post_processing,
                              segmented_coords_norm_t1, segmented_coords_norm_t2, smoothing: float, beta=BETA,
                              lambda_=LAMBDA):
        if post_processing == "tps":
            tracked_coords_norm_t2 = tps_with_two_ref(matched_pairs, segmented_coords_norm_t2,
                                                      segmented_coords_norm_t1, confirmed_coords_norm_t1, smoothing)
        elif post_processing == "prgls":
            tracked_coords_norm_t2, _ = prgls_with_two_ref(normalized_prob, segmented_coords_norm_t2,
                                                           segmented_coords_norm_t1, confirmed_coords_norm_t1,
                                                           beta=beta, lambda_=lambda_)
        else:
            raise ValueError("post_prossing should be either 'tps' or 'prgls'")
        return tracked_coords_norm_t2

    def predict_cell_positions_ensemble(self, skipped_volumes: List[int], t2: int, coord_t1: Coordinates,
                                        beta: float, lambda_: float, sampling_number: int = 20,
                                        adjacent: bool = False):
        coord_prgls = []
        for t1 in get_volumes_list(current_vol=t2, skip_volumes=skipped_volumes, sampling_number=sampling_number,
                                   adjacent=adjacent):
            loaded_coord_t1 = np.load(
                str(self.results_dir / TRACK_RESULTS / COORDS_REAL / f"coords{str(t1).zfill(4)}.npy"))
            loaded_coord_t1_ = Coordinates(loaded_coord_t1, coord_t1.interpolation_factor, coord_t1.voxel_size,
                                           dtype="real")
            coord_prgls.append(self.predict_cell_positions(t1=t1, t2=t2, confirmed_coord_t1=loaded_coord_t1_, beta=beta,
                                                           lambda_=lambda_).real)
        return Coordinates(trim_mean(coord_prgls, 0.1, axis=0),
                           interpolation_factor=self.proofed_coords_vol1.interpolation_factor,
                           voxel_size=self.proofed_coords_vol1.voxel_size, dtype="real")

    def match_by_ffn(self, t1: int, t2: int, confirmed_coord_t1: Coordinates = None, similarity_threshold=0.3,
                     match_method: str = "coherence"):
        self.match_method = match_method
        self.similarity_threshold = similarity_threshold
        print(f"Matching method: {self.match_method}")
        print(f"Threshold for similarity: {self.similarity_threshold}")
        assert t2 not in self.miss_frame
        segmented_pos_t1 = self._get_segmented_pos(t1)
        segmented_pos_t2 = self._get_segmented_pos(t2)

        if confirmed_coord_t1 is None:
            confirmed_coord_t1 = segmented_pos_t1

        pairs_px2 = self.match_two_point_sets(confirmed_coord_t1, segmented_pos_t2)
        plot_initial_matching(confirmed_coord_t1.real, segmented_pos_t2.real, pairs_px2, t1, t2)

    def match_two_point_sets(self, confirmed_coord_t1, segmented_pos_t2):
        confirmed_coords_norm_t1, (mean_t1, scale_t1) = normalize_points(confirmed_coord_t1.real, return_para=True)
        segmented_coords_norm_t2 = (segmented_pos_t2.real - mean_t1) / scale_t1
        initial_matching = initial_matching_ffn(self.ffn_model, confirmed_coords_norm_t1, segmented_coords_norm_t2,
                                                K_POINTS)
        updated_matching = initial_matching.copy()
        pairs_px2 = get_match_pairs(updated_matching, segmented_coords_norm_t2, confirmed_coords_norm_t1,
                                    threshold=self.similarity_threshold, method=self.match_method)
        return pairs_px2

    def _get_segmented_pos(self, t: int) -> Coordinates:
        interp_factor = self.proofed_coords_vol1.interpolation_factor
        voxel_size = self.proofed_coords_vol1.voxel_size

        pos = Coordinates(np.load(str(self.results_dir / SEG / f"coords{str(t).zfill(4)}.npy")),
                          interpolation_factor=interp_factor, voxel_size=voxel_size, dtype="raw")
        return pos

    def activities(self, raw_path: str, discard_ratio: float = 0.1):
        tracked_labels_path = self.results_dir / TRACK_RESULTS / LABELS
        filenames = glob(str(tracked_labels_path / "*t*.tif"))
        assert len(filenames) > 0, f"No labels files were found in {tracked_labels_path / '*t*.tif'}"
        numbers = [int(re.findall(r"t(\d+)", f)[0]) for f in filenames]
        smallest_number = min(numbers)
        largest_number = max(numbers)

        for t in range(smallest_number, largest_number + 1):
            print(f"{t=}...", end="\r")
            try:
                # Load 2D slices at time t
                raw = load_2d_slices_at_time(raw_path, t=t)
            except FileNotFoundError:
                # Handle missing image files
                print(f"Warning: Raw images at t={t - 1} cannot be loaded! Stop calculation!")
                break

            try:
                # Load 2D slices at time t
                labels_img = load_2d_slices_at_time(str(tracked_labels_path / "*t%04i*.tif"), t=t, do_normalize=False)
            except FileNotFoundError:
                # Handle missing image files
                print(f"Warning: Label images at t={t - 1} cannot be loaded!")
                if t == smallest_number:
                    print("Warning: stop calculation!")
                    break
                else:
                    print(f"Warning: skip volume {t - 1}!")
                    activities[t - smallest_number, :] = np.nan
                    continue

            if t == smallest_number:
                cell_num = np.max(labels_img)
                activities = np.zeros((largest_number - smallest_number + 1, cell_num))

            per = (1 - discard_ratio) * 100
            for label in range(1, cell_num + 1):
                intensity_label_i = raw[labels_img == label]
                if intensity_label_i.size == 0:
                    activities[t - smallest_number, label - 1] = np.nan
                else:
                    threshold = np.percentile(intensity_label_i, per)
                    activities[t - smallest_number, label - 1] = np.mean(
                        intensity_label_i[intensity_label_i > threshold])
        return activities


def plot_initial_matching(ref_ptrs: ndarray, tgt_ptrs: ndarray, pairs_px2: ndarray, t1: int, t2: int, fig_width_px=1800,
                          dpi=96):
    """Draws the initial matching between two sets of 3D points and their matching relationships.

    Args:
        ref_ptrs (ndarray): A 2D array of shape (n, 3) containing the reference points.
        tgt_ptrs (ndarray): A 2D array of shape (n, 3) containing the target points.
        pairs_px2 (ndarray): A 2D array of shape (m, 2) containing the pairs of matched points.
        fig_width_px (int): The width of the output figure in pixels. Default is 1200.
        dpi (int): The resolution of the output figure in dots per inch. Default is 96.

    Raises:
        AssertionError: If the inputs have invalid shapes or data types.
    """

    # Validate the inputs
    assert isinstance(ref_ptrs, ndarray) and ref_ptrs.ndim == 2 and ref_ptrs.shape[
        1] == 3, "ref_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(tgt_ptrs, ndarray) and tgt_ptrs.ndim == 2 and tgt_ptrs.shape[
        1] == 3, "tgt_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(pairs_px2, ndarray) and pairs_px2.ndim == 2 and pairs_px2.shape[
        1] == 2, "pairs_px2 should be a 2D array with shape (n, 2)"

    # Plot the scatters of the ref_points and tgt_points
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, fig_width_px, ref_ptrs, tgt_ptrs, t1, t2)

    # Plot the matching relationships between the two sets of points
    for ref_index, tgt_index in pairs_px2:
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptrs[ref_index, 1], -ref_ptrs[ref_index, 0]])
        pt2 = np.asarray([tgt_ptrs[tgt_index, 1], -tgt_ptrs[tgt_index, 0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="C1")
        ax2.add_artist(con)


def coherence_match(updated_match_matrix: ndarray, segmented_coords_norm_t1, segmented_coords_norm_t2, threshold):
    matched_pairs = greedy_match(updated_match_matrix, threshold)
    for i in range(5):
        coherence = calc_min_path(matched_pairs, segmented_coords_norm_t1, segmented_coords_norm_t2)
        updated_match_matrix = np.sqrt(updated_match_matrix * coherence)
        matched_pairs = greedy_match(updated_match_matrix, threshold)
    matched_pairs = filter_outliers(matched_pairs, segmented_coords_norm_t1, segmented_coords_norm_t2, neighbors=10)
    return matched_pairs


def hungarian_match(match_score_matrix_: ndarray, match_score_matrix_updated: ndarray, similarity_threshold: float):
    row_indices, col_indices = linear_sum_assignment(match_score_matrix_updated, maximize=True)
    match_pairs = []
    for r, c in zip(row_indices, col_indices):
        if match_score_matrix_[r, c] > similarity_threshold and match_score_matrix_updated[r, c] > similarity_threshold:
            match_pairs.append((c, r))
    return np.asarray(match_pairs)


def greedy_match(updated_match_matrix: ndarray, threshold: float = 1e-6):
    working_match_score_matrix = updated_match_matrix.copy()
    match_pairs = []
    for pair_number in range(working_match_score_matrix.shape[1]):
        max_match_score = working_match_score_matrix.max()
        if max_match_score < threshold:
            break
        target_index, reference_index = np.unravel_index(working_match_score_matrix.argmax(),
                                                         working_match_score_matrix.shape)
        match_pairs.append((reference_index, target_index))

        working_match_score_matrix[target_index, :] = -1
        working_match_score_matrix[:, reference_index] = -1
    return np.asarray(match_pairs)


def get_match_pairs(updated_match_matrix: ndarray, segmented_coords_norm_t2, segmented_coords_norm_t1,
                    threshold=0.5, method="coherence") -> ndarray:
    """Match points from two point sets by simply choosing the pairs with the highest probability subsequently"""
    if method == "greedy":
        return greedy_match(updated_match_matrix, threshold)
    if method == "hungarian":
        return hungarian_match(updated_match_matrix, updated_match_matrix, threshold)
    if method == "coherence":
        return coherence_match(updated_match_matrix, segmented_coords_norm_t1, segmented_coords_norm_t2, threshold)
    raise ValueError("method should be 'greedy', 'hungarian' or 'coherence'")


def cal_norm_prob(matched_pairs, shape):
    normalized_prob = np.full(shape, 0.1 / (shape[1] - 1))
    for ref, tgt in matched_pairs:
        normalized_prob[tgt, ref] = 0.9
    return normalized_prob


def tps_with_two_ref(matched_pairs: List[Tuple[int, int]], ptrs_tgt_mx3: ndarray, ptrs_ref_nx3: ndarray,
                     tracked_ref_lx3: ndarray, smoothing: float) -> ndarray:
    matched_pairs_array = np.asarray(matched_pairs)
    prts_ref_matched_n1x3 = ptrs_ref_nx3[matched_pairs_array[:, 0], :]
    prts_tgt_matched_m1x3 = ptrs_tgt_mx3[matched_pairs_array[:, 1], :]
    tps = RBFInterpolator(prts_ref_matched_n1x3, prts_tgt_matched_m1x3, kernel='thin_plate_spline', smoothing=smoothing)
    # Apply the TPS transformation to the source points
    return tps(tracked_ref_lx3)


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
        if dist_sqrt < 1E-3:
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


def evenly_distributed_volumes(current_vol: int, sampling_number: int) -> List[int]:
    """Get evenly distributed previous volumes"""
    interval = (current_vol - 1) // sampling_number
    start = np.mod(current_vol - 1, sampling_number) + 1
    return list(range(start, current_vol - interval + 1, interval))


def get_volumes_list(current_vol: int, skip_volumes: List[int], sampling_number: int = 20, adjacent: bool = False) -> \
        List[int]:
    if current_vol - 1 < sampling_number:
        vols_list = list(range(1, current_vol))
    else:
        if adjacent:
            vols_list = list(range(current_vol - sampling_number, current_vol))
        else:
            vols_list = evenly_distributed_volumes(current_vol, sampling_number)
    vols_list = [vol for vol in vols_list if vol not in skip_volumes]
    return vols_list
