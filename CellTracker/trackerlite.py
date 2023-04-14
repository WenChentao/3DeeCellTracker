from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from numpy import ndarray
from scipy.special import softmax
import tensorflow.keras as ks

from CellTracker.coord_image_transformer import Coordinates
from CellTracker.ffn import initial_matching_ffn, normalize_points, FFN

BETA, LAMBDA, MAX_ITERATION = (3, 3, 2000)
K_POINTS = 20


class TrackerLite:
    """
    A class that tracks cells in 3D time-lapse images using a trained FFN model.
    """

    def __init__(self, coords_dir: str, ffn_model_path: str, track_results_dir: str,
                 proofed_coords_vol1: Coordinates, miss_frame: List[int] = None):
        """
        Initialize a new instance of the TrackerLite class.

        Args:
            coords_dir: The path to the directory containing the coordinates files.
            ffn_model_path: The path to the FFN model file.
            track_results_dir: The path to the directory for saving the tracking results.
            proofed_coords_vol1: The confirmed cell positions at time step t1.
            miss_frame: A list of missing frames.
        """
        if not isinstance(coords_dir, str):
            raise TypeError(f"coords_path should be a string, but got {type(coords_dir)}")
        if not isinstance(ffn_model_path, str):
            raise TypeError(f"ffn_model_path should be a string, but got {type(ffn_model_path)}")
        if not isinstance(track_results_dir, str):
            raise TypeError(f"track_path should be a string, but got {type(track_results_dir)}")
        if miss_frame is not None and not isinstance(miss_frame, List):
            raise TypeError(f"miss_frame should be a list or None, but got {type(miss_frame)}")

        self.coords_dir = Path(coords_dir)
        self.ffn_model_path = ffn_model_path
        self.ffn_model = FFN()

        try:
            dummy_input = np.random.random((1, 122))
            _ = self.ffn_model(dummy_input)
            self.ffn_model.load_weights(ffn_model_path)
        except (OSError, ValueError) as e:
            raise ValueError(f"Failed to load the FFN model from {ffn_model_path}: {e}") from e

        self.track_results_dir = Path(track_results_dir)
        (self.track_results_dir / "figure").mkdir(parents=True, exist_ok=True)
        (self.track_results_dir / "coords").mkdir(parents=True, exist_ok=True)

        self.proofed_coords_vol1 = proofed_coords_vol1
        self.miss_frame = [] if miss_frame is None else miss_frame

    def predict_cell_positions(self, t1: int, t2: int, confirmed_coord_t1: Coordinates = None,
                               beta: float = BETA, lambda_: float = LAMBDA, save_fig: bool = False):
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

        matching_matrix = initial_matching_ffn(self.ffn_model, confirmed_coords_norm_t1, segmented_coords_norm_t2, K_POINTS)
        normalized_prob, _ = simple_match(matching_matrix)

        tracked_coords_norm_t2, _ = prgls_quick(normalized_prob, segmented_coords_norm_t2, confirmed_coords_norm_t1,
                                        beta=beta, lambda_=lambda_)
        tracked_coords_t2 =  tracked_coords_norm_t2 * scale_t1 + mean_t1

        fig = plot_prgls_prediction(confirmed_coord_t1.real, segmented_pos_t2.real, tracked_coords_t2)
        if save_fig:
            fig.savefig(self.track_results_dir / "figure"/ f"matching_{str(t2).zfill(4)}.png", facecolor='white')
            plt.close()
        return Coordinates(tracked_coords_t2,
                           interpolation_factor=self.proofed_coords_vol1.interpolation_factor,
                           voxel_size=self.proofed_coords_vol1.voxel_size,
                           dtype="real")

    def match_by_ffn(self, t1: int, t2: int, confirmed_coord_t1: Coordinates = None):
        assert t2 not in self.miss_frame
        segmented_pos_t1 = self._get_segmented_pos(t1)
        segmented_pos_t2 = self._get_segmented_pos(t2)

        if confirmed_coord_t1 is None:
            confirmed_coord_t1 = segmented_pos_t1

        confirmed_coords_norm_t1, (mean_t1, scale_t1) = normalize_points(confirmed_coord_t1.real, return_para=True)
        segmented_coords_norm_t2 = (segmented_pos_t2.real - mean_t1) / scale_t1

        matching_matrix = initial_matching_ffn(self.ffn_model, confirmed_coords_norm_t1, segmented_coords_norm_t2, K_POINTS)
        _, pairs_px2 = simple_match(matching_matrix)
        plot_initial_matching(confirmed_coord_t1.real, segmented_pos_t2.real, pairs_px2)
        #plot_initial_matching(confirmed_coord_t1.real[:,[2,1,0]], segmented_pos_t2.real[:,[2,1,0]], pairs_px2)

    def _get_segmented_pos(self, t: int) -> Coordinates:
        interp_factor = self.proofed_coords_vol1.interpolation_factor
        voxel_size = self.proofed_coords_vol1.voxel_size

        pos = Coordinates(np.load(self.coords_dir / f"coords{str(t).zfill(4)}.npy"),
                          interpolation_factor=interp_factor, voxel_size=voxel_size, dtype="raw")
        return pos


def plot_initial_matching(ref_ptrs: ndarray, tgt_ptrs: ndarray, pairs_px2: ndarray, fig_width_px=1200, dpi=96):
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
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, fig_width_px, ref_ptrs, tgt_ptrs)

    # Plot the matching relationships between the two sets of points
    for ref_index, tgt_index in pairs_px2:
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptrs[ref_index, 1], -ref_ptrs[ref_index, 0]])
        pt2 = np.asarray([tgt_ptrs[tgt_index, 1], -tgt_ptrs[tgt_index, 0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="C1")
        ax2.add_artist(con)


def plot_prgls_prediction(ref_ptrs: ndarray, tgt_ptrs: ndarray, predicted_ref_ptrs: ndarray, fig_width_px=1200, dpi=96):
    """Draws the initial matching between two sets of 3D points and their matching relationships.

    Args:
        ref_ptrs (ndarray): A 2D array of shape (n, 3) containing the positions of reference points.
        tgt_ptrs (ndarray): A 2D array of shape (n, 3) containing the positions of target points.
        predicted_ref_ptrs (ndarray): A 2D array of shape (n, 3) containing the predicted positions of reference points
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
    assert isinstance(predicted_ref_ptrs, ndarray) and predicted_ref_ptrs.ndim == 2 and predicted_ref_ptrs.shape[
        1] == 3, "predicted_ref_ptrs should be a 2D array with shape (n, 3)"

    # Plot the scatters of the ref_points and tgt_points
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, fig_width_px, ref_ptrs, tgt_ptrs)

    # Plot the matching relationships between the two sets of points
    for ref_ptr, tgt_ptr in zip(ref_ptrs, predicted_ref_ptrs):
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptr[1], -ref_ptr[ 0]])
        pt2 = np.asarray([tgt_ptr[1], -tgt_ptr[0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="C1")
        ax2.add_artist(con)

    return fig


def plot_two_pointset_scatters(dpi, fig_width_px, ref_ptrs, tgt_ptrs):
    # Calculate the figure size based on the input width and dpi
    fig_width_in = fig_width_px / dpi  # convert to inches assuming the given dpi
    fig_height_in = fig_width_in / 1.618  # set height to golden ratio
    # Determine whether to use a top-down or left-right layout based on the aspect ratio of the point sets
    ref_range_y, ref_range_x, _ = np.max(ref_ptrs, axis=0) - np.min(ref_ptrs, axis=0)
    tgt_range_y, tgt_range_x, _ = np.max(tgt_ptrs, axis=0) - np.min(tgt_ptrs, axis=0)
    top_down = ref_range_x + tgt_range_x >= ref_range_y + tgt_range_y
    # Create the figure and subplots
    if top_down:
        #print("Using top-down layout")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width_in, fig_height_in))
    else:
        #print("Using left-right layout")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in))
    # Plot the point sets on the respective subplots
    ax1.scatter(ref_ptrs[:, 1], -ref_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 1')
    ax2.scatter(tgt_ptrs[:, 1], -tgt_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 2')

    # # Set the aspect ratio to 'equal' for both subplots
    # ax1.axis("equal")
    # ax2.axis("equal")
    # # Apply tight layout to minimize whitespace in the figure
    # plt.tight_layout()

    unify_xy_lims(ax1, ax2)

    # Set plot titles or y-axis labels based on the layout
    if top_down:
        ax1.set_ylabel("Point Set t1")
        ax2.set_ylabel("Point Set t2")
    else:
        ax1.set_title("Point Set t1")
        ax2.set_title("Point Set t2")
    return ax1, ax2, fig


def unify_xy_lims(ax1, ax2):
    # Determine the shared x_lim and y_lim
    x_lim = [min(ax1.get_xlim()[0], ax2.get_xlim()[0]), max(ax1.get_xlim()[1], ax2.get_xlim()[1])]
    y_lim = [min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1])]
    # Set the same x_lim and y_lim on both axes
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)


def simple_match(initial_match_matrix: ndarray) -> ndarray:
    """Match points from two point sets by simply choosing the pairs with the highest probability subsequently"""
    match_matrix = initial_match_matrix.copy()
    pairs_list = []
    for ptr_num in range(match_matrix.shape[1]):
        max_value = match_matrix.max()
        if max_value < 0.5:
            break
        tgt_index, ref_index = np.unravel_index(match_matrix.argmax(), match_matrix.shape)
        pairs_list.append((ref_index, tgt_index))

        match_matrix[tgt_index, :] = 0
        match_matrix[:, ref_index] = 0
    pairs_px2 = np.array(pairs_list)
    normalized_prob = np.full_like(match_matrix, 0.1 / (match_matrix.shape[1] -1))
    for ref, tgt in pairs_px2:
        normalized_prob[tgt, ref] = 0.9
    return normalized_prob, pairs_px2


def prgls_quick(init_match_mxn, ptrs_tgt_mx3: ndarray, tracked_ref_nx3: ndarray,
                beta: float, lambda_: float, max_iteration: int = MAX_ITERATION) \
        -> Tuple[ndarray, ndarray]:
    """
    Get coherent movements from the initial matching by PR-GLS algorithm
    """


    # Initiate parameters
    ratio_outliers = 0.05  # This is the gamma
    distance_weights_nxn = gaussian_kernel(tracked_ref_nx3, tracked_ref_nx3, beta ** 2)  # This is the Gram matrix
    sigma_square = dist_squares(tracked_ref_nx3, ptrs_tgt_mx3).mean() / 3  # This is the sigma^2
    predicted_coord_ref_nx3 = tracked_ref_nx3.copy()  # This is the T(X)

    ############################################################################
    # iteratively update predicted_ref_n1x3, ratio_outliers, sigma_square, and posterior_mxn. Plot and save results
    ############################################################################
    for iteration in range(1, max_iteration):
        # E-step: update posterior probability P_mxn
        posterior_mxn = estimate_posterior(init_match_mxn, sigma_square, predicted_coord_ref_nx3, ptrs_tgt_mx3, ratio_outliers)

        # M-step: update predicted positions of reference set
        # movements_basis_3xn is the parameter C
        movements_basis_3xn = solve_movements_ref(sigma_square, lambda_, posterior_mxn, predicted_coord_ref_nx3, ptrs_tgt_mx3, distance_weights_nxn)
        movements_ref_nx3 = np.dot(movements_basis_3xn, distance_weights_nxn).T
        if iteration > 1:
            predicted_coord_ref_nx3 += movements_ref_nx3  # The first estimation is not reliable thus is discarded
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

    return predicted_coord_ref_nx3, posterior_mxn


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
    p_pos_j_and_j_match_i_mxn = (1 - ratio_outliers) * prior_p_mxn * p_pos_j_when_j_match_i_mxn / (2 * np.pi * initial_sigma_square) ** 1.5
    posterior_sum_m = np.sum(p_pos_j_and_j_match_i_mxn, axis=1) + ratio_outliers / vol
    posterior_mxn = p_pos_j_and_j_match_i_mxn / posterior_sum_m[:, None]
    return posterior_mxn


def softmax_normalize(similarity_matrix_mxn: ndarray) -> ndarray:
    return scipy.special.softmax(similarity_matrix_mxn, axis=1)


def row_wise_normalize(similarity_matrix_mxn: ndarray) -> ndarray:
    return similarity_matrix_mxn / np.sum(similarity_matrix_mxn, axis=1, keepdims=True)


def non_max_suppression_normalize(similarity_matrix_mxn: ndarray, threshold=0.5):
    n = similarity_matrix_mxn.shape[1]
    init_match_mxn = np.full_like(similarity_matrix_mxn, fill_value=1 / n)
    similarity_temp_mxn = similarity_matrix_mxn.copy()
    for point_in_ref in range(n):
        similarity_max = similarity_temp_mxn.max()
        if similarity_max < threshold:
            break
        max_row, max_col = np.unravel_index(similarity_temp_mxn.argmax(), similarity_temp_mxn.shape)
        init_match_mxn[max_row, :] = 0.1 / (n - 1)
        init_match_mxn[max_row, max_col] = 0.9
        similarity_temp_mxn[max_row, :] = 0
        similarity_temp_mxn[:, max_col] = 0
    return init_match_mxn


def solve_movements_ref(initial_sigma_square, lambda_, posterior_mxn, ptrs_ref_nx3, ptrs_tgt_mx3,
                        scaling_factors_nxn):
    n = ptrs_ref_nx3.shape[0]
    posterior_sum_diag_nxn = np.diag(np.sum(posterior_mxn, axis=0))
    coefficient_nxn = np.dot(scaling_factors_nxn,
                             posterior_sum_diag_nxn) + lambda_ * initial_sigma_square * np.identity(n)
    dependent_3xn = np.dot(ptrs_tgt_mx3.T, posterior_mxn) - np.dot(ptrs_ref_nx3.T, posterior_sum_diag_nxn)
    movements_ref_3xn = np.linalg.solve(coefficient_nxn.T, dependent_3xn.T).T
    return movements_ref_3xn

