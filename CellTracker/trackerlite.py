import re
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from numpy import ndarray
from scipy.interpolate import RBFInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.stats import trim_mean

from CellTracker.coord_image_transformer import Coordinates, plot_predicted_movements, plot_two_pointset_scatters, \
    plot_predicted_movements_one_panel, CoordsToImageTransformer
from CellTracker.v1_modules.ffn import initial_matching_ffn, FFN
from CellTracker.fpm import FlexiblePointMatcherOriginal, initial_matching_fpm
from CellTracker.robust_match import filter_matching_outliers, calc_min_path, add_or_remove_points, \
    get_full_cell_candidates
from CellTracker.stardist3dcustom import StarDist3DCustom
from CellTracker.utils import load_2d_slices_at_time, normalize_points, set_unique_xlim

FIGURE = "figure"
COORDS_REAL = "coords_real"
LABELS = "labels"
TRACK_RESULTS = "track_results"
SEG = "seg"
BETA, LAMBDA, MAX_ITERATION = (5.0, 5.0, 2000)
K_POINTS = 20


class TrackerLite:
    """
    A class that tracks cells in 3D time-lapse images using a trained FFN model.
    """

    def __init__(self, results_dir: str, match_model_name: str, proofed_coords_vol1: Coordinates,
                 coords2image: CoordsToImageTransformer, stardist_model: StarDist3DCustom,
                 similarity_threshold=0.3, match_method: str = "coherence", miss_frame: List[int] = None,
                 basedir: str = "ffn_models", model_type: str = "ffn"):
        """
        Initialize a new instance of the TrackerLite class.

        Parameters
        ----------
        results_dir:
            The path to the directory containing the results.
        match_model_name:
            The filename without extension to the model file.
        proofed_coords_vol1:
            The confirmed cell positions at time step t1.
        coords2image:
            The CoordsToImageTransformer instance.
        stardist_model:
            The StarDist3DCustom instance.
        similarity_threshold:
            The threshold for removing outlier matching between two cells from two frames.
        match_method:
            The method for matching cells. "coherence", "greedy" or "hungarian".
        miss_frame:
            A list of missing frames.
        basedir:
            The base directory of the match model.
        model_type:
            The type of the match model. "ffn" or "fpm".
        """
        if miss_frame is not None and not isinstance(miss_frame, List):
            raise TypeError(f"miss_frame should be a list or None, but got {type(miss_frame)}")

        self.coords2image = coords2image
        self.stardist_model = stardist_model

        self.results_dir = Path(results_dir)
        (self.results_dir / TRACK_RESULTS / FIGURE).mkdir(parents=True, exist_ok=True)
        (self.results_dir / TRACK_RESULTS / COORDS_REAL).mkdir(parents=True, exist_ok=True)
        (self.results_dir / TRACK_RESULTS / LABELS).mkdir(parents=True, exist_ok=True)

        self.match_method = match_method
        self.similarity_threshold = similarity_threshold

        self.match_model_path = Path(basedir) / (match_model_name + ".h5")
        if model_type == "ffn":
            self.match_model = FFN()
            dummy_input = np.random.random((1, 122))
            self.initial_matching = initial_matching_ffn
        elif model_type == "fpm":
            self.match_model = FlexiblePointMatcherOriginal()
            dummy_input = np.random.random((1, 22, 4, 2))
            self.initial_matching = initial_matching_fpm
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        try:
            _ = self.match_model(dummy_input)
            self.match_model.load_weights(str(self.match_model_path))
        except (OSError, ValueError) as e:
            raise ValueError(f"Failed to load the match model from {self.match_model_path}: {e}") from e

        self.proofed_coords_vol1 = proofed_coords_vol1
        self.miss_frame = [] if miss_frame is None else miss_frame

    def predict_cell_positions(self, t1: int, t2: int, confirmed_coord_t1: Coordinates = None,
                               match_confirmed_t1_and_seg_t1=None,
                               beta: float = BETA, lambda_: float = LAMBDA, smoothing=0,
                               post_processing: str = "prgls", filter_points: bool = True, verbosity: int = 0,
                               with_shift: bool = True, learning_rate: float = 0.5) -> Tuple[Coordinates, ndarray]:
        """
        Predicts the positions of cells in a 3D image at time step t2, based on their positions at time step t1.

        Parameters
        ----------
        t1:
            The time step t1.
        t2:
            The time step t2.
        confirmed_coord_t1:
            The confirmed cell positions at time step t1.
        match_confirmed_t1_and_seg_t1: ndarray, shape (n, 2)
            The matching between the confirmed cell positions at time step t1 and the segmentation at time step t1.
        beta:
            The beta parameter for the prgls model.
        lambda_:
            The lambda parameter for the prgls model.
        smoothing:
            The smoothing parameter for the tps model.
        post_processing:
            The post-processing method. "prgls" or "tps".
        filter_points:
            Whether to filter out the outliers points that do not have corresponding points.
        verbosity:
            The verbosity level. 0-4. 0: no figure, 1: only final matching figure, 2: initial and final matching figures,
            3: all figures during iterations only in y-x view, 4: all figures during iterations with additional z-x view.
        with_shift:
            Whether to show the t1 and t2 points in a shift way.
        learning_rate:
            The learning rate for updating the predicted points, between 0 and 1.

        Returns:
        --------
        tracked_coords_t2_pred: Coordinates
            The predicted cell positions of confirmed cells at time step t2.
        match_confirmed_t1_and_seg_t2: ndarray, shape (n, 2)
            The matching between the confirmed cell positions at time step t1 and the segmentation at time step t2.
        """
        def _plot_matching(title: str, zy_view: bool = False):
            s_ = np.s_[:, [0,2,1]] if zy_view else np.s_[:, :]
            title = title + ", zy-view" if zy_view else title
            fig = plot_matching((filtered_segmented_coords_norm_t1 * scale_t1 + mean_t1)[s_],
                                (filtered_segmented_coords_norm_t2 * scale_t1 + mean_t1)[s_],
                                updated_matched_pairs, t1, t2)
            fig.suptitle(title, fontsize=20, y=0.9)

        def _plot_move_seg(title: str, zy_view: bool = False):
            s_ = np.s_[:, [0,2,1]] if zy_view else np.s_[:, :]
            title = title + ", zy-view" if zy_view else title
            fig = plot_move(predicted_coords_set1[s_], segmented_coords_norm_t2[s_], predicted_coords_t1_to_t2[s_], t1, t2)
            fig.suptitle(title, fontsize=20, y=0.9)

        def _plot_move_final(title: str, zy_view: bool = False):
            s_ = np.s_[:, [0,2,1]] if zy_view else np.s_[:, :]
            title = title + ", zy-view" if zy_view else title
            fig = plot_move(confirmed_coord_t1.real[s_], segmented_pos_t2.real[inliers_ori[1]][s_], tracked_coords_t2[s_], t1, t2)
            fig.suptitle(title, fontsize=20, y=0.9)

        assert t2 not in self.miss_frame

        if confirmed_coord_t1 is None:
            segmented_pos_t1, inliers_t1 = self._get_segmented_pos(t1)
            confirmed_coord_t1 = segmented_pos_t1
            match_confirmed_t1_and_seg_t1 = np.arange(confirmed_coord_t1.real.shape[0]).repeat(2).reshape(-1, 2)

        segmented_pos_t2, inliers_t2 = self._get_segmented_pos(t2)

        plot_matching = plot_initial_matching if with_shift else plot_initial_matching_one_panel
        plot_move = plot_predicted_movements if with_shift else plot_predicted_movements_one_panel

        # Normalize point sets
        confirmed_coords_norm_t1, (mean_t1, scale_t1) = normalize_points(confirmed_coord_t1.real, return_para=True)
        segmented_coords_norm_t2 = (segmented_pos_t2.real - mean_t1) / scale_t1
        segmented_coords_norm_t1 = (segmented_pos_t1.real - mean_t1) / scale_t1

        filtered_segmented_coords_norm_t1 = segmented_coords_norm_t1[inliers_t1]
        filtered_segmented_coords_norm_t2 = segmented_coords_norm_t2[inliers_t2]
        predicted_coords_set1 = segmented_coords_norm_t1.copy()
        n = segmented_coords_norm_t1.shape[0]
        m = segmented_coords_norm_t2.shape[0]
        inliers_ori = (inliers_t1, inliers_t2)

        iter = 4
        for i in range(iter):
            inliers_pre = (inliers_ori[0], inliers_ori[1])
            # Generate similarity scores between the filtered segmented coords
            similarity_scores = self.initial_matching(self.match_model, filtered_segmented_coords_norm_t1,
                                                     filtered_segmented_coords_norm_t2,
                                                     K_POINTS)
            updated_similarity_scores = similarity_scores.copy()
            # Generate updated_matched_pairs, which is the indices of the matched pairs in the filtered segmented coords
            updated_matched_pairs = get_match_pairs(updated_similarity_scores, filtered_segmented_coords_norm_t1,
                                            filtered_segmented_coords_norm_t2, threshold=self.similarity_threshold,
                                            method=self.match_method)

            # Generate figures showing the matching
            if (verbosity >= 2 and i == 0) or (verbosity >= 3 and i >= 1):
                _plot_matching(f"Matching iter={i}")
            if verbosity >= 4:
                _plot_matching(f"Matching iter={i}", zy_view=True)

            # Predict the corresponding positions in t2 of all the segmented cells in t1
            match_seg_t1_seg_t2 = np.column_stack(
                (inliers_pre[0][updated_matched_pairs[:, 0]], inliers_pre[1][updated_matched_pairs[:, 1]]))

            predicted_coords_t1_to_t2 = predict_new_positions(
                match_seg_t1_seg_t2, predicted_coords_set1, predicted_coords_set1, segmented_coords_norm_t2,
                post_processing, smoothing, (m, n), beta, lambda_)

            if i == iter - 1:
                tracked_coords_norm_t2 = predicted_coords_t1_to_t2.copy()
                common_elements_seg_t1 = np.intersect1d(match_confirmed_t1_and_seg_t1[:, 1], match_seg_t1_seg_t2[:, 0])
                index_pairs = np.asarray([(np.where(match_confirmed_t1_and_seg_t1[:, 1] == element)[0][0],
                                np.where(match_seg_t1_seg_t2[:, 0] == element)[0][0]) for element in
                               common_elements_seg_t1])
                match_confirmed_t1_and_seg_t2 = np.concatenate((match_confirmed_t1_and_seg_t1[index_pairs[:,0], 0:1],
                                                                match_seg_t1_seg_t2[index_pairs[:,1], 1:2]), axis=1)
                tracked_coords_norm_t2[match_confirmed_t1_and_seg_t2[:, 0], :] = segmented_coords_norm_t2[match_confirmed_t1_and_seg_t2[:, 1], :]
                break

            if verbosity >= 3:
                _plot_move_seg(f"Predict movements after {i} iteration")
            if verbosity >= 4:
                _plot_move_seg(f"Predict movements after {i} iteration", zy_view=True)

            if filter_points:
                # Predict the corresponding positions in t1 of all the segmented cells in t2
                predicted_coords_t2_to_t1 = predict_new_positions(
                    match_seg_t1_seg_t2[:, [1, 0]], segmented_coords_norm_t2, segmented_coords_norm_t2, predicted_coords_set1,
                    post_processing, smoothing, (n, m), beta, lambda_)

                filtered_segmented_coords_norm_t1, filtered_segmented_coords_norm_t2, inliers_ori = \
                    add_or_remove_points(
                    predicted_coords_t1_to_t2, predicted_coords_t2_to_t1, predicted_coords_set1, segmented_coords_norm_t2,
                    match_seg_t1_seg_t2)

            predicted_coords_set1 = (predicted_coords_set1 + (predicted_coords_t1_to_t2 - predicted_coords_set1) * learning_rate)
            filtered_segmented_coords_norm_t1 = predicted_coords_set1[inliers_ori[0]]

        self.print_info(post_processing)

        tracked_coords_t2 = tracked_coords_norm_t2 * scale_t1 + mean_t1
        if verbosity >= 1:
            _plot_move_final(f"Final prediction")
        if verbosity >= 4:
            _plot_move_final(f"Final prediction", zy_view=True)

        tracked_coords_t2_pred = Coordinates(tracked_coords_t2, interpolation_factor=self.proofed_coords_vol1.interpolation_factor,
                                  voxel_size=self.proofed_coords_vol1.voxel_size, dtype="real")
        return tracked_coords_t2_pred, match_confirmed_t1_and_seg_t2

    def print_info(self, post_processing):
        print(f"Matching method: {self.match_method}")
        print(f"Threshold for similarity: {self.similarity_threshold}")
        print(f"Post processing method: {post_processing}")

    def predict_cell_positions_ensemble(self, skipped_volumes: List[int], t2: int, coord_t1: Coordinates,
                                        beta: float = BETA, lambda_: float = LAMBDA, sampling_number: int = 20,
                                        adjacent: bool = False, post_processing: str = "prgls", verbosity: int = 0):
        coord_prgls = []
        for t1 in get_volumes_list(current_vol=t2, skip_volumes=skipped_volumes, sampling_number=sampling_number,
                                   adjacent=adjacent):
            loaded_coord_t1 = np.load(
                str(self.results_dir / TRACK_RESULTS / COORDS_REAL / f"coords{str(t1).zfill(4)}.npy"))
            loaded_coord_t1_ = Coordinates(loaded_coord_t1, coord_t1.interpolation_factor, coord_t1.voxel_size,
                                           dtype="real")
            coord_prgls.append(self.predict_cell_positions(t1=t1, t2=t2, confirmed_coord_t1=loaded_coord_t1_, beta=beta,
                                                           lambda_=lambda_, post_processing=post_processing,
                                                           verbosity=verbosity).real)
        return Coordinates(trim_mean(coord_prgls, 0.1, axis=0),
                           interpolation_factor=self.proofed_coords_vol1.interpolation_factor,
                           voxel_size=self.proofed_coords_vol1.voxel_size, dtype="real")

    def match_by_ffn(self, t1: int, t2: int, confirmed_coord_t1: Coordinates = None):
        print(f"Matching method: {self.match_method}")
        print(f"Threshold for similarity: {self.similarity_threshold}")
        assert t2 not in self.miss_frame
        segmented_pos_t1, extra_pos_t1 = self._get_segmented_pos(t1)
        segmented_pos_t2, extra_pos_t2 = self._get_segmented_pos(t2)

        if confirmed_coord_t1 is None:
            confirmed_coord_t1 = segmented_pos_t1

        pairs_px2 = self.match_two_point_sets(confirmed_coord_t1, segmented_pos_t2)
        fig = plot_initial_matching(confirmed_coord_t1.real, segmented_pos_t2.real, pairs_px2, t1, t2)

    def match_two_point_sets(self, confirmed_coord_t1, segmented_pos_t2):
        confirmed_coords_norm_t1, (mean_t1, scale_t1) = normalize_points(confirmed_coord_t1.real, return_para=True)
        segmented_coords_norm_t2 = (segmented_pos_t2.real - mean_t1) / scale_t1
        initial_matching = self.initial_matching(self.match_model, confirmed_coords_norm_t1, segmented_coords_norm_t2,
                                                K_POINTS)
        updated_matching = initial_matching.copy()
        pairs_px2 = get_match_pairs(updated_matching, confirmed_coords_norm_t1, segmented_coords_norm_t2,
                                    threshold=self.similarity_threshold, method=self.match_method)
        return pairs_px2

    def _get_segmented_pos(self, t: int) -> Tuple[Coordinates, ndarray]:
        """Get segmented positions and extra positions from stardist model"""
        interp_factor = self.proofed_coords_vol1.interpolation_factor
        voxel_size = self.proofed_coords_vol1.voxel_size

        coordinates_stardist = np.load(str(self.results_dir / SEG / f"coords{str(t).zfill(4)}.npy"))
        prob_map = self.coords2image.load_prob_map(self.stardist_model.config.grid, t)
        extra_coordinates = get_full_cell_candidates(coordinates_stardist, prob_map)
        combined_coordinates = np.concatenate((coordinates_stardist, extra_coordinates), axis=0)
        pos = Coordinates(combined_coordinates, interpolation_factor=interp_factor, voxel_size=voxel_size, dtype="raw")
        inliers = np.arange(len(coordinates_stardist))
        return pos, inliers

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


def predict_new_positions(matched_pairs, confirmed_coords_norm_t1, segmented_coords_norm_t1, segmented_coords_norm_t2,
                          post_processing, smoothing: float, similarity_scores_shape: Tuple[int, int], beta=BETA, lambda_=LAMBDA):
    if post_processing == "tps":
        tracked_coords_norm_t2 = tps_with_two_ref(matched_pairs, segmented_coords_norm_t2,
                                                  segmented_coords_norm_t1, confirmed_coords_norm_t1, smoothing)
    elif post_processing == "prgls":
        normalized_prob = cal_norm_prob(matched_pairs, similarity_scores_shape)
        tracked_coords_norm_t2, _ = prgls_with_two_ref(normalized_prob, segmented_coords_norm_t2,
                                                       segmented_coords_norm_t1, confirmed_coords_norm_t1,
                                                       beta=beta, lambda_=lambda_)
    else:
        raise ValueError("post_prossing should be either 'tps' or 'prgls'")
    return tracked_coords_norm_t2


def plot_initial_matching(ref_ptrs: ndarray, tgt_ptrs: ndarray, pairs_px2: ndarray, t1: int, t2: int, fig_width_px=1800,
                          dpi=96, ids_ref=None, ids_tgt=None, show_3d: bool = False):
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
    if show_3d:
        fig = plot_matching_3d_plotly(ref_ptrs, tgt_ptrs, pairs_px2, ids_ref, ids_tgt)
        return fig
    # Plot the scatters of the ref_points and tgt_points
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, fig_width_px, ref_ptrs, tgt_ptrs, t1, t2, ids_ref, ids_tgt)

    # Plot the matching relationships between the two sets of points
    for ref_index, tgt_index in pairs_px2:
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptrs[ref_index, 1], -ref_ptrs[ref_index, 0]])
        pt2 = np.asarray([tgt_ptrs[tgt_index, 1], -tgt_ptrs[tgt_index, 0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="C1")
        ax2.add_artist(con)
    # ax1.axis('equal')
    # ax2.axis('equal')
    set_unique_xlim(ax1, ax2)
    plt.pause(0.1)
    return fig


def plot_matching_3d(points_t1, points_t2, pairs, ids_t1, ids_t2):
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection='3d')

    min_val = np.min([points_t1.min(), points_t2.min()])
    max_val = np.max([points_t1.max(), points_t2.max()])

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_zlim([min_val, max_val])

    for i, point in enumerate(points_t1):
        ax.scatter3D(*point, color='blue', s=50)
        ax.text(*point, str(ids_t1[i]), color='blue')

    for i, point in enumerate(points_t2):
        ax.scatter3D(*point, color='red', s=50)
        ax.text(*point, str(ids_t2[i]), color='red')

    ax.scatter3D(points_t1[:, 0], points_t1[:, 1], points_t1[:, 2], color='blue', s=50)

    ax.scatter3D(points_t2[:, 0], points_t2[:, 1], points_t2[:, 2], color='red', s=50)

    for start, end in pairs:
        ax.plot3D(*zip(points_t1[start], points_t1[end]), color='green')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_matching_3d_plotly(points_t1, points_t2, pairs, ids_t1, ids_t2):
    import plotly.graph_objects as go
    trace1 = go.Scatter3d(
        x=-points_t1[:, 0],
        y=-points_t1[:, 1],
        z=points_t1[:, 2],
        mode='markers+text',
        text=ids_t1,
        marker=dict(
            size=3,
            line=dict(
                color='blue',
                width=0.5
            ),
            opacity=0.8
        ),
        textposition='bottom center'
    )

    trace2 = go.Scatter3d(
        x=-points_t2[:, 0],
        y=-points_t2[:, 1],
        z=points_t2[:, 2],
        mode='markers+text',
        text=ids_t2,
        marker=dict(
            size=3,
            line=dict(
                color='red',
                width=0.5
            ),
            opacity=0.8
        ),
        textposition='bottom center'
    )

    traces = [trace1, trace2]

    # 绘制移动路径
    for start, end in pairs:
        traces.append(
            go.Scatter3d(
                x=[-points_t1[start, 0], -points_t2[end, 0]],
                y=[-points_t1[start, 1], -points_t2[end, 1]],
                z=[points_t1[start, 2], points_t2[end, 2]],
                mode='lines',
                line=dict(
                    color='green',
                    width=2
                ),
                showlegend=False
            )
        )

    layout = go.Layout(
        autosize=False,
        width=1600,
        height=800,
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=0, y=0, z=1)
            )
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_initial_matching_one_panel(ref_ptrs: ndarray, tgt_ptrs: ndarray, pairs_px2: ndarray, t1: int, t2: int, fig_width_px=1800,
                          dpi=96, ids_ref=None, ids_tgt=None):
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
    # Plot the scatters of the ref_points and tgt_points
    fig_width_in = fig_width_px / dpi  # convert to inches assuming the given dpi
    fig_height_in = fig_width_in / 1.618  # set height to golden ratio
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    plt.scatter(ref_ptrs[:, 1], -ref_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 1')
    plt.scatter(tgt_ptrs[:, 1], -tgt_ptrs[:, 0], marker="x", facecolors='r', edgecolors='r', label='Set 2')

    # Plot the matching relationships between the two sets of points
    for ref_index, tgt_index in pairs_px2:
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptrs[ref_index, 1], -ref_ptrs[ref_index, 0]])
        pt2 = np.asarray([tgt_ptrs[tgt_index, 1], -tgt_ptrs[tgt_index, 0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], lw=1, color="C1")
    plt.axis('equal')
    return fig


def coherence_match(updated_match_matrix: ndarray, segmented_coords_norm_t1, segmented_coords_norm_t2, threshold):
    matched_pairs = greedy_match(updated_match_matrix, threshold)
    for i in range(5):
        coherence = calc_min_path(matched_pairs, segmented_coords_norm_t1, segmented_coords_norm_t2)
        updated_match_matrix = np.sqrt(updated_match_matrix * coherence)
        matched_pairs = greedy_match(updated_match_matrix, threshold)
    matched_pairs = filter_matching_outliers(matched_pairs, segmented_coords_norm_t1, segmented_coords_norm_t2, neighbors=10)
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


def get_match_pairs(updated_match_matrix: ndarray, segmented_coords_norm_t1, segmented_coords_norm_t2, threshold=0.5,
                    method="coherence") -> ndarray:
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
