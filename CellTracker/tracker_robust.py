from CellTracker.coord_image_transformer import plot_prgls_prediction, Coordinates
from CellTracker.ffn import initial_matching_ffn, normalize_points
from CellTracker.robust_match import estimate_coords_with_knn_interpolation,  \
    estimate_coords_with_rf
from CellTracker.trackerlite import TrackerLite, prgls_with_two_ref, get_match_pairs, K_POINTS, BETA, LAMBDA


class TrackerRobust(TrackerLite):
    def predict_cell_positions(self, t1: int, t2: int, confirmed_coord_t1: Coordinates = None, draw_fig: bool = False):
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

        matching_matrix = initial_matching_ffn(self.ffn_model, segmented_coords_norm_t1, segmented_coords_norm_t2,
                                               K_POINTS)
        differences_matrix = segmented_coords_norm_t2[:, None, :] - confirmed_coords_norm_t1[None, :, :]
        normalized_prob, filtered_pairs = get_match_pairs(matching_matrix, segmented_coords_norm_t2, confirmed_coords_norm_t1)
        # pred_seg_coords_t2 = estimate_coords_with_svr(segmented_coords_t1=segmented_coords_norm_t1,
        #                                               confirmed_coords_t1=segmented_coords_norm_t1,
        #                                               filtered_pairs=filtered_pairs.tolist(),
        #                                               pairwise_differences_matrix=differences_matrix)

        tracked_coords_norm_t2 = estimate_coords_with_rf(segmented_coords_t1=segmented_coords_norm_t1,
                                                          confirmed_coords_t1=confirmed_coords_norm_t1,
                                                          filtered_pairs=filtered_pairs.tolist(),
                                                          pairwise_differences_matrix=differences_matrix)
        tracked_coords_t2 = tracked_coords_norm_t2 * scale_t1 + mean_t1
        if draw_fig:
            fig = plot_prgls_prediction(confirmed_coord_t1.real, segmented_pos_t2.real, tracked_coords_t2, t1, t2)

        return Coordinates(tracked_coords_t2,
                           interpolation_factor=self.proofed_coords_vol1.interpolation_factor,
                           voxel_size=self.proofed_coords_vol1.voxel_size,
                           dtype="real")