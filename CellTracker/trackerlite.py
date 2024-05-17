import os
import pickle
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from numpy import ndarray
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from CellTracker.coord_image_transformer import Coordinates, CoordsToImageTransformer
from CellTracker.global_match import _match_fpm_prgls, pairwise_pointsets_distances, rigid_transform, \
    ensemble_match_initial_20_volumes
from CellTracker.plot import plot_initial_matching, plot_initial_matching_one_panel, plot_predicted_movements, \
    plot_predicted_movements_one_panel, plot_pairs_and_movements
from CellTracker.simple_alignment import get_match_pairs, K_POINTS, align_by_control_points, greedy_match
from CellTracker.test_matching_models import cal_norm_prob, prgls_with_two_ref, affine_align_by_fpm, predict_matching_prgls
from CellTracker.v1_modules.ffn import initial_matching_ffn
from CellTracker.fpm import initial_matching_fpm
from CellTracker.robust_match import add_or_remove_points, get_extra_cell_candidates
from CellTracker.stardist3dcustom import StarDist3DCustom
from CellTracker.utils import load_2d_slices_at_time, normalize_points, load_2d_slices_at_time_quick, del_datasets
from CellTracker.test_matching_models import BETA, LAMBDA


class TrackerLite:
    """
    A class that tracks cells in 3D time-lapse images using a trained FFN model.
    """
    def __init__(self, results_dir: str, images_path: dict, match_model,
                 coords2image: CoordsToImageTransformer, stardist_model: StarDist3DCustom, model_type: str="fpm",
                 similarity_threshold=0.3, match_method: str = "coherence",  miss_frame: List[int] = None):
        """
        Initialize a new instance of the TrackerLite class.

        Parameters
        ----------
        results_dir:
            The path to the directory containing the results.
        match_model:
            The model for matching.
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
        model_type:
            The type of the match model. "ffn" or "fpm".
        """
        self.common_ids_in_coords = None
        self.common_ids_in_proof = None
        self.scale_vol1 = None
        self.mean_vol1 = None
        self.rotation_matrix = None
        self.images_path = images_path
        if miss_frame is not None and not isinstance(miss_frame, List):
            raise TypeError(f"miss_frame should be a list or None, but got {type(miss_frame)}")

        self.inv_tform_tx3x4 = None
        self.tform_tx3x4 = None
        self.corresponding_coords_txnx3 = None

        self.coords2image = coords2image
        self.stardist_model = stardist_model

        self.results_dir = Path(results_dir)

        self.match_method = match_method
        self.similarity_threshold = similarity_threshold

        self.match_model = match_model
        if model_type == "ffn":
            self.get_similarity = initial_matching_ffn
        elif model_type == "fpm":
            self.get_similarity = initial_matching_fpm
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.miss_frame = [] if miss_frame is None else miss_frame

    def get_cell_num_initial_20_vols(self):
        x = 20
        cell_num_list = []
        for i in tqdm(range(x)):
            t = self.tgt_list_20[i]
            pos_ref0, _ = self._get_segmented_pos(t)
            cell_num_list.append(len(pos_ref0.real))
        return cell_num_list

    @property
    def tgt_list_20(self):
        with open(str(self.results_dir / 'reference_target_vols_list.pkl'), 'rb') as file:
            reference_target_vols_list = pickle.load(file)
        tgt_list = [reference_target_vols_list[0][0][0]] + [tgt for _, tgt in reference_target_vols_list[:20 - 1]]
        return tgt_list

    def match_manual_ref0(self):
        """Remove cells whose centers are not included in the stardist predictions (including the weak cells)"""
        pos_ref0, inliers = self._get_segmented_pos(self.coords2image.proofed_coords_vol)
        dist_pxr = cdist(self.coords2image.coord_proof.real, pos_ref0.real)
        dist_rxr = cdist(pos_ref0.real, pos_ref0.real)
        np.fill_diagonal(dist_rxr, np.inf)
        threshold = dist_rxr.min()
        matched_ind = []
        matched_ind_in_proof = []
        for pair_number in range(dist_pxr.shape[0]):
            min_dist = dist_pxr.min()
            if min_dist > threshold:
                break
            proof_index, ref0_index = np.unravel_index(dist_pxr.argmin(), dist_pxr.shape)
            matched_ind.append(ref0_index)
            matched_ind_in_proof.append(proof_index)

            dist_pxr[proof_index, :] = np.inf
            dist_pxr[:, ref0_index] = np.inf
        return np.asarray(matched_ind), np.asarray(matched_ind_in_proof)

    def filter_proofed_cells(self, threshold: int = 15):
        """Remove cells that were not detected by stardist or the ones not exist in most of the other intial_x_volumes"""
        vol_num = 20
        segmented_pos_t1, inliers_t1 = self._get_segmented_pos(self.center_point)
        cum_match_counts=np.zeros((segmented_pos_t1.real.shape[0]), dtype=int)
        for i in tqdm(range(1, vol_num)):
            with h5py.File(str(Path(self.results_dir) / "matches_1_to_20.h5"), "r") as f:
                pairs = f[f"1_{i+1}"][:]
            for ref, _ in pairs:
                cum_match_counts[ref] += 1
        self.common_ids_in_coords, self.common_ids_in_proof = self.get_common_ids(cum_match_counts, threshold)
        print(f"Choose {len(self.common_ids_in_coords)} cells from {segmented_pos_t1.real.shape[0]} proofed cells, "
              f"with matches in {threshold * 100 // vol_num}% volumes of initial {vol_num} volumes")
        np.save(str(Path(self.results_dir) / "common_ids_vol_initial.npy"), self.common_ids_in_coords)

        self.coords2image.update_filtered_segmentation(self.common_ids_in_proof + 1, self.images_path, self.center_point)
        cell_num_list = self.get_cell_num_initial_20_vols()
        self.pairs_t1_to_t20_list = ensemble_match_initial_20_volumes(cell_num_list, str(Path(self.results_dir) / "matches_1_to_20.h5"))

    def draw_matching_common_cells(self, t_target: int, reference_target_vols_list: list):
        assert t_target < len(reference_target_vols_list)
        t_reference = reference_target_vols_list[0][0][0]
        coords_subset_norm_t1, segmented_coords_norm_t1, segmented_pos_t1, subset_t1 = self.load_rotated_normalized_coords(t_reference)
        t2 = reference_target_vols_list[t_target - 1][1]
        coords_subset_norm_t2, segmented_coords_norm_t2, segmented_pos_t2, subset_t2 = self.load_rotated_normalized_coords(t2)
        _, pairs = self.predict_cell_positions(t_reference, t2)
        ref_ptrs_confirmed = segmented_coords_norm_t1[self.common_ids_in_coords]
        segmented_coords_norm_t1, segmented_coords_norm_t2, ref_ptrs_confirmed = rotate_for_visualization(self.rotation_matrix,
                        (segmented_coords_norm_t1, segmented_coords_norm_t2, ref_ptrs_confirmed))
        fig = plot_initial_matching(segmented_coords_norm_t1, segmented_coords_norm_t2, pairs, t_reference, t2,
                                    ref_ptrs_confirmed=ref_ptrs_confirmed)

    def get_common_ids(self, cum_match_counts, threshold: int):
        matched_ind_in_ref, matched_ind_in_proof = self.match_manual_ref0()
        top_indices = np.nonzero(cum_match_counts>=threshold)
        _, ind_ref, _ = np.intersect1d(matched_ind_in_ref, top_indices, return_indices=True)
        return matched_ind_in_ref[ind_ref], matched_ind_in_proof[ind_ref]

    def rigid_alignment(self, fpm_model_rot=None, t_ref_start: int = 2, arrangement: str = "vertical"):
        """
        Calculate the parameters for rigid alignment between vol#1 and all other volumes, and the aligned coordinates.
        Store the results into a h5 file
        """
        t_tgt = 1
        rotation_matrix_to_t0 = self.cal_rotation_matrix_xyplane(t_initial=t_tgt)

        alignment_folder = Path(self.results_dir) / "alignment"
        alignment_folder.mkdir(exist_ok=True, parents=True)

        f_seg = h5py.File(str(Path(self.results_dir) / "seg.h5"), "r")
        t = f_seg["prob"].shape[0]

        coords_subset_norm_t1, mean_t1, scale_t1, segmented_coords_norm_t1, segmented_pos_t1, subset_t1 = self.load_normalized_coords(
            t_tgt)

        rigid_aligned_coords_txnx3 = np.full((t, *coords_subset_norm_t1.shape), np.nan)
        tform_tx3x4 = np.zeros((t, 3, 4))
        tform_inv_tx3x4 = np.zeros((t, 3, 4))

        rigid_aligned_coords_txnx3[0, ...] = coords_subset_norm_t1.copy()
        tform_tx3x4[0, :, :3] = np.eye(3)
        tform_inv_tx3x4[0, :, :3] = np.eye(3)

        path_rigid = str(Path(self.results_dir) / "rotation_alignment.h5")

        plt.ioff()

        with h5py.File(path_rigid, "a") as rigid_align_file:
            del_datasets(rigid_align_file, ["rigid_aligned_coords", "tform", "tform_inv"])
            rigid_align_file.create_dataset("rigid_aligned_coords", data=rigid_aligned_coords_txnx3)
            rigid_align_file.create_dataset("tform", data=tform_tx3x4)
            rigid_align_file.create_dataset("tform_inv", data=tform_inv_tx3x4)
            print(f"Initial rotation alignment results was saved in {path_rigid}")

            for t_ref in tqdm(range(t_ref_start, t+1)):
                coords_subset_norm_t2_pre, _, _, segmented_coords_norm_t2_pre, segmented_pos_t2_pre, subset_t2_pre = self.load_normalized_coords(
                    t_ref - 1, mean=mean_t1, scale=scale_t1)
                coords_subset_norm_t2, _, _, segmented_coords_norm_t2, segmented_pos_t2, subset_t2 = self.load_normalized_coords(
                    t_ref, mean=mean_t1, scale=scale_t1)
                matched_pairs_t2_t2pre = _match_fpm_prgls(self.match_model, fpm_model_rot, coords_subset_norm_t2, coords_subset_norm_t2_pre) #bottleneck
                if t_ref == 2:
                    updated_pairs_t2_t1 = matched_pairs_t2_t2pre.copy()
                else:
                    matched_pairs_t2pre_t1 = rigid_align_file[f't_{t_ref - 1:06d}.npy'][:]
                    updated_pairs_t2_t1 = link_pairs(matched_pairs_t2pre_t1, matched_pairs_t2_t2pre)

                rigid_aligned_ref_coords, tform = align_by_control_points(coords_subset_norm_t2, coords_subset_norm_t1,
                                                                          updated_pairs_t2_t1,
                                                                          method="euclidean")
                n, m = rigid_aligned_ref_coords.shape[0], coords_subset_norm_t1.shape[0]
                _, similarity_scores = predict_matching_prgls(updated_pairs_t2_t1,
                                                              rigid_aligned_ref_coords,
                                                              rigid_aligned_ref_coords,
                                                              coords_subset_norm_t1,
                                                              (m, n), beta=BETA, lambda_=LAMBDA)
                updated_pairs_t2_t1 = greedy_match(similarity_scores, threshold=0.4)
                rigid_aligned_ref_coords, tform = align_by_control_points(coords_subset_norm_t2, coords_subset_norm_t1,
                                                                          updated_pairs_t2_t1,
                                                                          method="euclidean")

                if f't_{t_ref:06d}.npy' in rigid_align_file:
                    del rigid_align_file[f't_{t_ref:06d}.npy']
                rigid_align_file.create_dataset(f't_{t_ref:06d}.npy', data=updated_pairs_t2_t1)

                for id_ref, id_tgt in updated_pairs_t2_t1:
                    rigid_aligned_coords_txnx3[t_ref - 1, id_tgt, :] = rigid_aligned_ref_coords[id_ref, :]
                tform_tx3x4[t_ref - 1, ...] = tform.params[:3, :]
                tform_inv_tx3x4[t_ref - 1, ...] = np.linalg.inv(tform.params)[:3, :]

                coords_subset_norm_t1_, coords_subset_norm_t2_, rigid_aligned_ref_coords_ = rotate_for_visualization(
                    rotation_matrix_to_t0,(coords_subset_norm_t1, coords_subset_norm_t2, rigid_aligned_ref_coords))

                fig1 = self._plot_matching(f"Match (raw)", coords_subset_norm_t1_, coords_subset_norm_t2_,
                                    scale_t1, mean_t1,  updated_pairs_t2_t1[:, [1, 0]], t_tgt, t_ref, plot_initial_matching,display=False)
                fig1.savefig('./figure1.png', dpi=90, facecolor='white')
                plt.close(fig1)
                fig2 = self._plot_matching(f"Matching (aligned)", coords_subset_norm_t1_, rigid_aligned_ref_coords_,
                                    scale_t1, mean_t1,  updated_pairs_t2_t1[:, [1, 0]], t_tgt, t_ref, plot_initial_matching, display=False)
                fig2.savefig('./figure2.png', dpi=90, facecolor='white')
                plt.close(fig2)
                combine_two_png(t=t_ref, alignment_folder=alignment_folder, arrangement=arrangement)

                rigid_align_file["rigid_aligned_coords"][t_ref - 1, :, :] = rigid_aligned_coords_txnx3[t_ref - 1]
                rigid_align_file["tform"][t_ref - 1, :, :] = tform.params[:3, :]
                rigid_align_file["tform_inv"][t_ref - 1, :, :] = np.linalg.inv(tform.params)[:3, :]

        plt.ion()

    def match_first_20_volumes(self):
        with open(str(self.results_dir / 'reference_target_vols_list.pkl'), 'rb') as file:
            reference_target_vols_list = pickle.load(file)

        num = 20
        tgt_list = [reference_target_vols_list[0][0][0]] + [tgt for _, tgt in reference_target_vols_list[:num-1]]
        with h5py.File(str(Path(self.results_dir) / f"matches_1_to_{num}.h5"), "w") as f:
            for i in tqdm(range(num)):
                t1 = tgt_list[i]
                for j in range(num):
                    if i == j:
                        continue
                    t2 = tgt_list[j]
                    _, pairs = self.predict_cell_positions(t1, t2)
                    f.create_dataset(f"{i + 1}_{j + 1}", data=pairs)

    def save_optimized_matches_in_first_20_volumes(self, max_repetition=2):
        """Predict the cell positions, and then save the predicted positions and pairs"""
        with open(str(self.results_dir / 'reference_target_vols_list.pkl'), 'rb') as file:
            reference_target_vols_list = pickle.load(file)

        t1 = reference_target_vols_list[0][0][0] # initial volume of the confirmed segmentation
        result_path  = Path(self.results_dir) / "matchings"
        result_path.mkdir(exist_ok=True, parents=True)
        plt.ioff()
        for t_target in tqdm(range(1, 20)):
            t2 = reference_target_vols_list[t_target - 1][1]
            coords_subset_norm_t1, segmented_coords_norm_t1, segmented_pos_t1, subset_t1 = self.load_rotated_normalized_coords(t1)
            coords_subset_norm_t2, segmented_coords_norm_t2, segmented_pos_t2, subset_t2 = self.load_rotated_normalized_coords(t2)
            pairs = self.pairs_t1_to_t20_list[t_target]

            ref_ptrs_confirmed, ref_ptrs_tracked_t2 = self.predict_pos_all_cells(pairs, segmented_coords_norm_t1,
                                                                                 segmented_coords_norm_t2)

            confirmed_coord, corrected_labels_image = self.finetune_positions_in_image(ref_ptrs_tracked_t2, t2, max_repetition=max_repetition)

            segmented_coords_norm_t1, segmented_coords_norm_t2, ref_ptrs_confirmed, _ref_ptrs_tracked_t2 = (
                rotate_for_visualization(self.rotation_matrix,
                            (segmented_coords_norm_t1, segmented_coords_norm_t2, ref_ptrs_confirmed, ref_ptrs_tracked_t2)))
            fig = plot_pairs_and_movements(segmented_coords_norm_t1, segmented_coords_norm_t2, t1, t2,
                                        ref_ptrs_confirmed, _ref_ptrs_tracked_t2, display_fig=False, show_ids=False)
            self.save_tracking_results(confirmed_coord, corrected_labels_image, t=t2, images_path=self.images_path)
            fig.savefig(str(result_path/ f"matching_t{t2:06d}.png"), dpi=90, facecolor='white')
            plt.close(fig)

        plt.ion()

    def finetune_positions_in_image(self, ref_ptrs_tracked_t2, t2, max_repetition: int):
        tracked_coords_t2 = rigid_transform(self.inv_tform_tx3x4[t2 - 1, ...],
                                            ref_ptrs_tracked_t2) * self.scale_vol1 + self.mean_vol1
        tracked_coords_t2_pred = Coordinates(tracked_coords_t2,
                                             interpolation_factor=self.coords2image.interpolation_factor,
                                             voxel_size=self.coords2image.image_pars["voxel_size_yxz"], dtype="real")
        confirmed_coord, corrected_labels_image = self.coords2image.accurate_correction(t2,
                                                                                        self.stardist_model.config.grid,
                                                                                        tracked_coords_t2_pred,
                                                                                        ensemble=True, max_repetition=max_repetition)
        return confirmed_coord, corrected_labels_image

    def predict_pos_all_cells(self, pairs, segmented_coords_norm_t1, segmented_coords_norm_t2):
        # Find these cells that were not matched in previous procedures
        missed_cells = np.setdiff1d(self.common_ids_in_coords, pairs[:, 0], assume_unique=True)
        missed_cells, common_ind_miss, pairs_ind_miss = np.intersect1d(self.common_ids_in_coords, missed_cells,
                                                                       return_indices=True)
        intersect_cells, common_ind_intersect, pairs_ind_intersect = np.intersect1d(self.common_ids_in_coords,
                                                                                    pairs[:, 0], return_indices=True)
        ref_ptrs_confirmed = segmented_coords_norm_t1[self.common_ids_in_coords]

        # Assign positions for these matched cells
        ref_ptrs_tracked_t2 = np.zeros_like(ref_ptrs_confirmed)
        for ref, id1, id2 in zip(intersect_cells, common_ind_intersect, pairs_ind_intersect):
            ref_ptrs_tracked_t2[id1, :] = segmented_coords_norm_t2[pairs[id2, 1], :]

        # Assign postions for the unmatched cells with KNN predictions
        if len(missed_cells) > 0:
            ref_ptrs_tracked_t2[common_ind_miss] = predict_new_positions_knn(pairs, segmented_coords_norm_t1,
                                                                             segmented_coords_norm_t2,
                                                                             segmented_coords_norm_t1[missed_cells, :])
        return ref_ptrs_confirmed, ref_ptrs_tracked_t2

    def track_vols_after20(self, num_ensemble: int = 1, restart: int = 0, max_repetition=2):
        with open(str(self.results_dir / 'reference_target_vols_list.pkl'), 'rb') as file:
            reference_target_vols_list = pickle.load(file)

        with h5py.File(str(self.results_dir / "matching_results.h5"), "a") as f:
            for i, pairs in enumerate(self.pairs_t1_to_t20_list[1:]):
                _, tgt = reference_target_vols_list[i]
                if f"pairs_initial_to_{tgt}" in f:
                    del f[f"pairs_initial_to_{tgt}"]
                f.create_dataset(f"pairs_initial_to_{tgt}", data=pairs)

        t0 = reference_target_vols_list[0][0][0]
        coords_subset_norm_t0, segmented_coords_norm_t0, segmented_pos_t0, subset_t0 = self.load_rotated_normalized_coords(t0)
        num_cells_t0 = segmented_pos_t0.real.shape[0]

        plt.ioff()
        for i in tqdm(range(19 + restart, len(reference_target_vols_list))):
            ref_list, tgt = reference_target_vols_list[i]
            coords_subset_norm_tgt, segmented_coords_norm_tgt, segmented_pos_tgt, subset_tgt = self.load_rotated_normalized_coords(tgt)
            num_cells_tgt = segmented_pos_tgt.real.shape[0]
            matches_matrix_ini_to_tgt = np.zeros((num_cells_t0, num_cells_tgt), dtype=int)

            # Calculate the matching by FPM + PRGLS + Ensemble votes
            for ref in ref_list[:num_ensemble]:
                _, pairs_ref_tgt = self.predict_cell_positions(ref, tgt)
                if t0 == ref:
                    pairs_t0_ref = None
                else:
                    with h5py.File(str(self.results_dir / "matching_results.h5"), "r") as f:
                        pairs_t0_ref = f[f"pairs_initial_to_{ref}"][:]
                matches_matrix_ini_to_tgt = combine_links(pairs_t0_ref, pairs_ref_tgt, matches_matrix_ini_to_tgt)
            pairs_t0_tgt = matrix2pairs(matches_matrix_ini_to_tgt, num_cells_t0, num_ensemble)

            # Re-link from initial volume to target volume and save the matched pairs
            _, similarity_scores = predict_matching_prgls(pairs_t0_tgt,
                                                          segmented_coords_norm_t0,
                                                          segmented_coords_norm_t0,
                                                          segmented_coords_norm_tgt,
                                                          (num_cells_tgt, num_cells_t0), beta=BETA, lambda_=LAMBDA)
            updated_pairs_t0_tgt = greedy_match(similarity_scores, threshold=0.4)
            with h5py.File(str(self.results_dir / "matching_results.h5"), "a") as f:
                if f"pairs_initial_to_{tgt}" in f:
                    del f[f"pairs_initial_to_{tgt}"]
                f.create_dataset(f"pairs_initial_to_{tgt}", data=updated_pairs_t0_tgt)

            # Predict positions of all confirmed cells
            confirmed_cells_pos_t0, confirmed_cells_pos_tgt = self.predict_pos_all_cells(updated_pairs_t0_tgt, segmented_coords_norm_t0,
                                                                                 segmented_coords_norm_tgt)
            # Finetune the cell positions in the raw image
            confirmed_cells_finetuned_pos_tgt, finetuned_labels_image = self.finetune_positions_in_image(confirmed_cells_pos_tgt, tgt,
                                                                                                         max_repetition=max_repetition)

            _segmented_coords_norm_t0, _segmented_coords_norm_tgt, _confirmed_cells_pos_t0, _confirmed_cells_pos_tgt = \
                rotate_for_visualization(self.rotation_matrix,
                    (segmented_coords_norm_t0, segmented_coords_norm_tgt, confirmed_cells_pos_t0, confirmed_cells_pos_tgt))
            fig = plot_pairs_and_movements(_segmented_coords_norm_t0, _segmented_coords_norm_tgt, t0, tgt,
                                           _confirmed_cells_pos_t0, _confirmed_cells_pos_tgt, display_fig=False, show_ids=False)
            self.save_tracking_results(confirmed_cells_finetuned_pos_tgt, finetuned_labels_image, t=tgt, images_path=self.images_path)
            fig.savefig(str(Path(self.results_dir) / "matchings" / f"matching_t{tgt:06d}.png"), dpi=90, facecolor='white')
            plt.close(fig)
        plt.ion()

    def _num_all_cells(self, t):
        with h5py.File(str(self.results_dir / 'coords.h5'), 'r') as f:
            num_inliers_t0 = f[f'combined_coords_{str(t - 1).zfill(6)}'].shape[0]
        return num_inliers_t0

    def save_tracking_results(self, coords: Coordinates, corrected_labels_image: ndarray, t: int,
                              images_path: dict):
        """
        Save the tracking results, including coordinates, corrected labels image, and the merged image + label

        Parameters
        ----------
        coords : Coordinates
            The corrected coordinates of cell centers.
        corrected_labels_image : ndarray
            The corrected labels image.
        t : int
            The time point of the tracked image.
        images_path : dict
            The path to the raw image.
        """
        with h5py.File(str(self.coords2image.results_folder / 'tracking_results.h5'), 'a') as f:
            f["tracked_labels"][t - 1, :, 0, :, :] = corrected_labels_image.transpose((2, 0, 1))
            f["tracked_coordinates"][t - 1, :, :] = coords.real

        raw_img = load_2d_slices_at_time(images_path, t=t, channel_name="channel_nuclei")
        self.coords2image.save_merged_labels(corrected_labels_image, raw_img, t, self.coords2image.cmap_colors)

    def pairwise_pointsets_distances(self):
        self.corresponding_coords_txnx3, self.tform_tx3x4, self.inv_tform_tx3x4 = self.load_rotation_alignment()

        self.distances_txt = pairwise_pointsets_distances(self.corresponding_coords_txnx3)
        self.mean_coords_nx3 = np.nanmean(self.corresponding_coords_txnx3, axis=0)

        t, n, _ = self.corresponding_coords_txnx3.shape
        dist_t = np.zeros((t))
        for i in range(n):
            dist_i_t = cdist(self.corresponding_coords_txnx3[:, i, :], self.mean_coords_nx3[i:i+1, :], metric='euclidean')[:, 0]
            # Replaces any NaN values in the distance matrix with the mean of the non-NaN values
            dist_i_t[np.isnan(dist_i_t)] = np.nanmean(dist_i_t)
            dist_t += dist_i_t
        self.center_point = np.argmin(dist_t) + 1
        return self.distances_txt, self.center_point

    def load_rotation_alignment(self):
        with h5py.File(str(Path(self.results_dir) / "rotation_alignment.h5"), "r") as f:
            corresponding_coords_txnx3 = f["rigid_aligned_coords"][:]
            tform_tx3x4 = f["tform"][:]
            inv_tform_tx3x4 = f["tform_inv"][:]
        return corresponding_coords_txnx3, tform_tx3x4, inv_tform_tx3x4

    def rigid_align(self, points_nx3: ndarray, t):
        return rigid_transform(self.tform_tx3x4[t-1], points_nx3)

    def inv_rigid_align(self, points_nx3: ndarray, t):
        return rigid_transform(self.inv_tform_tx3x4[t-1], points_nx3)

    def load_normalized_coords(self, t: int, mean: float = None, scale: float = None):
        segmented_pos, subset = self._get_segmented_pos(t)
        if mean is None or scale is None:
            segmented_coords_norm, (mean, scale) = normalize_points(segmented_pos.real, return_para=True)
        else:
            segmented_coords_norm = (segmented_pos.real - mean) / scale
        coords_subset_norm = segmented_coords_norm[subset]
        return coords_subset_norm, mean, scale, segmented_coords_norm, segmented_pos, subset

    def cal_rotation_matrix_xyplane(self, t_initial: int):
        segmented_pos, subset = self._get_segmented_pos(t_initial)
        points = segmented_pos.real[subset][:, :2]

        pca = PCA(n_components=1)
        pca.fit(points)
        pc1 = pca.components_[0]

        angle = np.arccos(pc1[0] / np.linalg.norm(pc1))

        if pc1[1] < 0:
            angle = -angle

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        rotation_matrix = rotation_matrix.dot(np.asarray([[0, -1], [1, 0]]))
        return rotation_matrix

    def load_rotated_normalized_coords(self, t: int, t0: int = 1):
        if self.corresponding_coords_txnx3 is None:
            self.corresponding_coords_txnx3, self.tform_tx3x4, self.inv_tform_tx3x4 = self.load_rotation_alignment()

        if self.mean_vol1 is None:
            _, self.mean_vol1, self.scale_vol1, _, _, _ = self.load_normalized_coords(t0)

        _, _, _, segmented_coords_norm, segmented_pos, subset = self.load_normalized_coords(
            t, mean=self.mean_vol1, scale=self.scale_vol1)
        segmented_coords_norm = rigid_transform(self.tform_tx3x4[t-1,...], segmented_coords_norm)
        coords_subset_norm = segmented_coords_norm[subset]
        return coords_subset_norm, segmented_coords_norm, segmented_pos, subset

    def predict_cell_positions(self, t1: int, t2: int, confirmed_coord_t1: ndarray = None, subset_confirmed = None,
                               beta: float = BETA, lambda_: float = LAMBDA, filter_points: bool = True, verbosity: int = 0,
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
        filter_points:
            Whether to filter out the outliers points that do not have corresponding points.
        verbosity:
            The verbosity level. 0-4. 0: no figure, 1: only final matching figure, 2: initial and final matching figures,
            3: all figures during iterations only in y-x view, 4: all figures during iterations with additional z-x view.
        with_shift:
            Whether to show the t1 and t2 points in a shift way.
        learning_rate:
            The learning rate for updating the predicted points, between 0 and 1.

        Returns
        --------
        tracked_coords_t2_pred: Coordinates
            The predicted cell positions of confirmed cells at time step t2.
        pairs_seg_t1_seg_t2: ndarray, shape (n, 2)
            The matching between the segmentation at time step t1 and the segmentation at time step t2, including these "weak" cells
        """
        post_processing = "prgls"
        smoothing = 0  # not used in "prgls"

        assert t2 not in self.miss_frame
        plot_matching = plot_initial_matching if with_shift else plot_initial_matching_one_panel
        plot_move = plot_predicted_movements if with_shift else plot_predicted_movements_one_panel

        # Load normalized coordinates at t1 and t2: segmented_pos is un-rotated, while other coords are rotated
        coords_subset_norm_t1, segmented_coords_norm_t1, segmented_pos_t1, subset_t1 = self.load_rotated_normalized_coords(t1)
        coords_subset_norm_t2, segmented_coords_norm_t2, segmented_pos_t2, subset_t2 = self.load_rotated_normalized_coords(t2)
        subset = (subset_t1, subset_t2)

        if confirmed_coord_t1 is None:
            confirmed_coords_norm_t1 = coords_subset_norm_t1.copy()
            subset_confirmed = subset_t1
        else:
            confirmed_coords_norm_t1 = (confirmed_coord_t1 - self.mean_vol1) / self.scale_vol1

        n, m = segmented_coords_norm_t1.shape[0], segmented_coords_norm_t2.shape[0]
        aligned_coords_subset_norm_t1, coords_subset_norm_t2, _, affine_tform = affine_align_by_fpm(self.match_model,
                                                                          coords_norm_t1=coords_subset_norm_t1,
                                                                          coords_norm_t2=coords_subset_norm_t2)
        aligned_segmented_coords_norm_t1 = affine_tform(segmented_coords_norm_t1)
        aligned_confirmed_coords_norm_t1 = affine_tform(confirmed_coords_norm_t1)
        moved_seg_coords_t1 = aligned_segmented_coords_norm_t1.copy()

        iter = 3
        for i in range(iter):
            _matched_pairs_subset = self.initial_matching(aligned_coords_subset_norm_t1, coords_subset_norm_t2,
                                                          self.mean_vol1, self.scale_vol1, t1, t2, plot_matching,
                                                          verbosity, i)
            matched_pairs = np.column_stack(
                (subset[0][_matched_pairs_subset[:, 0]], subset[1][_matched_pairs_subset[:, 1]]))

            if i == iter - 1:
                tracked_coords_t1_to_t2, posterior_mxn = predict_new_positions(
                    matched_pairs, aligned_confirmed_coords_norm_t1, aligned_segmented_coords_norm_t1,
                    segmented_coords_norm_t2,
                    post_processing, smoothing, (m, n), beta, lambda_)

                tracked_coords_norm_t2 = tracked_coords_t1_to_t2.copy()
                pairs_seg_t1_seg_t2 = greedy_match(posterior_mxn, threshold=0.5)
                pairs_in_confirmed_subset = np.asarray(
                    [(np.nonzero(subset_confirmed==i)[0][0], j) for i, j in pairs_seg_t1_seg_t2 if i in subset_confirmed])
                tracked_coords_norm_t2[pairs_in_confirmed_subset[:, 0], :] = segmented_coords_norm_t2[pairs_in_confirmed_subset[:, 1], :]
            else:
                # Predict the corresponding positions in t2 of all the segmented cells in t1
                predicted_coords_t1_to_t2, posterior_mxn = predict_new_positions(
                    matched_pairs, aligned_segmented_coords_norm_t1, aligned_segmented_coords_norm_t1,
                    segmented_coords_norm_t2,
                    post_processing, smoothing, (m, n), beta, lambda_)

                moved_seg_coords_t1_rot, segmented_coords_norm_t2_rot, predicted_coords_t1_to_t2_rot = rotate_for_visualization(
                    self.rotation_matrix, (moved_seg_coords_t1, segmented_coords_norm_t2 , predicted_coords_t1_to_t2))
                if verbosity >= 3:
                    self._plot_move_seg(f"Predict movements after {i} iteration", moved_seg_coords_t1_rot,
                                        segmented_coords_norm_t2_rot,
                                        predicted_coords_t1_to_t2_rot, self.scale_vol1, self.mean_vol1, t1, t2, plot_move)
                if verbosity >= 4:
                    self._plot_move_seg(f"Predict movements after {i} iteration", moved_seg_coords_t1_rot,
                                        segmented_coords_norm_t2_rot,
                                        predicted_coords_t1_to_t2_rot, self.scale_vol1, self.mean_vol1, t1, t2, plot_move, zy_view=True)

                if filter_points:
                    # Predict the corresponding positions in t1 of all the segmented cells in t2
                    predicted_coords_t2_to_t1, _ = predict_new_positions(
                        matched_pairs[:, [1, 0]], segmented_coords_norm_t2, segmented_coords_norm_t2, aligned_segmented_coords_norm_t1,
                        post_processing, smoothing, (n, m), beta, lambda_)

                    aligned_coords_subset_norm_t1, coords_subset_norm_t2, subset = \
                        add_or_remove_points(
                            predicted_coords_t1_to_t2, predicted_coords_t2_to_t1,
                            aligned_segmented_coords_norm_t1, segmented_coords_norm_t2,
                            matched_pairs)

                moved_seg_coords_t1 += (predicted_coords_t1_to_t2 - moved_seg_coords_t1) * learning_rate
                aligned_coords_subset_norm_t1 = moved_seg_coords_t1[subset[0]]

        tracked_coords_t2 = tracked_coords_norm_t2 * self.scale_vol1 + self.mean_vol1

        segmented_coords_norm_t1_rotated, segmented_coords_norm_t2_rot = rotate_for_visualization(self.rotation_matrix,
                        (segmented_coords_norm_t1, segmented_coords_norm_t2))
        if verbosity >= 1:
            self.print_info(post_processing)
            self._plot_matching_final(f"Matching iter={i}",
                                      segmented_coords_norm_t1_rotated * self.scale_vol1 + self.mean_vol1,
                                      segmented_coords_norm_t2_rot * self.scale_vol1 + self.mean_vol1, pairs_seg_t1_seg_t2, t1, t2,
                                      plot_matching)
            self._plot_matching_final(f"Matching iter={i}",
                                segmented_pos_t1.real, segmented_pos_t2.real, pairs_seg_t1_seg_t2, t1, t2, plot_matching)
        if verbosity >= 4:
            self._plot_matching_final(f"Matching iter={i}",
                                      segmented_pos_t1.real, segmented_pos_t2.real, pairs_seg_t1_seg_t2, t1, t2,
                                      plot_matching, zy_view=True)

        tracked_coords_t2_pred = Coordinates(tracked_coords_t2,
                                             interpolation_factor=self.coords2image.interpolation_factor,
                                             voxel_size=self.coords2image.image_pars["voxel_size_yxz"], dtype="real")
        return tracked_coords_t2_pred, pairs_seg_t1_seg_t2

    def initial_matching(self, filtered_segmented_coords_norm_t1, filtered_segmented_coords_norm_t2,
                         mean_t2, scale_t2, t1, t2, plot_matching, verbosity, i):
        # Generate similarity scores between the filtered segmented coords
        similarity_scores = self.get_similarity(self.match_model, filtered_segmented_coords_norm_t1,
                                                filtered_segmented_coords_norm_t2, K_POINTS)
        # Generate matched_pairs, which is the indices of the matched pairs in the filtered segmented coords
        matched_pairs = get_match_pairs(similarity_scores, filtered_segmented_coords_norm_t1,
                                        filtered_segmented_coords_norm_t2, threshold=self.similarity_threshold,
                                        method=self.match_method)

        filtered_segmented_coords_norm_t1_rotated, filtered_segmented_coords_norm_t2_rotated = rotate_for_visualization(
            self.rotation_matrix,(filtered_segmented_coords_norm_t1, filtered_segmented_coords_norm_t2))

        # Generate figures showing the matching
        if (verbosity >= 2 and i == 0) or (verbosity >= 3 and i >= 1):
            fig = self._plot_matching(f"Matching iter={i}", filtered_segmented_coords_norm_t1_rotated,
                                      filtered_segmented_coords_norm_t2_rotated, scale_t2, mean_t2, matched_pairs, t1, t2, plot_matching)
        if verbosity >= 4:
            fig = self._plot_matching(f"Matching iter={i}", filtered_segmented_coords_norm_t1_rotated,
                                      filtered_segmented_coords_norm_t2_rotated, scale_t2, mean_t2, matched_pairs, t1, t2, plot_matching,
                                      zy_view=True)
        return matched_pairs

    @staticmethod
    def _plot_matching(title: str, coords_subset_t1, coords_subset_t2, scale_t2, mean_t2,
                       _matched_pairs_subset, t1, t2, plot_matching,
                       zy_view: bool = False, display=True):
        s_ = np.s_[:, [0,2,1]] if zy_view else np.s_[:, :]
        title = title + ", zy-view" if zy_view else title
        fig = plot_matching(ref_ptrs=(coords_subset_t1 * scale_t2 + mean_t2)[s_],
                            tgt_ptrs=(coords_subset_t2 * scale_t2 + mean_t2)[s_],
                            pairs_px2=_matched_pairs_subset, t1=t1, t2=t2, display_fig=display)
        fig.suptitle(title, fontsize=20, y=0.99)
        return fig

    @staticmethod
    def _plot_matching_final(title: str, coords_t1, coords_t2,
                             _matched_pairs_subset, t1, t2, plot_matching,
                             zy_view: bool = False):
        s_ = np.s_[:, [0,2,1]] if zy_view else np.s_[:, :]
        title = title + ", zy-view" if zy_view else title
        fig = plot_matching(ref_ptrs=coords_t1[s_],
                            tgt_ptrs=coords_t2[s_],
                            pairs_px2=_matched_pairs_subset,
                            t1=t1, t2=t2)
        fig.suptitle(title, fontsize=20, y=0.9)

    @staticmethod
    def _plot_move_seg(title: str, moved_coords_t1, segmented_coords_norm_t2, predicted_coords_t1_to_t2,
                       scale_t2, mean_t2, t1, t2, plot_move, zy_view: bool = False):
        s_ = np.s_[:, [0,2,1]] if zy_view else np.s_[:, :]
        title = title + ", zy-view" if zy_view else title
        fig, _ = plot_move(ref_ptrs=moved_coords_t1[s_] * scale_t2 + mean_t2,
                        tgt_ptrs=segmented_coords_norm_t2[s_] * scale_t2 + mean_t2,
                        predicted_ref_ptrs=predicted_coords_t1_to_t2[s_] * scale_t2 + mean_t2,
                        t1=t1, t2=t2)
        fig.suptitle(title, fontsize=20, y=0.9)

    @staticmethod
    def _plot_move_final(title: str, confirmed_coord_t1, segmented_pos_t2, tracked_coords_t2, subset, t1, t2, plot_move, zy_view: bool = False):
        s_ = np.s_[:, [0,2,1]] if zy_view else np.s_[:, :]
        title = title + ", zy-view" if zy_view else title
        fig, _ = plot_move(
            ref_ptrs=confirmed_coord_t1.real[s_],
            tgt_ptrs=segmented_pos_t2.real[subset[1]][s_],
            predicted_ref_ptrs=tracked_coords_t2[s_],
            t1=t1, t2=t2
        )
        fig.suptitle(title, fontsize=20, y=0.9)

    def print_info(self, post_processing):
        print(f"Matching method: {self.match_method}")
        print(f"Threshold for similarity: {self.similarity_threshold}")
        print(f"Post processing method: {post_processing}")

    def match_by_nn(self, t1: int, t2: int, match_model=None, match_method=None, top_down=True):
        if match_method is None:
            print(f"Matching method: {self.match_method}")
        else:
            print(f"Matching method: {match_method}")
        print(f"Threshold for similarity: {self.similarity_threshold}")
        assert t2 not in self.miss_frame
        segmented_pos_t1, inliers_t1 = self._get_segmented_pos(t1)
        segmented_pos_t2, inliers_t2 = self._get_segmented_pos(t2)

        coords_subset_t1 = segmented_pos_t1.real[inliers_t1]
        coords_subset_t2 = segmented_pos_t2.real[inliers_t2]

        pairs_px2 = self.match_two_point_sets(coords_subset_t1, coords_subset_t2, match_model=match_model, match_method=match_method)

        fig = plot_initial_matching(coords_subset_t1, coords_subset_t2, pairs_px2, t1, t2, top_down=top_down)
        return pairs_px2

    def match_two_point_sets(self, confirmed_coord_t1, segmented_pos_t2, match_model=None, match_method=None):
        confirmed_coords_norm_t1, (mean_t1, scale_t1) = normalize_points(confirmed_coord_t1.real, return_para=True)
        segmented_coords_norm_t2 = (segmented_pos_t2.real - mean_t1) / scale_t1
        if match_model is None:
            match_model = self.match_model
        if match_method is None:
            match_method = self.match_method
        initial_matching = self.get_similarity(match_model, confirmed_coords_norm_t1, segmented_coords_norm_t2,
                                               K_POINTS)
        updated_matching = initial_matching.copy()
        pairs_px2 = get_match_pairs(updated_matching, confirmed_coords_norm_t1, segmented_coords_norm_t2,
                                    threshold=self.similarity_threshold, method=match_method)
        return pairs_px2

    def _get_segmented_pos(self, t: int) -> Tuple[Coordinates, ndarray]:
        """Get segmented positions and extra positions from stardist model"""
        interp_factor = self.coords2image.interpolation_factor
        voxel_size = self.coords2image.voxel_size
        image_size_yxz = self.coords2image.image_pars["image_size_yxz"]

        if (self.coords2image.results_folder / 'coords.h5').exists():
            with h5py.File(str(self.coords2image.results_folder / 'coords.h5'), 'r') as f:
                combined_coordinates = f[f'combined_coords_{str(t-1).zfill(6)}'][:]
                num_inliers = f[f'combined_coords_{str(t-1).zfill(6)}'].attrs["num"]
        else:
            with h5py.File(str(self.results_dir / "seg.h5"), "r") as seg_file:
                coordinates_stardist = seg_file[f'coords_{str(t-1).zfill(6)}'][:]
                prob_map = self.coords2image.load_prob_map(self.stardist_model.config.grid, t - 1, seg_file, image_size_yxz)
            extra_coordinates = get_extra_cell_candidates(coordinates_stardist, prob_map)
            if extra_coordinates.shape[0] != 0:
                combined_coordinates = np.concatenate((coordinates_stardist, extra_coordinates), axis=0)
            else:
                combined_coordinates = coordinates_stardist
            num_inliers = len(coordinates_stardist)

        pos = Coordinates(combined_coordinates, interpolation_factor=interp_factor, voxel_size=voxel_size, dtype="raw")
        inliers = np.arange(num_inliers)
        return pos, inliers

    def combine_weak_cells(self):
        image_size_yxz = self.coords2image.image_pars["image_size_yxz"]

        with h5py.File(str(self.results_dir / "seg.h5"), "r") as seg_file, \
              h5py.File(str(self.coords2image.results_folder / 'coords.h5'), 'a') as coords_file:
            vol_num = seg_file["prob"].shape[0]
            for t in tqdm(range(vol_num)):
                coordinates_stardist = seg_file[f'coords_{str(t).zfill(6)}'][:]
                prob_map = self.coords2image.load_prob_map(self.stardist_model.config.grid, t, seg_file, image_size_yxz)
                extra_coordinates = get_extra_cell_candidates(coordinates_stardist, prob_map) # bottleneck
                if extra_coordinates.shape[0] != 0:
                    combined_coordinates = np.concatenate((coordinates_stardist, extra_coordinates), axis=0)
                else:
                    combined_coordinates = coordinates_stardist
                num = len(coordinates_stardist)

                dset = coords_file.create_dataset(f'combined_coords_{str(t).zfill(6)}', data=combined_coordinates)
                dset.attrs["num"] = num

    def activities(self, discard_ratio: float = 0.1):
        with h5py.File(str(self.results_dir / "seg.h5"), "r") as seg_file:
            vol_num = seg_file["prob"].shape[0]
        file_extension = os.path.splitext(self.images_path["h5_file"])[1]
        assert file_extension in [".h5", ".hdf5", ".nwb"], "Currently only TIFF sequences or HDF5/NWB dataset are supported"

        per = (1 - discard_ratio) * 100

        with h5py.File(str(self.results_dir / "tracking_results.h5"), "a") as track_file, \
                        h5py.File(self.images_path["h5_file"], 'r') as f_raw:
            track_file.attrs["t_initial"] = self.center_point
            track_file.attrs["voxel_size_yxz"] = self.coords2image.image_pars["voxel_size_yxz"]
            track_file.attrs["raw_dset"] = self.images_path["dset"]
            track_file.attrs["raw_channel_nuclei"] = self.images_path["channel_nuclei"]
            track_file.attrs["raw_channel_activity"] = self.images_path["channel_activity"]

            cell_num = np.max(track_file["tracked_labels"][self.center_point - 1, :, 0, :, :])
            activities_txn = np.zeros((vol_num, cell_num))
            coords_txnx3 = np.zeros((vol_num, cell_num, 3))
            for t in tqdm(range(1, vol_num+1)):
                try:
                    # Load 2D slices at time t
                    raw = load_2d_slices_at_time_quick(self.images_path, f_raw, t=t,
                                                       channel_name="channel_activity", do_normalize=False)
                except FileNotFoundError:
                    # Handle missing image files
                    print(f"Warning: Raw images at t={t - 1} cannot be loaded! Stop calculation!")
                    break

                # Load 2D slices at time t
                labels_img = track_file["tracked_labels"][t-1, :, 0, :, :]
                coords_txnx3[t - 1, :] = np.asarray(ndimage.measurements.center_of_mass(labels_img > 0, labels_img, range(1, cell_num + 1)))

                found_bbox = ndimage.find_objects(labels_img, max_label=cell_num)
                for label in range(1, cell_num + 1):
                    bbox = found_bbox[label - 1]
                    if found_bbox[label - 1] is not None:
                        intensity_label_i = raw[bbox][labels_img[bbox] == label]
                        threshold = np.percentile(intensity_label_i, per)
                        activities_txn[t - 1, label - 1] = np.mean(
                            intensity_label_i[intensity_label_i > threshold])
                    else:
                        activities_txn[t - 1, label - 1] = np.nan

            del_datasets(track_file, ["activities_txn", "coords_txnx3"])
            track_file.create_dataset("activities_txn", data=activities_txn)
            track_file.create_dataset("coords_txnx3", data=coords_txnx3)

        return activities_txn

    def visualize_ensemble_sampling(self, points_tx2: ndarray, max_num_refs: int = 20, width: int = 1, start_t=0):
        with open(str(self.results_dir / 'reference_target_vols_list.pkl'), 'rb') as file:
            reference_target_vols_list = pickle.load(file)

        visualize_folder = Path(self.results_dir) / "ensemble_visualize"
        visualize_folder.mkdir(exist_ok=True, parents=True)

        self.corresponding_coords_txnx3, self.tform_tx3x4, self.inv_tform_tx3x4 = self.load_rotation_alignment()
        self.rotation_matrix = self.cal_rotation_matrix_xyplane(t_initial=self.center_point)
        self.aligned_corresponding_coords_txnx3 = np.matmul(self.corresponding_coords_txnx3[:, :, :2], self.rotation_matrix)

        tgt_list = np.asarray([self.center_point] + [id for _, id in reference_target_vols_list])
        t = points_tx2.shape[0]

        fig = plt.figure(figsize=(12 + 4 * width, 11))
        main_gs = gridspec.GridSpec(1, 2, width_ratios=[3, width], figure=fig)

        ax1 = fig.add_subplot(main_gs[0, 0])
        ncols = 2 if max_num_refs > 10 else 1
        nrows = int(np.ceil(max_num_refs/2)) if max_num_refs > 10 else max_num_refs
        right_gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=main_gs[0, 1])

        ax_list = []
        for i in range(nrows):
            ax = fig.add_subplot(right_gs[i, 0])
            ax_list.append(ax)

        if ncols == 2:
            for i in range(nrows):
                ax = fig.add_subplot(right_gs[i, 1])
                ax_list.append(ax)

        bbox = ax_list[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        aspect_ratio = bbox.width / bbox.height

        xmin = np.nanmin(self.aligned_corresponding_coords_txnx3[..., 1])
        xmax = np.nanmax(self.aligned_corresponding_coords_txnx3[..., 1])
        ymin = np.nanmin(self.aligned_corresponding_coords_txnx3[..., 0])
        ymax = np.nanmax(self.aligned_corresponding_coords_txnx3[..., 0])

        if (xmax - xmin) > aspect_ratio * (ymax - ymin):
            y_mid = (ymax + ymin) / 2
            ymax = y_mid + (xmax - xmin) / (2 * aspect_ratio)
            ymin = y_mid - (xmax - xmin) / (2 * aspect_ratio)
        else:
            x_mid = (xmax + xmin) / 2
            xmax = x_mid + (ymax - ymin) * aspect_ratio / 2
            xmin = x_mid - (ymax - ymin) * aspect_ratio / 2

        for frame in tqdm(range(start_t, t - 1)):
            refs, tgt = reference_target_vols_list[frame]

            ax_list[0].set_title(f'tgt = {tgt}', c="r")

            ax1.scatter(points_tx2[:, 0], points_tx2[:, 1], c="grey")
            ax1.scatter(points_tx2[tgt_list[:frame + 1] - 1, 0], points_tx2[tgt_list[:frame + 1] - 1, 1], c="orange")
            ax1.scatter(points_tx2[refs - 1, 0], points_tx2[refs - 1, 1], c="b")
            ax1.scatter(points_tx2[tgt - 1, 0], points_tx2[tgt - 1, 1], c="r")
            ax1.set_title('MDS visualization of the point sets')

            for i in range(max_num_refs):
                if i + 1 > len(refs):
                    ax_list[i].set_xticklabels([])
                    ax_list[i].set_yticklabels([])
                    continue
                pts = self.aligned_corresponding_coords_txnx3[refs[i] - 1, ...]
                pts_2 = self.aligned_corresponding_coords_txnx3[tgt - 1, ...]
                ax_list[i].scatter(pts[:, 1], pts[:, 0], c="b", s=5)
                ax_list[i].scatter(pts_2[:, 1], pts_2[:, 0], c="r", s=5)
                ax_list[i].set_ylim(ymin, ymax)
                ax_list[i].set_xlim(xmin, xmax)
                ax_list[i].invert_yaxis()
                ax_list[i].set_xticklabels([])
                ax_list[i].set_yticklabels([])
                ax_list[i].text(1, 1, f'{refs[i]}', transform=ax_list[i].transAxes, ha='right', va='top', c="b")

            plt.tight_layout()
            fig.savefig(str(visualize_folder / f'{frame:06d}.png'), dpi=90, facecolor='white')
            ax1.cla()
            for ax in ax_list:
                ax.cla()
        plt.close(fig)

    def save_subregions(self):
        with h5py.File(str(self.results_dir / "tracking_results.h5"), "a") as track_file:
            track_file.attrs["num_cells"] = len(self.coords2image.updated_subregions)
            group = track_file.create_group("subregions")
            for i, (slices_xyz, subregion) in enumerate(self.coords2image.updated_subregions):
                if f"subregion_{i+1}" in track_file:
                    del track_file[f"subregion_{i+1}"]
                if f"subregion_{i+1}" in group:
                    del group[f"subregion_{i+1}"]
                dset = group.create_dataset(f"subregion_{i+1}", data = subregion)
                dset.attrs["slice_xyz"] = (slices_xyz[0].start, slices_xyz[0].stop,
                                           slices_xyz[1].start, slices_xyz[1].stop,
                                           slices_xyz[2].start, slices_xyz[2].stop)

    def cache_max_projections(self):
        # Cache max projections of raw image
        try:
            with h5py.File(self.images_path["h5_file"], 'r+') as f_raw:
                if "max_projection_raw" not in f_raw:
                    t, z, c, y, x = f_raw[self.images_path["dset"]].shape
                    dtype = f_raw[self.images_path["dset"]].dtype
                    max_activity_dset = f_raw.create_dataset("max_projection_raw",
                                                         (t, c, y + z * self.coords2image.interpolation_factor, x),
                                                         chunks=(1, c, y + z * self.coords2image.interpolation_factor, x),
                                                         compression="gzip", dtype=dtype, compression_opts=1)

                    print("Calulating max projection of raw images...")
                    for _t in tqdm(range(t)):
                        raw_activity = f_raw[self.images_path["dset"]][_t, :, :, :, :].transpose((1, 2, 0, 3))
                        max_activity_dset[_t, ...] = np.concatenate((raw_activity.max(axis=2),
                                                                     np.repeat(raw_activity.max(axis=1),
                                                                               self.coords2image.interpolation_factor, axis=1)),
                                                                    axis=1)
        except Exception as e:
            print(f"errors occurred: {e}")
            with h5py.File(self.images_path["dset"], 'a') as f_raw:
                if "max_projection_raw" in f_raw:
                    del f_raw["max_projection_raw"]
                    print(f"dataset max_projection_raw has been deleted")
            raise

        # Cache max projections of tracked labels
        try:
            with h5py.File(str(self.results_dir / "tracking_results.h5"), "r+") as track_file:
                if "max_projection_labels" not in track_file:
                    t, z, c, y, x = track_file["tracked_labels"].shape
                    dtype = track_file["tracked_labels"].dtype
                    max_labels_dset = track_file.create_dataset("max_projection_labels",
                                                         (t, y + z * self.coords2image.interpolation_factor, x),
                                                         chunks=(1, y + z * self.coords2image.interpolation_factor, x),
                                                         compression="gzip", dtype=dtype, compression_opts=1)

                    print("Calulating max projection of tracked labels...")
                    for _t in tqdm(range(t)):
                        labels = track_file["tracked_labels"][_t, :, 0, :, :].transpose((1, 0, 2))
                        max_labels_dset[_t, ...] = np.concatenate((labels.max(axis=1),
                                                                     np.repeat(labels.max(axis=0),
                                                                               self.coords2image.interpolation_factor, axis=0)),
                                                                    axis=0)
        except Exception as e:
            print(f"errors occurred: {e}")
            with h5py.File(str(self.results_dir / "tracking_results.h5"), "r+") as track_file:
                if "max_projection_labels" in track_file:
                    del track_file["max_projection_labels"]
                    print(f"dataset max_projection_labels has been deleted")
            raise

def predict_new_positions(matched_pairs, confirmed_coords_norm_t1, segmented_coords_norm_t1, segmented_coords_norm_t2,
                          post_processing, smoothing: float, similarity_scores_shape: Tuple[int, int], beta=BETA, lambda_=LAMBDA):
    posterior_mxn = None
    if post_processing == "tps":
        tracked_coords_norm_t2 = tps_with_two_ref(matched_pairs, segmented_coords_norm_t2,
                                                  segmented_coords_norm_t1, confirmed_coords_norm_t1, smoothing)
    elif post_processing == "prgls":
        tracked_coords_norm_t2, posterior_mxn = predict_by_prgls(matched_pairs, confirmed_coords_norm_t1,
                                                                 segmented_coords_norm_t1, segmented_coords_norm_t2,
                                                                 similarity_scores_shape, beta, lambda_)
    else:
        raise ValueError("post_prossing should be either 'tps' or 'prgls'")
    return tracked_coords_norm_t2, posterior_mxn


def predict_by_prgls(matched_pairs, confirmed_coords_norm_t1, segmented_coords_norm_t1, segmented_coords_norm_t2,
                     similarity_scores_shape, beta, lambda_):
    normalized_prob = cal_norm_prob(matched_pairs, similarity_scores_shape)
    tracked_coords_norm_t2, posterior_mxn = prgls_with_two_ref(normalized_prob, segmented_coords_norm_t2,
                                                               segmented_coords_norm_t1, confirmed_coords_norm_t1,
                                                               beta=beta, lambda_=lambda_)
    return tracked_coords_norm_t2, posterior_mxn


def predict_new_positions_knn(pairs_lx2: ndarray, coords_t1: ndarray, coords_t2: ndarray, confirmed_coords_t1: ndarray,
                              n_neighbors: int = 5):
    src = coords_t1[pairs_lx2[:, 0]]
    dst = coords_t2[pairs_lx2[:, 1]]
    tracked_coords_t1 = np.zeros_like(confirmed_coords_t1)
    from sklearn.neighbors import NearestNeighbors

    knn_model = NearestNeighbors(n_neighbors=n_neighbors).fit(src)
    neighbors_inds = knn_model.kneighbors(confirmed_coords_t1, return_distance=False)
    for i, inds in enumerate(neighbors_inds):
        _src = src[inds, :]
        _dst = dst[inds, :]
        mean_mov = np.mean(_dst - _src, axis=0)
        tracked_coords_t1[i, :] = confirmed_coords_t1[i, :] + mean_mov
    return tracked_coords_t1


def tps_with_two_ref(matched_pairs: List[Tuple[int, int]], ptrs_tgt_mx3: ndarray, ptrs_ref_nx3: ndarray,
                     tracked_ref_lx3: ndarray, smoothing: float) -> ndarray:
    matched_pairs_array = np.asarray(matched_pairs)
    prts_ref_matched_n1x3 = ptrs_ref_nx3[matched_pairs_array[:, 0], :]
    prts_tgt_matched_m1x3 = ptrs_tgt_mx3[matched_pairs_array[:, 1], :]
    tps = RBFInterpolator(prts_ref_matched_n1x3, prts_tgt_matched_m1x3, kernel='thin_plate_spline', smoothing=smoothing)
    # Apply the TPS transformation to the source points
    return tps(tracked_ref_lx3)


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


def link_pairs(matched_pairs_t2pre_t1, matched_pairs_t2_t2pre):
    pairs_t2_t1 = []
    for mid, ref in matched_pairs_t2pre_t1:
        if mid in matched_pairs_t2_t2pre[:, 1]:
            index = np.nonzero(matched_pairs_t2_t2pre[:, 1]==mid)[0][0]
            pairs_t2_t1.append((matched_pairs_t2_t2pre[index, 0], ref))
    return np.asarray(pairs_t2_t1)


def rotate_for_visualization(rotation_matrix, coords_norm: Tuple[ndarray]):
    coords_norm_rotated = [coords_i.copy() for coords_i in coords_norm]
    if rotation_matrix is not None:
        for i in range(len(coords_norm_rotated)):
            coords_norm_rotated[i][:, :2] = coords_norm_rotated[i][:, :2].dot(rotation_matrix)
    return coords_norm_rotated


def combine_two_png(t: int, alignment_folder: Path, arrangement="vertical"):
    from PIL import Image

    image1 = Image.open('./figure1.png')
    image2 = Image.open('./figure2.png')

    if arrangement=="vertical":
        width = max(image1.width, image2.width)
        height = image1.height + image2.height

        new_image = Image.new('RGB', (width, height))

        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, image1.height))
    else:
        width = image1.width + image2.width
        height = max(image1.height, image2.height)

        new_image = Image.new('RGB', (width, height))

        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))

    new_image.save(str(alignment_folder / f't_{t:06d}.png'))


def combine_links(pairs_t0_ref: ndarray, pairs_ref_tgt: ndarray, matches_matrix_ini_to_tgt: ndarray):
    if pairs_t0_ref is None:
        for ref, tgt in pairs_ref_tgt:
            matches_matrix_ini_to_tgt[ref, tgt] += 1
        return matches_matrix_ini_to_tgt

    common_mid_values, pos1, pos2 = np.intersect1d(pairs_t0_ref[:, 1], pairs_ref_tgt[:, 0], return_indices=True)
    for value, p1, p2 in zip(common_mid_values, pos1, pos2):
        matches_matrix_ini_to_tgt[pairs_t0_ref[p1, 0], pairs_ref_tgt[p2, 1]] += 1
    return matches_matrix_ini_to_tgt


def matrix2pairs(matches_matrix_ini_to_tgt, num_cells_t0, num_ensemble):
    pairs_t0_tgt = []
    for i in range(num_cells_t0):
        ref, tgt = np.unravel_index(np.argmax(matches_matrix_ini_to_tgt, axis=None),
                                    matches_matrix_ini_to_tgt.shape)
        if matches_matrix_ini_to_tgt[ref, tgt] < int(np.ceil(num_ensemble * 0.75)):
            break
        pairs_t0_tgt.append((ref, tgt))
        matches_matrix_ini_to_tgt[ref, :] = 0
        matches_matrix_ini_to_tgt[:, tgt] = 0
    pairs_t0_tgt = np.asarray(pairs_t0_tgt)
    return pairs_t0_tgt
