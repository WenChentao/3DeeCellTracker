from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple
import gc

from tensorflow.python.keras import backend as K
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm

from CellTracker.analyses import draw_signals
from CellTracker.coord_image_transformer import Coordinates, CoordsToImageTransformer
from CellTracker.fpm import FPMPart2Model
from CellTracker.global_match import iterative_alignment, pairwise_pointsets_distances, rigid_transform, \
    ensemble_match_initial_20_volumes, get_reference_target_vols_list, get_mds_2d_projection, \
    visualize_pairwise_distances
from CellTracker.plot import plot_initial_matching, plot_predicted_movements, \
    plot_pairs_and_movements
from CellTracker.robust_match import add_or_remove_points, get_extra_cell_candidates, filter_matching_outliers_global
from CellTracker.simple_alignment import align_by_control_points, greedy_match
from CellTracker.stardistwrapper import Segmentation, create_cmap, plot_img_label_max_projection, \
    plot_img_label_max_projection_xz
from CellTracker.test_matching_models import BETA, LAMBDA, load_fpm
from CellTracker.test_matching_models import cal_norm_prob, prgls_with_two_ref, affine_align_by_fpm, \
    predict_matching_prgls, _match_pure_fpm
from CellTracker.utils import load_2d_slices_at_time, normalize_points, del_datasets


class CacheResult:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.status_file = self.cache_path / 'task_status.json'
        self.task_names = ("segmentation", "get_segmented_coords", "rigid_alignment",
                           "calculate_distance", "optimize_order", "segment_ref_volume",
                           "interpolate_segmentation", 'match_first_20_volumes', "filter_proofed_cells",
                           "optimize_matches_20vols", "track_vols_after20", "activities",
                           "cache_max_projections"
                           )
        self.initialize_cache()

    @property
    def seg_file(self):
        return self.cache_path / "seg.h5"

    @property
    def coords_file(self):
        return self.cache_path / "coords.h5"

    @property
    def align_file(self):
        return self.cache_path / "rigid_alignment.h5"

    @property
    def align_folder(self):
        return self.cache_path / "rigid_alignment"

    @property
    def align_temp_folder(self):
        return self.align_folder / ".rigid_alignment_temp"

    @property
    def distance_file(self):
        return self.cache_path / 'distances.h5'

    @property
    def postures_2d_file(self):
        return self.cache_path / 'postures_tx2.npy'

    @property
    def order_file(self):
        return self.cache_path / 'tracking_order.pkl'

    @property
    def order_visualization_folder(self):
        return self.cache_path / "order_visualization"

    @property
    def auto_corrected_seg_file(self):
        return self.cache_path / 'auto_corrected_segmentation.npy'

    @property
    def subregion_file(self):
        return self.cache_path / "subregions_unfiltered.h5"

    @property
    def coords_t_initial_file(self):
        return self.cache_path / 'coords_proof.npy'

    @property
    def matches_1_to_20_file(self):
        return self.cache_path / "matches_20x20.h5"

    @property
    def common_ids_file(self):
        return self.cache_path / "common_ids_vol_initial.npy"

    @property
    def filtered_coords_t0_file(self):
        return self.cache_path / "coords_proof_filtered.npy"

    @property
    def cmap_file(self):
        return self.cache_path / "cmap.npy"

    @property
    def merged_labels_folder(self):
        return self.cache_path / "merged_labels"

    @property
    def merged_labels_xz_folder(self):
        return self.cache_path / "merged_labels_xz"

    @property
    def matchings_folder(self):
        return self.cache_path / "matchings"

    @property
    def matching_file(self):
        return self.cache_path / "matching_results.h5"

    def initialize_cache(self):
        if not self.status_file.exists():
            self.tasks = {}
            for task_name in self.task_names:
                self.tasks[task_name] = {"completed": False, "filepath": ""}
            with open(self.status_file, 'w') as f:
                json.dump(self.tasks, f, indent=4, ensure_ascii=False)
        else:
            with open(self.status_file, 'r') as f:
                self.tasks = json.load(f)
        print(self.tasks)

    def task_done(self, task_name: str, filepath: Path | str):
        if task_name not in self.task_names:
            warnings.warn(f"Task name {task_name} not recognized. Please use one of {self.task_names}")
            return
        self.tasks[task_name]["completed"] = True
        self.tasks[task_name]["filepath"] = str(filepath)
        with open(self.status_file, 'w') as f:
            json.dump(self.tasks, f, indent=4, ensure_ascii=False)

    def should_skip(self, task_name: str, force_redo: bool) -> bool:
        if task_name not in self.task_names:
            raise ValueError(f"Task name {task_name} not recognized. Please use one of {self.task_names}")
        if self.tasks[task_name]["completed"]:
            if not force_redo:
                print(f"Skipping {task_name} because it has already been completed.")
                return True
            else:
                print(f"Forcing redo of {task_name} because force_redo is True.")
                self.reset_states(task_name)
        return False

    def reset_states(self, task_name: str):
        ind = self.task_names.index(task_name)
        for i in range(ind, len(self.task_names)):
            name = self.task_names[i]
            self.tasks[name] = {"completed": False, "filepath": ""}


class RotationPCA:
    def cal_rot_matrix(self, points_2d):
        pca = PCA(n_components=1)
        pca.fit(points_2d)
        pc1 = pca.components_[0]
        angle = np.arccos(pc1[0] / np.linalg.norm(pc1))
        if pc1[1] < 0:
            angle = -angle
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        self.rotation_matrix = rotation_matrix.dot(np.asarray([[0, -1], [1, 0]]))

    def rotate_points(self, coords_norm_nx3) -> ndarray:
        coords_norm_rotated = coords_norm_nx3.copy()
        coords_norm_rotated[:, :2] = coords_norm_rotated[:, :2].dot(self.rotation_matrix)
        return coords_norm_rotated


def _plot_matching(title: str, coords_t1, coords_t2, scale, mean_3,
                   paris, t1, t2, zy_view: bool = False, display=True):
    s_ = np.s_[:, [0,2,1]] if zy_view else np.s_[:, :]
    title = title + ", zy-view" if zy_view else title
    fig = plot_initial_matching(ref_ptrs=(coords_t1 * scale + mean_3)[s_],
                                tgt_ptrs=(coords_t2 * scale + mean_3)[s_],
                                pairs_px2=paris, t1_name=t1, t2_name=t2, display_fig=display)
    fig.suptitle(title, fontsize=20, y=0.99)
    return fig


class DrawMatching:

    def __init__(self, segmented_pos_norm_t1,
                 segmented_pos_norm_t2,
                 segmented_pos_t1,
                 segmented_pos_t2,
                 rotpca, mean, scale, t1, t2, verbosity):
        self.segmented_pos_norm_t1 = segmented_pos_norm_t1
        self.segmented_pos_norm_t2 = segmented_pos_norm_t2
        self.segmented_pos_t1 = segmented_pos_t1
        self.segmented_pos_t2 = segmented_pos_t2
        self.rotpca = rotpca
        self.mean = mean
        self.scale = scale
        self.t1 = t1
        self.t2 = t2
        self.verbosity = verbosity

    def _plot_matching(self, iter: int, coords_subset_t1, coords_subset_t2, _matched_pairs_subset, zy_view = False):
        _plot_matching(f"Matching iter={iter}", coords_subset_t1,coords_subset_t2,
                       self.scale, self.mean, _matched_pairs_subset,
                       self.t1, self.t2, zy_view=zy_view)

    def display_initial_matching(self, matched_pairs,
                                 coords_norm_t1, coords_norm_t2,
                                 iter):
        if self.verbosity < 1:
            return

        coords_norm_t1_rotated = self.rotpca.rotate_points(coords_norm_t1)
        coords_norm_t2_rotated = self.rotpca.rotate_points(coords_norm_t2)

        # Generate figures showing the matching
        if (self.verbosity >= 2 and iter == 0) or (self.verbosity >= 3 and iter >= 1):
            fig = self._plot_matching(iter,
                                      coords_norm_t1_rotated,
                                      coords_norm_t2_rotated,
                                      matched_pairs)
        if self.verbosity >= 4:
            fig = self._plot_matching(iter,
                                      coords_norm_t1_rotated,
                                      coords_norm_t2_rotated,
                                      matched_pairs, zy_view=True)
        return

    def _plot_move_seg(self,title: str, moved_coords_t1, segmented_coords_norm_t2, predicted_coords_t1_to_t2,
                       scale_t2, mean_t2, zy_view: bool = False):
        s_ = np.s_[:, [0, 2, 1]] if zy_view else np.s_[:, :]
        title = title + ", zy-view" if zy_view else title
        fig, _ = plot_predicted_movements(ref_ptrs=moved_coords_t1[s_] * scale_t2 + mean_t2,
                                          tgt_ptrs=segmented_coords_norm_t2[s_] * scale_t2 + mean_t2,
                                          predicted_ref_ptrs=predicted_coords_t1_to_t2[s_] * scale_t2 + mean_t2,
                                          t1=self.t1, t2=self.t2)
        fig.suptitle(title, fontsize=20, y=0.9)

    def display_predicted_movements(self, seg_coords_norm_t1, predicted_coords_t1_in_t2,
                                    iter):
        if self.verbosity < 1:
            return
        seg_coords_t1_rot = self.rotpca.rotate_points(seg_coords_norm_t1)
        seg_coords_t2_rot = self.rotpca.rotate_points(self.segmented_pos_norm_t2)
        predicted_coords_t1_in_t2_rot = self.rotpca.rotate_points(predicted_coords_t1_in_t2)
        if self.verbosity >= 3:
            self._plot_move_seg(f"Predict movements after {iter} iteration", seg_coords_t1_rot,
                                seg_coords_t2_rot,
                                predicted_coords_t1_in_t2_rot, self.scale, self.mean)
        if self.verbosity >= 4:
            self._plot_move_seg(f"Predict movements after {iter} iteration", seg_coords_t1_rot,
                                seg_coords_t2_rot,
                                predicted_coords_t1_in_t2_rot,
                                self.scale, self.mean,
                                zy_view=True)

    def _plot_matching_final(self, coords_t1, coords_t2,
                             _matched_pairs_subset, plot_matching,
                             zy_view: bool = False):
        s_ = np.s_[:, [0,2,1]] if zy_view else np.s_[:, :]
        title = "Final matching" + ", zy-view" if zy_view else "Final matching"
        fig = plot_matching(ref_ptrs=coords_t1[s_],
                            tgt_ptrs=coords_t2[s_],
                            pairs_px2=_matched_pairs_subset,
                            t1=self.t1, t2=self.t2)
        fig.suptitle(title, fontsize=20, y=0.9)

    def display_final_matching(self, pairs):
        if self.verbosity < 1:
            return
        coords_norm_t1_rotated = self.rotpca.rotate_points(self.segmented_pos_norm_t1)
        coords_norm_t2_rot = self.rotpca.rotate_points(self.segmented_pos_norm_t2)
        if self.verbosity >= 1:
            self._plot_matching_final(coords_norm_t1_rotated * self.scale + self.mean,
                                      coords_norm_t2_rot * self.scale + self.mean,
                                      pairs,plot_initial_matching)
            self._plot_matching_final(self.segmented_pos_t1.real,
                                      self.segmented_pos_t2.real, pairs,
                                      plot_initial_matching)
        if self.verbosity >= 4:
            self._plot_matching_final(self.segmented_pos_t1.real,
                                      self.segmented_pos_t2.real,
                                      pairs, 
                                      plot_initial_matching, zy_view=True)

class RigidAlignH5:
    def __init__(self, rigid_align_file: h5py.File):
        self.rigid_align_file = rigid_align_file

    def initialize_align_file(self, coords_nx3, coords_shape, vol_num):
        self.aligned_coords_txnx3 = np.full((vol_num, *coords_shape), np.nan)
        tform_tx3x4 = np.zeros((vol_num, 3, 4))
        tform_inv_tx3x4 = np.zeros((vol_num, 3, 4))
        self.aligned_coords_txnx3[0, ...] = coords_nx3.copy()
        tform_tx3x4[0, :, :3] = np.eye(3)
        tform_inv_tx3x4[0, :, :3] = np.eye(3)
        del_datasets(self.rigid_align_file, ["rigid_aligned_coords", "tform", "tform_inv"])
        self.rigid_align_file.create_dataset("rigid_aligned_coords", data=self.aligned_coords_txnx3)
        self.rigid_align_file.create_dataset("tform", data=tform_tx3x4)
        self.rigid_align_file.create_dataset("tform_inv", data=tform_inv_tx3x4)

    def update_align_file(self, t: int, tform):
        self.rigid_align_file["rigid_aligned_coords"][t - 1, :, :] = self.aligned_coords_txnx3[t - 1]
        self.rigid_align_file["tform"][t - 1, :, :] = tform.params[:3, :]
        self.rigid_align_file["tform_inv"][t - 1, :, :] = np.linalg.inv(tform.params)[:3, :]

    def load_pairs(self, t: int) -> ndarray:
        return self.rigid_align_file[f'paris_t_{t:06d}'][:]

    def update_matching_results(self, t: int, pairs):
        dset_name = f'paris_t_{t:06d}'
        if dset_name in self.rigid_align_file:
            del self.rigid_align_file[dset_name]
        self.rigid_align_file.create_dataset(dset_name, data=pairs)


class RigidAlignFigure:
    def __init__(self, temp_folder: Path, target_folder: Path, rotpca: RotationPCA,
                 mean_3: ndarray, scale: float):
        self.temp_folder = temp_folder
        self.target_folder = target_folder
        self.rotpca = rotpca
        self.mean_3 = mean_3
        self.scale = scale

    def save_figure1(self, fig1):
        fig1.savefig(str(self.temp_folder / 'figure1.png'), dpi=90, facecolor='white')
        plt.close(fig1)

    def save_figure2(self, fig2):
        fig2.savefig(str(self.temp_folder / 'figure2.png'), dpi=90, facecolor='white')
        plt.close(fig2)

    def combine_two_png(self, t: int, arrangement="horizontal"):
        from PIL import Image

        image1 = Image.open(str(self.temp_folder / 'figure1.png'))
        image2 = Image.open(str(self.temp_folder / 'figure2.png'))

        if arrangement == "vertical":
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

        new_image.save(str(self.target_folder / f't_{t:06d}.png'))

    def save_matching_figures(self, t, coords_norm_t0, coords_norm_t2, aligned_coords_t2,
                              pairs):
        t0 = 1
        coords_rot_t1 = self.rotpca.rotate_points(coords_norm_t0)
        coords_rot_t2 = self.rotpca.rotate_points(coords_norm_t2)
        aligned_coords_t2_rot = self.rotpca.rotate_points(aligned_coords_t2)

        fig1 = _plot_matching(f"Match (raw)", coords_rot_t1, coords_rot_t2,
                              self.scale, self.mean_3, pairs[:, [1, 0]], t0, t, display=False)
        self.save_figure1(fig1)
        fig2 = _plot_matching(f"Matching (aligned)", coords_rot_t1, aligned_coords_t2_rot,
                              self.scale, self.mean_3, pairs[:, [1, 0]], t0, t, display=False)
        self.save_figure2(fig2)
        self.combine_two_png(t=t)


def link_t2_to_t0(pairs_t2_t1, rigid_h5, t_ref):
    if t_ref == 2:
        pairs_t2_t0 = pairs_t2_t1.copy()
    else:
        matched_pairs_t1_t0 = rigid_h5.load_pairs(t_ref - 1)
        pairs_t2_t0 = link_pairs(matched_pairs_t1_t0, pairs_t2_t1)
    return pairs_t2_t0


def match_by_prgls(coords_t2, coords_t1, initial_pairs_t2_t1):
    n, m = coords_t2.shape[0], coords_t1.shape[0]
    _, similarity_scores = predict_matching_prgls(initial_pairs_t2_t1,
                                                  coords_t2,
                                                  coords_t2,
                                                  coords_t1,
                                                  (m, n), beta=BETA, lambda_=LAMBDA)
    updated_pairs_t2_t1 = greedy_match(similarity_scores, threshold=0.4)
    updated_pairs_t2_t1 = filter_matching_outliers_global(updated_pairs_t2_t1, coords_t2,
                                                          coords_t1,
                                                          threshold_mdist=5 ** 2)
    return updated_pairs_t2_t1


def match_manual_ref0(pos_segmented: ndarray, pos_proofed: ndarray):
    """Remove cells whose centers are not included in the stardist predictions (including the weak cells)"""
    dist_rxr = cdist(pos_segmented, pos_segmented)
    dist_pxr = cdist(pos_proofed, pos_segmented)
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


class TrackerLite:
    """
    A class that tracks cells in 3D time-lapse images using a trained FPM model.
    """
    T_Initial: int

    def __init__(self, results_folder, raw_img_param, segmentation_param, fpm_model):
        """
        """
        self.Common_Ids_In_Coords = None
        self.Common_Ids_In_Proof = None
        self.Scale_Vol1 = None
        self.Mean_Vol1 = None
        self.rotation_matrix = None # TODO: check why this is not calcuated
        self.Raw_Img_Param = raw_img_param

        self.Inv_Tform_Tx3x4 = None
        self.Tform_Tx3x4 = None
        self.Corresponding_Coords_TxNx3 = None

        self.Results_Dir = Path(results_folder)
        self.cache = CacheResult(self.Results_Dir / ".temp")

        self.seg = Segmentation(self, segmentation_param)
        _fpm_model = load_fpm(fpm_model_path=fpm_model["path"],
                               match_model=fpm_model["architecture"])
        self.Fpm_Models = (_fpm_model, FPMPart2Model(_fpm_model.comparator))
        self.coords2image: CoordsToImageTransformer = CoordsToImageTransformer(self)
        self.Image_Size_YXZ = self._get_image_size()

    def _get_image_size(self):
        x = load_2d_slices_at_time(images_path=self.Raw_Img_Param, channel_name="channel_nuclei", t=1)
        print(f"Raw image shape at vol1: {x.shape}", f"Dtype: {x.dtype}")
        return (*x.shape[1:], x.shape[0])

    @property
    def tracking_results_file(self):
        return self.Results_Dir / "tracking_results.h5"

    def get_segmented_coordinates(self, force_redo=False):
        self.Vol_Num = h5py.File(str(self.cache.seg_file), "r")["prob"].shape[0]
        if self.cache.should_skip("get_segmented_coords", force_redo=force_redo):
            return

        if self.cache.coords_file.exists():
            self.cache.coords_file.unlink()
        with h5py.File(str(self.cache.seg_file), "r") as seg_file, \
              h5py.File(str(self.cache.coords_file), 'a') as coords_file:
            for t in tqdm(range(self.Vol_Num)):
                coordinates_stardist = seg_file[f'coords_{str(t).zfill(6)}'][:]
                prob_map = self.coords2image.load_prob_map(t, seg_file, self.Image_Size_YXZ)
                extra_coordinates = get_extra_cell_candidates(coordinates_stardist, prob_map) # bottleneck
                if extra_coordinates.shape[0] != 0:
                    combined_coordinates = np.concatenate((coordinates_stardist, extra_coordinates), axis=0)
                else:
                    combined_coordinates = coordinates_stardist
                num = len(coordinates_stardist)

                dset = coords_file.create_dataset(f'combined_coords_{str(t).zfill(6)}', data=combined_coordinates)
                dset.attrs["num"] = num
            self.cache.task_done("get_segmented_coords", self.cache.coords_file)

    def _get_segmented_pos(self, t: int) -> Tuple[Coordinates, ndarray]:
        """Get segmented positions and extra positions from stardist model"""
        interp_factor = self.coords2image.Interp_Factor
        voxel_size = self.coords2image.Voxel_Size

        with h5py.File(str(self.cache.coords_file), 'r') as f:
            combined_coordinates = f[f'combined_coords_{str(t-1).zfill(6)}'][:]
            num_inliers = f[f'combined_coords_{str(t-1).zfill(6)}'].attrs["num"]

        pos = Coordinates(combined_coordinates, interpolation_factor=interp_factor,
                          voxel_size=voxel_size, dtype="raw")
        inliers = np.arange(num_inliers)
        return pos, inliers

    def rotate_to_PC_axes(self, t: int) -> RotationPCA:
        segmented_pos, subset = self._get_segmented_pos(t)
        points_2d = segmented_pos.real[subset][:, :2]
        rot_pca = RotationPCA()
        rot_pca.cal_rot_matrix(points_2d)
        return rot_pca

    def rigid_alignment(self, fpm_model_rot=None, t_ref_start: int = 2, force_redo=False):
        """
        Calculate the parameters for rigid alignment between vol#1 and all other volumes, and the aligned coordinates.
        Store the results into a h5 file
        """
        if self.cache.should_skip("rigid_alignment", force_redo=force_redo):
            return
        t0 = 1
        coords_subset_norm_t0, mean_t0_3, scale_t0, _, _, _ = self.load_normalized_coords(t0)
        coords_subset_t0_shape = coords_subset_norm_t0.shape
        self.cache.align_temp_folder.mkdir(exist_ok=True, parents=True)
        if fpm_model_rot is None:
            fpm_models_rot = None
        else:
            fpm_models_rot = (fpm_model_rot, FPMPart2Model(fpm_model_rot.comparator))

        if self.cache.align_file.exists():
            self.cache.align_file.unlink()
        with h5py.File(str(self.cache.align_file), "a") as rigid_align_file:
            rigid_h5 = RigidAlignH5(rigid_align_file)
            rigid_h5.initialize_align_file(
                coords_subset_norm_t0, coords_subset_t0_shape, self.Vol_Num)
            rotpca = self.rotate_to_PC_axes(t=t0)
            rigid_fig = RigidAlignFigure(self.cache.align_temp_folder, self.cache.align_folder,
                                         rotpca, mean_t0_3, scale_t0)
            plt.ioff()
            for t_ref in tqdm(range(t_ref_start, self.Vol_Num + 1)):
                coords_subset_norm_t2_pre, _, _, _, _, _ = self.load_normalized_coords(
                    t_ref - 1, mean_3=mean_t0_3, scale=scale_t0)
                coords_subset_norm_t2, _, _, _, _, _ = self.load_normalized_coords(
                    t_ref, mean_3=mean_t0_3, scale=scale_t0)
                # fpm + prgls: t2 to t1
                matched_pairs_t2_t1 = iterative_alignment(
                    self.Fpm_Models, fpm_models_rot,
                    coords_subset_norm_t2, coords_subset_norm_t2_pre)  # bottleneck
                # link t2 to t0
                matched_pairs_t2_t0 = link_t2_to_t0(matched_pairs_t2_t1, rigid_h5, t_ref)
                # rigid alignment: t2 to t0
                aligned_coords_t2, _ = align_by_control_points(
                    coords_subset_norm_t2, coords_subset_norm_t0,
                    matched_pairs_t2_t0, method="euclidean")

                # prgls: t2 (aligned) to t0
                matched_pairs_t2_t0 = match_by_prgls(aligned_coords_t2,
                                                     coords_subset_norm_t0,
                                                     matched_pairs_t2_t0)
                rigid_h5.update_matching_results(t_ref, matched_pairs_t2_t0)

                # rigid alignment (again): t2 to t0
                aligned_coords_t2, tform = align_by_control_points(coords_subset_norm_t2, coords_subset_norm_t0,
                                                                          matched_pairs_t2_t0,
                                                                          method="euclidean")
                for id_ref, id_tgt in matched_pairs_t2_t0:
                    rigid_h5.aligned_coords_txnx3[t_ref - 1, id_tgt, :] = aligned_coords_t2[id_ref, :]
                rigid_h5.update_align_file(t_ref, tform)
                rigid_fig.save_matching_figures(t_ref,
                                                coords_subset_norm_t0,
                                                coords_subset_norm_t2,
                                                aligned_coords_t2,
                                                matched_pairs_t2_t0)
            plt.ion()
        self.cache.task_done("rigid_alignment", self.cache.align_file)

    def load_rotation_alignment(self):
        with h5py.File(str(self.cache.align_file), "r") as f:
            corresponding_coords_txnx3 = f["rigid_aligned_coords"][:]
            tform_tx3x4 = f["tform"][:]
            inv_tform_tx3x4 = f["tform_inv"][:]
        return corresponding_coords_txnx3, tform_tx3x4, inv_tform_tx3x4

    def calculate_distances(self, force_redo: bool = False):
        self.Corresponding_Coords_TxNx3, self.Tform_Tx3x4, self.Inv_Tform_Tx3x4 = self.load_rotation_alignment()
        if self.cache.should_skip('calculate_distance', force_redo):
            with h5py.File(str(self.cache.distance_file), 'r') as file:
                self.Distances_TxT = file['distances_txt'][:]
                self.T_Initial = file.attrs['t_initial']
            return

        mean_coords_nx3 = np.nanmean(self.Corresponding_Coords_TxNx3, axis=0)
        self.Distances_TxT = pairwise_pointsets_distances(self.Corresponding_Coords_TxNx3)
        t, n, _ = self.Corresponding_Coords_TxNx3.shape
        dist_t = np.zeros((t))
        for i in range(n):
            dist_i_t = cdist(self.Corresponding_Coords_TxNx3[:, i, :], mean_coords_nx3[i:i + 1, :], metric='euclidean')[:, 0]
            # Replaces any NaN values in the distance matrix with the mean of the non-NaN values
            dist_i_t[np.isnan(dist_i_t)] = np.nanmean(dist_i_t)
            dist_t += dist_i_t
        self.T_Initial = np.argmin(dist_t) + 1
        with h5py.File(str(self.cache.distance_file), 'a') as file:
            del_datasets(file, ['distances_txt'])
            file.create_dataset('distances_txt', data=self.Distances_TxT)
            file.attrs['t_initial'] = self.T_Initial
            self.cache.task_done('calculate_distance', self.cache.distance_file)

    def view_postures_mds(self):
        points_tx2 = self.get_postures_array()
        visualize_pairwise_distances(points_tx2, self.T_Initial, show_id=False, show_trajectory=True)

    def get_postures_array(self):
        if self.cache.postures_2d_file.exists():
            print("loading postures 2d projection ...")
            points_tx2 = np.load(str(self.cache.postures_2d_file))
        else:
            print("Calculate postures 2d projection ...")
            points_tx2 = get_mds_2d_projection(self.Distances_TxT)
            np.save(str(self.cache.postures_2d_file), points_tx2)
        return points_tx2

    def optimize_tracking_order(self, max_num_refs: int = 20, force_redo: bool = False):
        if self.cache.should_skip('optimize_order', force_redo):
            with open(str(self.cache.order_file), 'rb') as file:
                self.order_list = pickle.load(file)
            self.Max_Num_Refs = max_num_refs
            return
        self.order_list = get_reference_target_vols_list(
            self.Distances_TxT, self.T_Initial, max_num_refs)
        with open(str(self.cache.order_file), 'wb') as file:
            pickle.dump(self.order_list, file)
        self.cache.task_done('optimize_order', self.cache.order_file)
        self.Max_Num_Refs = max_num_refs

    def visualize_ensemble_sampling(self, max_num_refs: int = 20, width: int = 1, start_t=0):
        points_tx2 = self.get_postures_array()
        self.cache.order_visualization_folder.mkdir(exist_ok=True, parents=True)
        rotation_matrix = self.rotate_to_PC_axes(t=1).rotation_matrix
        aligned_corresponding_coords_txnx3 = np.matmul(self.Corresponding_Coords_TxNx3[:, :, :2], rotation_matrix)

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

        xmin = np.nanmin(aligned_corresponding_coords_txnx3[..., 1])
        xmax = np.nanmax(aligned_corresponding_coords_txnx3[..., 1])
        ymin = np.nanmin(aligned_corresponding_coords_txnx3[..., 0])
        ymax = np.nanmax(aligned_corresponding_coords_txnx3[..., 0])

        if (xmax - xmin) > aspect_ratio * (ymax - ymin):
            y_mid = (ymax + ymin) / 2
            ymax = y_mid + (xmax - xmin) / (2 * aspect_ratio)
            ymin = y_mid - (xmax - xmin) / (2 * aspect_ratio)
        else:
            x_mid = (xmax + xmin) / 2
            xmax = x_mid + (ymax - ymin) * aspect_ratio / 2
            xmin = x_mid - (ymax - ymin) * aspect_ratio / 2

        tgt_list = self.tgt_list
        for frame in tqdm(range(start_t, t - 1)):
            refs, tgt = self.order_list[frame]

            ax_list[0].set_title(f'tgt = {tgt}', c="r")

            ax1.scatter(points_tx2[:, 0], points_tx2[:, 1], c="grey")
            ax1.scatter(points_tx2[tgt_list[:frame + 1] - 1, 0],
                        points_tx2[tgt_list[:frame + 1] - 1, 1], c="orange")
            ax1.scatter(points_tx2[refs - 1, 0], points_tx2[refs - 1, 1], c="b")
            ax1.scatter(points_tx2[tgt - 1, 0], points_tx2[tgt - 1, 1], c="r")
            ax1.set_title('MDS visualization of the point sets')

            for i in range(max_num_refs):
                if i + 1 > len(refs):
                    ax_list[i].set_xticklabels([])
                    ax_list[i].set_yticklabels([])
                    continue
                pts = aligned_corresponding_coords_txnx3[refs[i] - 1, ...]
                pts_2 = aligned_corresponding_coords_txnx3[tgt - 1, ...]
                ax_list[i].scatter(pts[:, 1], pts[:, 0], c="b", s=5)
                ax_list[i].scatter(pts_2[:, 1], pts_2[:, 0], c="r", s=5)
                ax_list[i].set_ylim(ymin, ymax)
                ax_list[i].set_xlim(xmin, xmax)
                ax_list[i].invert_yaxis()
                ax_list[i].set_xticklabels([])
                ax_list[i].set_yticklabels([])
                ax_list[i].text(1, 1, f'{refs[i]}', transform=ax_list[i].transAxes, ha='right', va='top', c="b")

            plt.tight_layout()
            fig.savefig(str(self.cache.order_visualization_folder / f'{frame:06d}.png'), dpi=90, facecolor='white')
            ax1.cla()
            for ax in ax_list:
                ax.cla()
        plt.close(fig)

    def segment_reference_volume(self, force_include_bright: bool = False, force_redo: bool = False):

        self.AutoSegPath = self.Results_Dir / f'segmentation_vol_{self.T_Initial}.h5'
        if self.cache.should_skip('segment_ref_volume', force_redo):
            print(f"Saved auto-segmentation as {self.AutoSegPath}")
            return
        x = load_2d_slices_at_time(images_path=self.Raw_Img_Param, channel_name="channel_nuclei", t=self.T_Initial)
        labels_zyx, details, prob_map_reduced_zyx, dist_map_reduced_zyxr = (
            self.seg.stardist_model._predict_instances_simple(
            x, n_tiles=self.seg.n_tiles, prob_thresh=self.seg.prob_thresh))
        grid_zyx = self.seg.stardist_model.config.grid
        prob_map_zyx = np.repeat(np.repeat(np.repeat(
            prob_map_reduced_zyx, grid_zyx[0], axis=0),
            grid_zyx[1], axis=1),
            grid_zyx[2], axis=2)
        with h5py.File(str(self.AutoSegPath), "w") as f:
            f.attrs["raw_dset"] = self.Raw_Img_Param["dset"]
            f.attrs["raw_channel_nuclei"] = self.Raw_Img_Param["channel_nuclei"]
            f.attrs["t_initial"] = self.T_Initial
            f.create_dataset("seg_labels_zyx", data=labels_zyx, compression="lzf")
            f.create_dataset("dist_map_reduced_zyxr", data=dist_map_reduced_zyxr, compression="lzf")
            f.create_dataset("prob_map_zyx", data=prob_map_zyx, compression="lzf")
            f.create_dataset("img_shape", data=details["img_shape"])
            f.create_dataset("prob_n", data=details["prob"])
            f.create_dataset("dist_n", data=details["dist"])
            f.create_dataset("coords_nx3", data=details["points"])
        print(f"Saved auto-segmentation as {self.AutoSegPath}")
        self.cache.task_done('segment_ref_volume', self.AutoSegPath)

        self.draw_segmentation(labels_zyx, x)

    def draw_segmentation(self, labels_zyx, image):
        cmap = create_cmap(labels_zyx.transpose((1, 2, 0)), voxel_size_yxz=self.coords2image.Voxel_Size,
                           max_color=20)
        plot_img_label_max_projection(
            image, labels_zyx, cmap, lbl_title="label pred (projection)", fig_width_px=2000)
        plot_img_label_max_projection_xz(
            image, labels_zyx, cmap, lbl_title="label pred (projection)", fig_width_px=2000,
            scale_z=self.coords2image.Interp_Factor)

    def modify_reference_segmentation(self, neuropal_loader_type: str = "none", neuropal_file_path: str = None):
        from .segmentation_inspector import SegmentationInspector
        inspector = SegmentationInspector(self, neuropal_loader_type)
        inspector.load_segmentation_results(tracking_result_path=self.AutoSegPath, neuropal_image_path=neuropal_file_path)
        inspector.load_raw_t0(path=self.Raw_Img_Param["h5_file"])

    def load_segmentation(self, force_redo: bool = False):
        if self.cache.should_skip('interpolate_segmentation', force_redo=force_redo):
            pass
        else:
            self.coords2image.load_segmentation(str(self.AutoSegPath))
            x = load_2d_slices_at_time(images_path=self.Raw_Img_Param, channel_name="channel_nuclei", t=self.T_Initial)
            cmap = create_cmap(self.coords2image.Proofed_Segmentation, self.coords2image.Voxel_Size)
            plot_img_label_max_projection(x, self.coords2image.Proofed_Segmentation.transpose(2, 0, 1),
                                          cmap, lbl_title="label proofed (projection)", fig_width_px=2000)
            plot_img_label_max_projection_xz(x, self.coords2image.Proofed_Segmentation.transpose(2, 0, 1),
                                             cmap, lbl_title="label proofed (projection)",
                                             fig_width_px=2000, scale_z=self.coords2image.Interp_Factor)
        self._interpolate_segmentation(force_redo=force_redo)

    def load_unfiltered_subregions(self):
        with h5py.File(str(self.cache.subregion_file), "r") as subregion_file:
            n_cells = subregion_file.attrs["num_cells"]
            group = subregion_file["subregions"]
            subregions = []
            for i in range(n_cells):
                cell_i = group[f"subregion_{i + 1}"]
                x0, x1, y0, y1, z0, z1 = cell_i.attrs["slice_xyz"]
                subregions.append(((slice(x0, x1), slice(y0, y1), slice(z0, z1)), cell_i[:]))
        return subregions

    def save_unfiltered_subregions(self):
        with h5py.File(str(self.cache.subregion_file), "a") as subregion_file:
            subregion_file.attrs["num_cells"] = len(self.coords2image.Sub_Regions)
            del_datasets(subregion_file, ["subregions",])
            group = subregion_file.create_group("subregions")
            for i, (slices_xyz, subregion) in enumerate(self.coords2image.Sub_Regions):
                if f"subregion_{i + 1}" in subregion_file:
                    del subregion_file[f"subregion_{i + 1}"]
                if f"subregion_{i + 1}" in group:
                    del group[f"subregion_{i + 1}"]
                dset = group.create_dataset(f"subregion_{i + 1}", data=subregion)
                dset.attrs["slice_xyz"] = (slices_xyz[0].start, slices_xyz[0].stop,
                                           slices_xyz[1].start, slices_xyz[1].stop,
                                           slices_xyz[2].start, slices_xyz[2].stop)

    def _interpolate_segmentation(self, force_redo: bool = False):
        if self.cache.should_skip('interpolate_segmentation', force_redo=force_redo):
            self.coords2image.Sub_Regions = self.load_unfiltered_subregions()
            with h5py.File(str(self.cache.auto_corrected_seg_file), "r") as file:
                self.coords2image.Auto_Corrected_Segmentation = file["auto_corrected_seg"][:]
            coord_proof = np.load(str(self.cache.coords_t_initial_file))
            self.coords2image.Coord_Proof = Coordinates(coord_proof,
                                                        self.coords2image.Interp_Factor,
                                                        self.coords2image.Voxel_Size,
                                                        dtype="raw")
            self.coords2image.Z_Slice_Original_Labels = slice(
                self.coords2image.Interp_Factor // 2,
                self.coords2image.Interp_Factor * self.Image_Size_YXZ[2],
                self.coords2image.Interp_Factor)
            return
        self.coords2image.interpolate_labels()
        with h5py.File(str(self.cache.auto_corrected_seg_file), "a") as file:
            del_datasets(file, ["auto_corrected_seg",])
            file.create_dataset("auto_corrected_seg", data=self.coords2image.Auto_Corrected_Segmentation,
                                compression="lzf")
        self.save_unfiltered_subregions()
        np.save(str(self.cache.coords_t_initial_file), np.asarray(self.coords2image.Coord_Proof.raw))
        cmap = create_cmap(self.coords2image.Auto_Corrected_Segmentation,
                           voxel_size_yxz=self.coords2image.Voxel_Size, max_color=20,
                           dist_type="2d_projection")
        x = load_2d_slices_at_time(images_path=self.Raw_Img_Param, channel_name="channel_nuclei", t=self.T_Initial)
        plot_img_label_max_projection(x, self.coords2image.Auto_Corrected_Segmentation.transpose(2, 0, 1),
                                      cmap, lbl_title="label interpolated (projection)", fig_width_px=2200)
        plot_img_label_max_projection_xz(x, self.coords2image.Auto_Corrected_Segmentation.transpose(2, 0, 1),
                                         cmap, lbl_title="label interpolated (projection)", fig_width_px=2200,
                                         scale_z=self.coords2image.Interp_Factor)
        self.cache.task_done('interpolate_segmentation',
                             ",".join([str(self.cache.auto_corrected_seg_file),
                                       str(self.cache.subregion_file),
                                       str(self.cache.coords_t_initial_file)]))

    def predict_cell_positions(self, t1: int, t2: int, pairs_init: ndarray = None,
                               beta: float = BETA, lambda_: float = LAMBDA, verbosity: int = 0,
                               learning_rate: float = 0.5) -> ndarray:
        """
        Predicts the positions of cells in a 3D image at time step t2, based on their positions at time step t1.

        Parameters
        ----------
        t1:
            The time step t1.
        t2:
            The time step t2.
        pairs_init:
            if not None, use it to restrict the search region in FPM matching
        beta:
            The beta parameter for the prgls model.
        lambda_:
            The lambda parameter for the prgls model.
        verbosity:
            The verbosity level. 0-4. 0: no figure, 1: only final matching figure, 2: initial and final matching figures,
            3: all figures during iterations only in y-x view, 4: all figures during iterations with additional z-x view.
        learning_rate:
            The learning rate for updating the predicted points, between 0 and 1.

        Returns
        --------
        pairs_seg_t1_seg_t2: ndarray, shape (n, 2)
            The matching between the segmentation at time step t1 and the segmentation at time step t2, including these "weak" cells
        """
        # Load normalized coordinates at t1 and t2: segmented_pos is un-rotated, while other coords are rotated
        coords_subset_norm_t1, segmented_coords_norm_t1, segmented_pos_t1, subset_t1 = (
            self.load_rotated_normalized_coords(t1)
        )
        coords_subset_norm_t2, segmented_coords_norm_t2, segmented_pos_t2, subset_t2 = (
            self.load_rotated_normalized_coords(t2)
        )
        subset = (subset_t1, subset_t2)

        confirmed_coords_norm_t1 = coords_subset_norm_t1.copy()
        subset_confirmed = subset_t1

        n, m = segmented_coords_norm_t1.shape[0], segmented_coords_norm_t2.shape[0]
        aligned_coords_subset_norm_t1, coords_subset_norm_t2, _, affine_tform = affine_align_by_fpm(
            self.Fpm_Models,
            coords_norm_t1=coords_subset_norm_t1,
            coords_norm_t2=coords_subset_norm_t2
        )
        aligned_segmented_coords_norm_t1 = affine_tform(segmented_coords_norm_t1)
        aligned_confirmed_coords_norm_t1 = affine_tform(confirmed_coords_norm_t1)
        moved_seg_coords_t1 = aligned_segmented_coords_norm_t1.copy()

        rotpca_t1 = None
        if verbosity > 0:
            rotpca_t1 = self.rotate_to_PC_axes(t=t1)
        self.matching_fig = DrawMatching(segmented_coords_norm_t1,
                                         segmented_coords_norm_t2,
                                         segmented_pos_t1,
                                         segmented_pos_t2,
                                         rotpca_t1, self.Mean_Vol1, self.Scale_Vol1,
                                         t1, t2, verbosity)

        similarity_subset = None
        iteration = 3
        # Iterative matching by FPM + PRGLS
        for i in range(iteration):
            _matched_pairs_subset = _match_pure_fpm(
                aligned_coords_subset_norm_t1, coords_subset_norm_t2,
                self.Fpm_Models, similarity_threshold=0.4, prob_mxn_initial=similarity_subset)
            self.matching_fig.display_initial_matching(_matched_pairs_subset,
                                                       aligned_coords_subset_norm_t1,
                                                       coords_subset_norm_t2,
                                                       i)

            matched_pairs = np.column_stack((subset[0][_matched_pairs_subset[:, 0]], subset[1][_matched_pairs_subset[:, 1]]))

            if i == iteration - 1:
                break

            # Predict the corresponding positions in t2 of all the segmented cells in t1
            predicted_coords_t1_to_t2, similarity_mxn = predict_by_prgls(
                matched_pairs, aligned_segmented_coords_norm_t1, aligned_segmented_coords_norm_t1, segmented_coords_norm_t2,
                (m, n), beta, lambda_)

            self.matching_fig.display_predicted_movements(moved_seg_coords_t1,
                                                          predicted_coords_t1_to_t2,
                                                          i)

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
            similarity_subset = similarity_mxn[np.ix_(subset[1], subset[0])]

        # Final prediction of cell positions by PRGLS
        tracked_coords_norm_t2, similarity_mxn = predict_by_prgls(
            matched_pairs, aligned_confirmed_coords_norm_t1, aligned_segmented_coords_norm_t1, segmented_coords_norm_t2,
            (m, n), beta, lambda_)

        pairs_seg_t1_seg_t2 = greedy_match(similarity_mxn, threshold=0.5)
        pairs_in_confirmed_subset = np.asarray([(np.nonzero(subset_confirmed == i)[0][0], j) for
                                                i, j in pairs_seg_t1_seg_t2 if i in subset_confirmed])
        tracked_coords_norm_t2[pairs_in_confirmed_subset[:, 0], :] = segmented_coords_norm_t2[
                                                                     pairs_in_confirmed_subset[:, 1], :]

        self.matching_fig.display_final_matching(pairs_seg_t1_seg_t2)
        return pairs_seg_t1_seg_t2

    def match_first_20_volumes(self, force_redo: bool = False):
        if self.cache.should_skip('match_first_20_volumes', force_redo=force_redo):
            return
        num = 20
        tgt_list_20 = self.tgt_list_20
        with h5py.File(str(self.cache.matches_1_to_20_file), "w") as f:
            for i in tqdm(range(num)):
                t1 = tgt_list_20[i]
                for j in range(num):
                    if i == j:
                        continue
                    t2 = tgt_list_20[j]
                    pairs = self.predict_cell_positions(t1, t2)
                    clear_memory()
                    del_datasets(f, [f"{i+1}_{j+1}"])
                    f.create_dataset(f"{i + 1}_{j + 1}", data=pairs)
        self.cache.task_done('match_first_20_volumes', str(self.cache.matches_1_to_20_file))

    def _filter_proofed_cells(self, skip: bool=False, threshold: int = 15, force_redo: bool = False):
        """Remove cells that were not detected by stardist or the ones not exist in most of the other intial_x_volumes"""
        if self.cache.should_skip('filter_proofed_cells', force_redo=force_redo):
            self.coords2image.Filtered_Subregions = self.load_filtered_subregions()
            coord_proof_filtered = np.load(str(self.cache.filtered_coords_t0_file))
            self.coords2image.Coord_Filtered_T0 = Coordinates(coord_proof_filtered,
                                                              self.coords2image.Interp_Factor,
                                                              self.coords2image.Voxel_Size, dtype="raw")
            self.coords2image.Cmap_Colors = np.load(str(self.cache.cmap_file))
            self.Common_Ids_In_Coords = np.load(str(self.cache.common_ids_file))
            return

        self.Common_Ids_In_Coords, common_ids_in_proof = self._extract_common_ids(threshold,skip)
        np.save(str(self.cache.common_ids_file), self.Common_Ids_In_Coords)

        self.coords2image.filter_segmentation_vol0(common_ids_in_proof + 1,
                                                   self.Raw_Img_Param, self.T_Initial)
        self.save_filtered_subregions()
        self.cache.task_done('filter_proofed_cells', str(self.cache.filtered_coords_t0_file))

    def _extract_common_ids(self, threshold, skip: bool):
        x_vols = 20
        threshold_out_of_19_volumes = threshold - 1
        segmented_pos_t1, _ = self._get_segmented_pos(self.T_Initial)

        if skip:
            cum_match_counts = np.full((segmented_pos_t1.real.shape[0]), np.inf)
        else:
            cum_match_counts = np.zeros((segmented_pos_t1.real.shape[0]), dtype=int)
            for i in tqdm(range(1, x_vols)):
                with h5py.File(str(self.cache.matches_1_to_20_file), "r") as f:
                    pairs = f[f"1_{i + 1}"][:]
                for ref, _ in pairs:
                    cum_match_counts[ref] += 1
        Common_Ids_In_Coords, common_ids_in_proof = self.get_common_ids(cum_match_counts,
                                                                             threshold_out_of_19_volumes)
        print(f"Choose {len(Common_Ids_In_Coords)} cells from {segmented_pos_t1.real.shape[0]} proofed cells")
        return Common_Ids_In_Coords, common_ids_in_proof

    def pairs_t1_to_t20_list(self):
        cell_num_list = self.get_cell_num_initial_20_vols()
        pairs_list_20 = ensemble_match_initial_20_volumes(
            cell_num_list, str(self.cache.matches_1_to_20_file))

        with h5py.File(str(self.cache.matching_file), "a") as f:
            for i, pairs in enumerate(pairs_list_20[1:]):
                _, tgt = self.order_list[i]
                if f"pairs_initial_to_{tgt}" in f:
                    del f[f"pairs_initial_to_{tgt}"]
                f.create_dataset(f"pairs_initial_to_{tgt}", data=pairs)

        return pairs_list_20

    def save_optimized_matches_in_first_20_volumes(self, max_repetition=2, force_redo: bool = False):
        """Predict the cell positions, and then save the predicted positions and pairs"""
        self._filter_proofed_cells(skip=True)
        if self.cache.should_skip('optimize_matches_20vols', force_redo=force_redo):
            pass
        else:
            t1 = self.T_Initial # initial volume of the confirmed segmentation
            self.cache.matchings_folder.mkdir(exist_ok=True, parents=True)
            plt.ioff()
            pairs_t1_to_t20_list = self.pairs_t1_to_t20_list()
            for t_target in tqdm(range(1, 20)):
                t2 = self.order_list[t_target - 1][1]
                _, segmented_coords_norm_t1, _, _ = self.load_rotated_normalized_coords(t1)
                _, segmented_coords_norm_t2, _, _ = self.load_rotated_normalized_coords(t2)
                pairs = pairs_t1_to_t20_list[t_target]

                ref_ptrs_confirmed, ref_ptrs_tracked_t2 = self.predict_pos_all_cells(
                    pairs, segmented_coords_norm_t1,
                    segmented_coords_norm_t2)

                confirmed_coord, corrected_labels_image = self.finetune_positions_in_image(
                    ref_ptrs_tracked_t2, t2, max_repetition=max_repetition)

                segmented_coords_norm_t1, segmented_coords_norm_t2, ref_ptrs_confirmed, _ref_ptrs_tracked_t2 = rotate_for_visualization(
                    self.rotation_matrix,
                    (segmented_coords_norm_t1, segmented_coords_norm_t2, ref_ptrs_confirmed, ref_ptrs_tracked_t2))
                fig = plot_pairs_and_movements(
                    segmented_coords_norm_t1, segmented_coords_norm_t2, t1, t2,
                    ref_ptrs_confirmed, _ref_ptrs_tracked_t2, display_fig=False, show_ids=False)
                self.save_tracking_results(
                    confirmed_coord, corrected_labels_image,
                    t=t2, images_path=self.Raw_Img_Param, step=t_target + 1)
                fig.savefig(str(self.cache.matchings_folder / f"matching_t{t2:06d}.png"), dpi=90, facecolor='white')
                plt.close(fig)

            plt.ion()
            self.cache.task_done('optimize_matches_20vols', str(self.cache.matchings_folder))

    def get_restart_time(self, restart_from_breakpoint):
        if restart_from_breakpoint:
            with h5py.File(str(self.tracking_results_file), 'r') as f:
                step = f.attrs["last_step"]
            restart_timing = step - 20
            assert restart_timing >= 0, "You need to track the 1-20 volumes first!"
            print(f"restart from step={step + 1}")
        else:
            restart_timing = 0
        return restart_timing

    def get_initial_pairs(self, num_cells_t0, num_cells_tgt, num_ensemble, ref_list, tgt):
        # TODO: slow, need optimization
        matches_matrix_t0_to_tgt = np.zeros((num_cells_t0, num_cells_tgt), dtype=int)
        for ref in ref_list[:num_ensemble]:
            pairs_ref_tgt = self.predict_cell_positions(ref, tgt)
            clear_memory()
            if self.T_Initial == ref:
                pairs_t0_ref = None
            else:
                with h5py.File(str(self.cache.matching_file), "r") as f:
                    pairs_t0_ref = f[f"pairs_initial_to_{ref}"][:]
            combine_links(matches_matrix_t0_to_tgt, pairs_t0_ref, pairs_ref_tgt)
        pairs_t0_tgt = matrix2pairs(
            matches_matrix_t0_to_tgt, num_cells_t0, num_ensemble)
        return pairs_t0_tgt

    def track_vols_after20(self, restart_from_breakpoint: bool = True, num_ensemble: int = None, max_repetition=2,
                           force_redo: bool = False):
        if self.cache.should_skip('track_vols_after20', force_redo=force_redo):
            return
        # Determine the number of reference volumes
        if num_ensemble is None:
            num_ensemble = self.Max_Num_Refs
            print(f"num_ensemble = {num_ensemble}")
        else:
            assert num_ensemble <= self.Max_Num_Refs, f"num_ensemble should be <= {self.Max_Num_Refs}"

        restart_timing = self.get_restart_time(restart_from_breakpoint)
        _, segmented_coords_norm_t0, _, _ = self.load_rotated_normalized_coords(self.T_Initial)
        num_cells_t0 = segmented_coords_norm_t0.shape[0]

        plt.ioff()
        for i in tqdm(range(19 + restart_timing, len(self.order_list))):
            ref_list, tgt = self.order_list[i]
            _, segmented_coords_norm_tgt, _, _ = self.load_rotated_normalized_coords(tgt)
            num_cells_tgt = segmented_coords_norm_tgt.shape[0]
            # Calculate the matching by FPM + PRGLS + Ensemble votes
            pairs_t0_tgt = self.get_initial_pairs(
                num_cells_t0, num_cells_tgt, num_ensemble, ref_list, tgt)

            # Re-link from initial volume to target volume and save the matched pairs
            _, similarity_scores = predict_matching_prgls(pairs_t0_tgt,
                                                          segmented_coords_norm_t0,
                                                          segmented_coords_norm_t0,
                                                          segmented_coords_norm_tgt,
                                                          (num_cells_tgt, num_cells_t0), beta=BETA, lambda_=LAMBDA)
            updated_pairs_t0_tgt = greedy_match(similarity_scores, threshold=0.4)
            with h5py.File(str(self.cache.matching_file), "a") as f:
                del_datasets(f, [f"pairs_initial_to_{tgt}"])
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
            fig = plot_pairs_and_movements(
                _segmented_coords_norm_t0, _segmented_coords_norm_tgt, self.T_Initial, tgt,
                _confirmed_cells_pos_t0, _confirmed_cells_pos_tgt,
                display_fig=False, show_ids=False)
            fig.savefig(str(self.cache.matchings_folder / f"matching_t{tgt:06d}.png"), dpi=90, facecolor='white')
            plt.close(fig)
            self.save_tracking_results(
                confirmed_cells_finetuned_pos_tgt, finetuned_labels_image,
                t=tgt, images_path=self.Raw_Img_Param, step=i + 2)
        plt.ion()
        self.cache.task_done('track_vols_after20', str(self.tracking_results_file))

    def save_activities_to_csv(self, activities_txn):
        csv_filename = self.cache.cache_path / "activities_txcell.csv"
        timings = np.arange(1, activities_txn.shape[0] + 1).reshape(-1, 1)
        activities_with_timings = np.concatenate((timings, activities_txn), axis=1)
        headers = ['timing'] + ['cell' + str(i) for i in range(1, activities_txn.shape[1] + 1)]
        np.savetxt(str(csv_filename), activities_with_timings,
                   delimiter=',', fmt='%.3f', header=','.join(headers), comments='')

    def activities(self, force_redo: bool = False):
        if self.cache.should_skip('activities', force_redo=force_redo):
            return

        with h5py.File(str(self.tracking_results_file), "a") as track_file, \
                        h5py.File(self.Raw_Img_Param["h5_file"], 'r') as f_raw:
            cell_num = np.max(track_file["tracked_labels"][self.T_Initial - 1, :, 0, :, :])
            activities_txn = np.zeros((self.Vol_Num, cell_num))
            coords_txnx3 = np.zeros((self.Vol_Num, cell_num, 3))
            for t in tqdm(range(1, self.Vol_Num + 1)):
                try:
                    raw = f_raw[self.Raw_Img_Param["dset"]][t - 1, :, self.Raw_Img_Param["channel_activity"], :, :]
                except FileNotFoundError:
                    # Handle missing image files
                    print(f"Warning: Raw images at t={t - 1} cannot be loaded! Stop calculation!")
                    break

                labels_img = track_file["tracked_labels"][t-1, :, 0, :, :]
                coords_txnx3[t - 1, :] = np.asarray(
                    ndimage.measurements.center_of_mass(labels_img > 0, labels_img, range(1, cell_num + 1)))
                found_bbox = ndimage.find_objects(labels_img, max_label=cell_num)
                for label in range(1, cell_num + 1):
                    bbox = found_bbox[label - 1]
                    if found_bbox[label - 1] is not None:
                        intensity_label_i = raw[bbox][labels_img[bbox] == label]
                        activities_txn[t - 1, label - 1] = np.mean(intensity_label_i)
                    else:
                        activities_txn[t - 1, label - 1] = np.nan

            track_file.attrs["t_initial"] = self.T_Initial
            track_file.attrs["voxel_size_yxz"] = self.coords2image.Voxel_Size
            track_file.attrs["raw_dset"] = self.Raw_Img_Param["dset"]
            track_file.attrs["raw_channel_nuclei"] = self.Raw_Img_Param["channel_nuclei"]
            track_file.attrs["raw_channel_activity"] = self.Raw_Img_Param["channel_activity"]
            del_datasets(track_file, ["activities_txn", "coords_txnx3"])
            track_file.create_dataset("activities_txn", data=activities_txn)
            track_file.create_dataset("coords_txnx3", data=coords_txnx3) # zyx order
        self.save_activities_to_csv(activities_txn)
        self.cache.task_done('activities', str(self.tracking_results_file))

    def cache_max_projections(self, force_redo: bool = False):
        if self.cache.should_skip('cache_max_projections', force_redo=force_redo):
            return
        # Cache max projections of raw image
        with h5py.File(self.Raw_Img_Param["h5_file"], 'r+') as f_raw:
            if "max_projection_raw" not in f_raw:
                t, z, c, y, x = f_raw[self.Raw_Img_Param["dset"]].shape
                dtype = f_raw[self.Raw_Img_Param["dset"]].dtype
                max_raw_dset = f_raw.create_dataset("max_projection_raw",
                                                    (t, c, y + z * self.coords2image.Interp_Factor, x),
                                                    chunks=(1, c, y + z * self.coords2image.Interp_Factor, x),
                                                    compression="gzip", dtype=dtype, compression_opts=1)

                print("Calulating max projection of raw images...")
                for _t in tqdm(range(t)):
                    raw_img = f_raw[self.Raw_Img_Param["dset"]][_t, :, :, :, :].transpose((1, 2, 0, 3))
                    max_raw_dset[_t, ...] = np.concatenate((raw_img.max(axis=2),
                                                                 np.repeat(raw_img.max(axis=1),
                                                                           self.coords2image.Interp_Factor, axis=1)),
                                                           axis=1)
            else:
                print("Max projection of raw images already exists!")

        # Cache max projections of tracked labels
        with h5py.File(str(self.tracking_results_file), "r+") as track_file:
            if "max_projection_labels" not in track_file:
                t, z, c, y, x = track_file["tracked_labels"].shape
                dtype = track_file["tracked_labels"].dtype
                max_labels_dset = track_file.create_dataset("max_projection_labels",
                                                            (t, y + z * self.coords2image.Interp_Factor, x),
                                                            chunks=(1, y + z * self.coords2image.Interp_Factor, x),
                                                            compression="gzip", dtype=dtype, compression_opts=1)

                print("Calulating max projection of tracked labels...")
                for _t in tqdm(range(t)):
                    labels_zyx = track_file["tracked_labels"][_t, :, 0, :, :]
                    max_labels_dset[_t, ...] = np.concatenate((labels_zyx.max(axis=0),
                                                                 np.repeat(labels_zyx.max(axis=1),
                                                                           self.coords2image.Interp_Factor, axis=0)),
                                                              axis=0)
            else:
                print("Max projection of tracked labels already exists!")
        self.cache.task_done('cache_max_projections', str(self.tracking_results_file))

    def draw_activities(self, figsize: tuple, column_n: int):
        with h5py.File(str(self.tracking_results_file), 'r') as f:
            activities = f["activities_txn"][:]
        draw_signals(activities, figsize=figsize, column_n=column_n)

    def get_cell_num_initial_20_vols(self):
        cell_num_list = []
        for i in tqdm(range(20)):
            t = self.tgt_list_20[i]
            pos_ref0, _ = self._get_segmented_pos(t)
            cell_num_list.append(len(pos_ref0.real))
        return cell_num_list

    @property
    def tgt_list_20(self):
        return self.tgt_list[:20]

    @property
    def tgt_list(self):
        return np.asarray([self.T_Initial] + [tgt for _, tgt in self.order_list])

    def get_common_ids(self, cum_match_counts, threshold: int):
        pos_ref0_segmented = self._get_segmented_pos(self.T_Initial)[0].real
        pos_ref0_proofed = self.coords2image.Coord_Proof.real
        matched_ind_in_ref, matched_ind_in_proof = match_manual_ref0(pos_ref0_segmented, pos_ref0_proofed)
        top_indices = np.nonzero(cum_match_counts>=threshold)
        _, ind_ref, _ = np.intersect1d(matched_ind_in_ref, top_indices, return_indices=True)
        return matched_ind_in_ref[ind_ref], matched_ind_in_proof[ind_ref]

    def finetune_positions_in_image(self, ref_ptrs_tracked_t2, t2, max_repetition: int):
        tracked_coords_t2 = rigid_transform(self.Inv_Tform_Tx3x4[t2 - 1, ...],
                                            ref_ptrs_tracked_t2) * self.Scale_Vol1 + self.Mean_Vol1
        tracked_coords_t2_pred = Coordinates(tracked_coords_t2,
                                             interpolation_factor=self.coords2image.Interp_Factor,
                                             voxel_size=self.coords2image.Voxel_Size, dtype="real")
        confirmed_coord, corrected_labels_image = self.coords2image.accurate_correction(t2,
                                                                                        tracked_coords_t2_pred,
                                                                                        ensemble=True, max_repetition=max_repetition)
        return confirmed_coord, corrected_labels_image

    def predict_pos_all_cells(self, pairs, segmented_coords_norm_t1, segmented_coords_norm_t2):
        # Find these cells that were not matched in previous procedures
        missed_cells = np.setdiff1d(self.Common_Ids_In_Coords, pairs[:, 0], assume_unique=True)
        missed_cells, common_ind_miss, pairs_ind_miss = np.intersect1d(self.Common_Ids_In_Coords, missed_cells,
                                                                       return_indices=True)
        intersect_cells, common_ind_intersect, pairs_ind_intersect = np.intersect1d(self.Common_Ids_In_Coords,
                                                                                    pairs[:, 0], return_indices=True)
        ref_ptrs_confirmed = segmented_coords_norm_t1[self.Common_Ids_In_Coords]

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

    def save_tracking_results(self, coords: Coordinates, corrected_labels_image: ndarray, t: int,
                              images_path: dict, step: int):
        """
        Save the tracking results, including coordinates, corrected labels image, and the merged image + label
        """
        raw_img = load_2d_slices_at_time(images_path, t=t, channel_name="channel_nuclei")
        self.coords2image.save_merged_labels(corrected_labels_image, raw_img, t, self.coords2image.Cmap_Colors)

        with h5py.File(str(self.tracking_results_file), 'a') as f:
            f["tracked_labels"][t - 1, :, 0, :, :] = corrected_labels_image.transpose((2, 0, 1))
            f["tracked_coordinates"][t - 1, :, :] = coords.real
            f.attrs["last_step"] = step

    def load_normalized_coords(self, t: int, mean_3: float = None, scale: float = None):
        segmented_pos, subset = self._get_segmented_pos(t)
        if mean_3 is None or scale is None:
            segmented_coords_norm, (mean_3, scale) = normalize_points(segmented_pos.real)
        else:
            segmented_coords_norm = (segmented_pos.real - mean_3) / scale
        coords_subset_norm = segmented_coords_norm[subset]
        return coords_subset_norm, mean_3, scale, segmented_coords_norm, segmented_pos, subset

    def load_rotated_normalized_coords(self, t: int):
        if self.Mean_Vol1 is None:
            _, self.Mean_Vol1, self.Scale_Vol1, _, _, _ = self.load_normalized_coords(t=1)

        _, _, _, segmented_coords_norm, segmented_pos, subset = self.load_normalized_coords(
            t, mean_3=self.Mean_Vol1, scale=self.Scale_Vol1)
        segmented_coords_norm = rigid_transform(self.Tform_Tx3x4[t - 1,...], segmented_coords_norm)
        coords_subset_norm = segmented_coords_norm[subset]
        return coords_subset_norm, segmented_coords_norm, segmented_pos, subset

    def save_filtered_subregions(self):
        with h5py.File(str(self.tracking_results_file), "a") as track_file:
            track_file.attrs["num_cells"] = len(self.coords2image.Filtered_Subregions)
            if "subregions" in track_file:
                del track_file["subregions"]
            group = track_file.create_group("subregions")
            for i, (slices_xyz, subregion) in enumerate(self.coords2image.Filtered_Subregions):
                if f"subregion_{i+1}" in track_file:
                    del track_file[f"subregion_{i+1}"]
                if f"subregion_{i+1}" in group:
                    del group[f"subregion_{i+1}"]
                dset = group.create_dataset(f"subregion_{i+1}", data = subregion)
                dset.attrs["slice_xyz"] = (slices_xyz[0].start, slices_xyz[0].stop,
                                           slices_xyz[1].start, slices_xyz[1].stop,
                                           slices_xyz[2].start, slices_xyz[2].stop)

    def load_filtered_subregions(self):
        with h5py.File(str(self.tracking_results_file), "r") as track_file:
            n_cells = track_file.attrs["num_cells"]
            group = track_file["subregions"]
            subregions = []
            for i in range(n_cells):
                cell_i = group[f"subregion_{i + 1}"]
                x0, x1, y0, y1, z0, z1 = cell_i.attrs["slice_xyz"]
                subregions.append(((slice(x0, x1), slice(y0, y1), slice(z0, z1)), cell_i[:]))
        return subregions


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


def evenly_distributed_volumes(current_vol: int, sampling_number: int) -> List[int]:
    """Get evenly distributed previous volumes"""
    interval = (current_vol - 1) // sampling_number
    start = np.mod(current_vol - 1, sampling_number) + 1
    return list(range(start, current_vol - interval + 1, interval))


def link_pairs(pairs_t1_t0, pairs_t2_t1):
    pairs_t2_t0 = []
    for mid, ref in pairs_t1_t0:
        if mid in pairs_t2_t1[:, 1]:
            index = np.nonzero(pairs_t2_t1[:, 1] == mid)[0][0]
            pairs_t2_t0.append((pairs_t2_t1[index, 0], ref))
    return np.asarray(pairs_t2_t0)


def rotate_for_visualization(rotation_matrix, coords_norm: Tuple[ndarray]):
    coords_norm_rotated = [coords_i.copy() for coords_i in coords_norm]
    if rotation_matrix is not None:
        for i in range(len(coords_norm_rotated)):
            coords_norm_rotated[i][:, :2] = coords_norm_rotated[i][:, :2].dot(rotation_matrix)
    return coords_norm_rotated


def combine_links(matches_matrix_ini_to_tgt: ndarray, pairs_t0_ref: ndarray, pairs_ref_tgt: ndarray):
    if pairs_t0_ref is None:
        for ref, tgt in pairs_ref_tgt:
            matches_matrix_ini_to_tgt[ref, tgt] += 1
        return

    common_mid_values, pos1, pos2 = np.intersect1d(pairs_t0_ref[:, 1], pairs_ref_tgt[:, 0], return_indices=True)
    for value, p1, p2 in zip(common_mid_values, pos1, pos2):
        matches_matrix_ini_to_tgt[pairs_t0_ref[p1, 0], pairs_ref_tgt[p2, 1]] += 1
    return


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


def clear_memory():
    K.clear_session()
    gc.collect()

