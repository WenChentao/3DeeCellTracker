from pathlib import Path
from typing import Tuple, List, Set

import h5py
import numpy as np
import scipy.ndimage.measurements as ndm
import skimage.filters as skf
import skimage.measure as skm
from PIL import Image
from numpy import ndarray
from skimage.segmentation import relabel_sequential
from tifffile import imread

from CellTracker.stardistwrapper import create_cmap, plot_img_label_max_projection
from CellTracker.utils import load_2d_slices_at_time, recalculate_cell_boundaries, del_datasets

PAD_WIDTH = 3

MERGED_LABELS_XZ = "merged_labels_xz"
MERGED_LABELS = "merged_labels"
SEG = "seg"
COORDS_REAL = "coords_real"
LABELS = "labels"
TRACK_RESULTS = "track_results"


class Coordinates:
    """
        A class to represent 3D coordinates and perform transformations between different coordinate systems.

        Attributes:
            interpolation_factor (int): The interpolation factor for the z-coordinate.
            voxel_size (np.ndarray): A 1D numpy array with the voxel size in each dimension (x, y, z).
    """

    def __init__(self, coords: ndarray, interpolation_factor: int, voxel_size: ndarray, dtype: str = "raw"):
        """
        Initialize the Coordinates object with given parameters.

        Args:
            coords (np.ndarray): A 2D numpy array with shape (n, 3) representing n coordinates in 3D space.
            interpolation_factor (int): The interpolation factor for the z-coordinate.
            voxel_size (np.ndarray): A 1D numpy array with the voxel size in each dimension (x, y, z).
            dtype (str, optional): The type of coordinates provided. Can be "raw", "real", or "interp". Defaults to "raw".
        """
        self.interpolation_factor = interpolation_factor
        self.voxel_size = np.asarray(voxel_size)
        if dtype == "raw":
            self._raw = coords.astype(np.float32)
        elif dtype == "real":
            self._raw = self._transform_z(coords.astype(np.float32), 1.0 / self.voxel_size).astype(np.float32)
        elif dtype == "interp":
            self._raw = self._transform_z(
                coords.astype(np.float32), np.asarray((1, 1, 1 / interpolation_factor))
            ).astype(np.float32)

    def __add__(self, other: 'Coordinates') -> 'Coordinates':
        """
        Add two Coordinates objects.

        Args:
            other (Coordinates): Another Coordinates object.

        Returns:
            Coordinates: A new Coordinates object representing the addition of the two input Coordinates objects.
        """
        return Coordinates(self._raw + other._raw, self.interpolation_factor, self.voxel_size, 'raw')

    def __sub__(self, other: 'Coordinates') -> 'Coordinates':
        """
        Subtract one Coordinates object from another.

        Args:
            other (Coordinates): Another Coordinates object.

        Returns:
            Coordinates: A new Coordinates object representing the difference between the two input Coordinates objects.
        """
        return Coordinates(self._raw - other._raw, self.interpolation_factor, self.voxel_size, 'raw')

    @staticmethod
    def _transform_z(coords_nx3: ndarray, factor_x3: ndarray) -> np.ndarray:
        """
        Transform the z-coordinate by scaling with factor.

        Args:
            coords_nx3 (np.ndarray): A 2D numpy array with shape (n, 3) representing n coordinates in 3D space.
            factor_x3 (np.ndarray): A 1D numpy array with scaling factors for each dimension (x, y, z).

        Returns:
            np.ndarray: A 2D numpy array with the transformed coordinates.
        """
        new_coords = coords_nx3.copy()
        new_coords = new_coords * factor_x3[None, :]
        return new_coords

    @property
    def real(self) -> np.ndarray:
        """
        Convert raw coordinates to real coordinates.

        Returns:
            np.ndarray: A 2D numpy array with the real coordinates.
        """
        return self._transform_z(self._raw, self.voxel_size)

    @property
    def interp(self) -> np.ndarray:
        """
        Convert raw coordinates to interpolated coordinates.

        Returns:
            np.ndarray: A 2D numpy array    with the interpolated coordinates.
        """
        return np.round(
            self._transform_z(
                self._raw, np.asarray((1, 1, self.interpolation_factor))
            )
        ).astype(np.int32)

    @property
    def raw(self):
        """
        Convert the internal raw coordinates to integer values.

        Returns:
            np.ndarray: A 2D numpy array with the raw coordinates as integers.
        """
        return np.round(self._raw).astype(np.int32)

    @property
    def cell_num(self) -> int:
        """
        Get the number of cells (coordinates) in the Coordinates object.

        Returns:
            int: The number of cells (coordinates) in the Coordinates object.
        """
        return self._raw.shape[0]


def mix_img_labels(corrected_labels_image: ndarray, raw_img: ndarray, interpolation_factor: int, alpha: float, cmap_colors, cutoff=(70, 99.5)):
    labels_rgb = cmap_colors[corrected_labels_image.max(axis=2)]
    labels_rgb = Image.fromarray((labels_rgb * 255).astype(np.uint8))
    labels_rgb_xz = cmap_colors[corrected_labels_image.max(axis=0)].transpose(1, 0, 2)
    labels_rgb_xz = np.repeat(labels_rgb_xz, interpolation_factor, axis=0)
    labels_rgb_xz = Image.fromarray((labels_rgb_xz * 255).astype(np.uint8))
    raw_img_xy = np.max(raw_img, axis=0)
    p_bottom, p_top = np.percentile(raw_img_xy, cutoff[0]), np.percentile(raw_img_xy, cutoff[1])
    normalized_raw_xy = (np.clip(raw_img_xy, p_bottom, p_top) * 255 / p_top).astype(np.uint8)
    raw_rgb = Image.fromarray(normalized_raw_xy).convert('RGB')
    raw_img_xz = np.max(raw_img, axis=1)
    raw_img_xz = np.repeat(raw_img_xz, interpolation_factor, axis=0)
    normalized_raw_xz = (np.clip(raw_img_xz, p_bottom, p_top) * 255 / p_top).astype(np.uint8)
    raw_rgb_xz = Image.fromarray(normalized_raw_xz).convert('RGB')
    merged_labels = Image.blend(labels_rgb, raw_rgb, alpha=alpha)
    merged_labels_xz = Image.blend(labels_rgb_xz, raw_rgb_xz, alpha=alpha)
    return merged_labels, merged_labels_xz


class CoordsToImageTransformer:
    """
    A class to transform cell coordinates into an image with the cells represented as labeled regions.
    """
    auto_corrected_segmentation: ndarray
    subregions: List[Tuple[Tuple[slice, slice, slice], ndarray]]
    proofed_segmentation: ndarray
    z_slice_original_labels: slice
    coord_proof: Coordinates
    use_8_bit: bool

    def __init__(self, results_folder: str, image_pars: dict):
        """
        Initialize the CoordsToImageTransformer with a specified results folder and voxel size.

        Parameters
        ----------
        results_folder : str
            Path to the folder where the results will be saved.
        voxel_size : tuple
            Voxel size in the format (x, y, z).
        """
        self.voxel_size = np.asarray(image_pars["voxel_size"])
        self.image_pars = image_pars
        self._cal_interp_factor()
        self.results_folder = Path(results_folder)

    def _cal_interp_factor(self):
        x_siz, y_siz, z_siz = self.image_pars["voxel_size_xyz"]
        self.interpolation_factor = int(np.round(z_siz / (x_siz * y_siz) ** 0.5))
        if self.interpolation_factor < 1:
            self.interpolation_factor = 1

    def load_segmentation(self, manual_seg_path: str, proof_vol: int) -> None:
        """
        Load the proofed segmentation from a single tif file.

        Parameters
        ----------
        manual_seg_path : str
            The path to the tif file of manually corrected segmentation.
        proof_vol: int
            The volume number of the manually corrected segmentation
        """
        # Load the proofed segmentation and relabel it to sequential integers
        proofed_segmentation = imread(manual_seg_path).transpose((1, 2, 0))
        self.proofed_segmentation, _, _ = relabel_sequential(proofed_segmentation)
        self.proofed_coords_vol = proof_vol

        # Print a message confirming that the proofed segmentation has been loaded and its shape
        print(
            f"Loaded the proofed segmentations with {np.count_nonzero(np.unique(proofed_segmentation))} cells")

    def update_filtered_segmentation(self, filtered_ids, images_path: dict, t_initial: int) -> None:
        """Remove cells with ids not in filtered_ids"""
        _trans_array = np.zeros((self.auto_corrected_segmentation.max()+1), dtype=int)
        _trans_array[filtered_ids] = np.arange(1, len(filtered_ids)+1)
        self.updated_proofed_segmentation = _trans_array[self.auto_corrected_segmentation]
        self.update_subregions(filtered_ids)

        h, w, z = self.proofed_segmentation.shape
        with h5py.File(images_path["h5_file"], 'r+') as f_raw:
            t = f_raw[images_path["dset"]].shape[0]
        resolution_h, resolution_w, resolution_z = self.image_pars["voxel_size"]

        dtype = np.uint8 if self.use_8_bit else np.uint16

        with h5py.File(str(self.results_folder / 'tracking_results.h5'), 'a') as f:
            if 'tracked_labels' not in f:
                f.create_dataset('tracked_labels', (t, z, 1, h, w), chunks=(1, z, 1, h, w),
                    compression="gzip", dtype=dtype, compression_opts=1)  # "lzf" compression can cause error in Fiji
            f["tracked_labels"].attrs['element_size_um'] = (resolution_z, resolution_h, resolution_w)
            if 'tracked_coordinates' not in f:
                f.create_dataset('tracked_coordinates', (t, *self.coord_proof.real.shape),
                                         compression="gzip", compression_opts=1)
            assert f["tracked_labels"].shape == (t, z, 1, h, w)
            assert f["tracked_coordinates"].shape == (t, *self.coord_proof.real.shape)

        with h5py.File(str(self.results_folder / 'tracking_results.h5'), 'a') as f:
            f["tracked_labels"][t_initial - 1, :, 0, :, :] = self.final_segmentation.transpose((2, 0, 1)).astype(dtype)
            f["tracked_coordinates"][t_initial - 1, :, :] = self.coord_proof.real

        cmap = create_cmap(self.final_segmentation, self.image_pars, max_color=20, dist_type="2d_projection")
        self.cmap_colors = np.zeros((len(cmap.colors), 3))
        for i in range(1, len(cmap.colors)):
            self.cmap_colors[i, :] = cmap.colors[i][:3]
        raw_img = load_2d_slices_at_time(images_path, t=t_initial)
        self.save_merged_labels(self.final_segmentation, raw_img, t_initial, self.cmap_colors)

        # Print a message confirming that the proofed segmentation has been loaded and its shape
        print(f"updated the proofed segmentations with {np.count_nonzero(np.unique(self.final_segmentation))} cells")

    def interpolate_labels(self, smooth_sigma: float = 2.5) -> None:
        """
        Interpolate the images along z axis and save the results in "track_results_xxx" folder

        Parameters
        ----------
        smooth_sigma : float, optional
            The sigma value to use for smoothing the interpolated images, by default 2.5.

        Raises
        ------
        ValueError
            If `interpolation_factor` is less than or equal to zero.

        Notes
        -----
        The image is interpolated and smoothed if z_scaling > 1, or smoothed if z_scaling = 1 by a Gaussian filter.
        After interpolation/smoothing, the cell boundaries are reassigned by 2d watershed. Then the interpolated cells
        are re-segmented by 3d connectivity to separate cells incorrectly labelled as the same cell.
        """

        def extract_regions(segmentation: ndarray):
            self.subregions = extract_cells_after_gaussian_filter(
                segmentation, interpolation_factor=interpolation_factor, smooth_sigma=smooth_sigma)

            # Get interpolated labels image
            interpolated_labels, cell_overlaps_mask = self.move_cells(self.subregions, self. proofed_segmentation, movements_nx3=None)

            # Recalculate cell boundaries
            auto_corrected_segmentation = recalculate_cell_boundaries(
                interpolated_labels[:, :, self.z_slice_original_labels],
                cell_overlaps_mask[:, :, self.z_slice_original_labels],
                sampling_xy=self.voxel_size[:2])
            return auto_corrected_segmentation

        interpolation_factor = self.interpolation_factor
        if interpolation_factor <= 0:
            raise ValueError("Interpolation factor must be greater than zero.")

        print("Interpolating images along z-axis...")

        # Calculate the original z-slice labels
        self.z_slice_original_labels = slice(interpolation_factor // 2,
                                             interpolation_factor * self.proofed_segmentation.shape[2],
                                             interpolation_factor)

        smoothed_labels = extract_regions(self.proofed_segmentation)

        # Fix segmentation errors
        corrected_segmentation, _ = fix_labeling_errors(smoothed_labels)

        self.auto_corrected_segmentation = extract_regions(corrected_segmentation)

        # Check if 8-bit is needed for saving the image
        self.use_8_bit = self.auto_corrected_segmentation.max() <= 255
        print(
            f"The interpolated segmentations contains {np.count_nonzero(np.unique(self.auto_corrected_segmentation))} cells")

        print("Calculating coordinates of cell centers...")
        # Calculate coordinates of cell centers at t=1
        coord_proof = ndm.center_of_mass(
            self.auto_corrected_segmentation > 0,
            self.auto_corrected_segmentation,
            range(1, self.auto_corrected_segmentation.max() + 1)
        )
        self.coord_proof = Coordinates(np.asarray(coord_proof), interpolation_factor, self.voxel_size, dtype="raw")

    def update_subregions(self, filtered_ids) -> None:
        updated_subregions = []
        for id in filtered_ids:
            updated_subregions.append(self.subregions[id - 1])
        self.updated_subregions = updated_subregions

        # Get interpolated labels image
        updated_labels, cell_overlaps_mask = self.move_cells(self.updated_subregions, self.updated_proofed_segmentation, movements_nx3=None)

        # Recalculate cell boundaries
        self.final_segmentation = recalculate_cell_boundaries(
            updated_labels[:, :, self.z_slice_original_labels],
            cell_overlaps_mask[:, :, self.z_slice_original_labels],
            sampling_xy=self.voxel_size[:2])

        # Check if 8-bit is needed for saving the image
        self.use_8_bit = self.final_segmentation.max() <= 255
        print(
            f"The updated segmentations contains {np.count_nonzero(np.unique(self.final_segmentation))} cells")

        # Calculate coordinates of cell centers at t=1
        print("Calculating coordinates of cell centers...")
        coord_proof = ndm.center_of_mass(
            self.final_segmentation > 0,
            self.final_segmentation,
            range(1, self.final_segmentation.max() + 1)
        )
        self.coord_proof = Coordinates(np.asarray(coord_proof), self.interpolation_factor, self.voxel_size, dtype="raw")

    def move_cells_in_3d_image(self, movements_nx3: ndarray = None, cells_missed: Set[int] = None):
        """
        Move cells in the 3D image according to the provided movements_nx3 and recalculate cell boundaries.

        Parameters
        ----------
        movements_nx3 : Optional[ndarray], default=None
            A 2D NumPy array of size n x 3, containing the movements for each cell. If not provided, no movements will be applied.
        cells_missed : Optional[Set[int]], default=None
            A set of cell indices that were missed during the tracking process. If not provided, all cells are assumed to be tracked.

        Returns
        -------
        output : ndarray
            The new image with moved cells and recalculated cell boundaries.
        """
        interpolated_labels, cell_overlaps_mask = self.move_cells(self.updated_subregions, self.updated_proofed_segmentation,
                                                                  movements_nx3=movements_nx3,
                                                                  cells_missed=cells_missed)
        return recalculate_cell_boundaries(
            interpolated_labels[:, :, self.z_slice_original_labels],
            cell_overlaps_mask[:, :, self.z_slice_original_labels],
            sampling_xy=self.voxel_size[:2], print_message=False)

    def move_cells(self, subregions: list, proofed_segmentation: ndarray, movements_nx3: ndarray = None, cells_missed: Set[int] = None):
        """
        Generate an image with labels indicating the moved cells and a mask image indicating the overlapping regions

        Parameters
        ----------
        movements_nx3 : ndarray, optional
            Movements of each cell, by default None.
        cells_missed : Set[int], optional
            A set of cell indices that were missed during the tracking process, by default None.

        Returns
        -------
        output : numpy.ndarray
            The new image with moved cells.
        mask : numpy.ndarray
            The new image with the overlapping regions.
        """

        return move_cells(
            subregions,
            proofed_segmentation,
            self.interpolation_factor,
            movements_nx3,
            cells_missed
        )

    def get_cells_on_boundary(self, coordinates_real_nx3: ndarray, ensemble: bool, boundary_xy: int = 6):
        """
        Get the indices of cells that are on the boundary of the image.

        Parameters
        ----------
        coordinates_real_nx3 : ndarray
            A 2D NumPy array of size n x 3 containing the real coordinates of the cells.
        ensemble : bool
            If True, the boundary_xy is ignored.
        boundary_xy : int, default=6
            Used to further reduce the size of the image boundary in the xy plane. Ignored if ensemble is True.

        Returns
        -------
        boundary_ids : ndarray
            A 1D NumPy array containing the indices of cells that are on the boundary.
        """
        if ensemble:
            boundary_xy = 0

        x_siz, y_siz, z_siz = self.proofed_segmentation.shape
        x, y, z = coordinates_real_nx3.T

        near_boundary = (
                (x < boundary_xy) |
                (y < boundary_xy) |
                (x > (x_siz - boundary_xy) * self.voxel_size[0]) |
                (y > (y_siz - boundary_xy) * self.voxel_size[1]) |
                (z < 0) |
                (z > z_siz * self.voxel_size[2])
        )
        boundary_ids = np.where(near_boundary)[0] + 1
        return boundary_ids

    def accurate_correction(self, t: int, grid: Tuple[int, int, int], coords: Coordinates, ensemble: bool,
                            max_repetition: int = 20):
        """
        Correct center positions of cells based on the probability map.

        Parameters
        ----------
        t : int
            The time point at which the correction is performed.
        grid : Tuple[int, int, int]
            A tuple representing the grid used in generating the probability map
        coords : Coordinates
            The coordinates of cell centers.
        ensemble : bool
            A boolean indicating whether the tracking is based on ensemble mode
        max_repetition : int, optional
            The maximum number of repetitions for the correction process, by default 20.

        Returns
        -------
        coords : Coordinates
            The corrected coordinates of cell centers.
        corrected_labels_image : ndarray
            The corrected labels image.
        """
        with h5py.File(str(self.results_folder / "seg.h5"), "r") as seg_file:
            prob_map = self.load_prob_map(grid, t - 1, seg_file, self.proofed_segmentation.shape)

        boundary_ids = set(self.get_cells_on_boundary(coords.real, ensemble=ensemble).tolist())

        for i in range(max_repetition):
            # update positions by correction
            coords, delta_coords = self._correction_once(prob_map, coords, boundary_ids)

            # stop the repetition if correction converged
            if np.max(delta_coords.interp) < 0.5:
                break
        corrected_labels_image = self.move_cells_in_3d_image((coords - self.coord_proof).interp, boundary_ids)
        return coords, corrected_labels_image

    def load_prob_map(self, grid: tuple, t: int, h5_file, image_size_xyz: tuple):
        prob_map = h5_file["prob"][t, ...].transpose((1,2,0))
        prob_map = np.repeat(np.repeat(np.repeat(prob_map, grid[1], axis=0), grid[2], axis=1), grid[0], axis=2)
        if prob_map.shape != image_size_xyz:
            x_lim, y_lim, z_lim = image_size_xyz
            prob_map = prob_map[:x_lim, :y_lim, :z_lim]
        return prob_map

    def _correction_once(self, prob_img: ndarray, coords: Coordinates, boundary_ids: Set[int]):
        """
        Perform one correction iteration on the coordinates based on the probability image and the
        provided coordinates.

        Parameters
        ----------
        prob_img : ndarray
            The probability image as a 3D array.
        coords : Coordinates
            The Coordinates object containing the original 3D cell coordinates.
        boundary_ids : Set[int]
            A set of integers representing the IDs of cells on the boundary.

        Returns
        -------
        corrected_coords : Coordinates
            The corrected 3D cell coordinates after applying the correction.
        delta_coords : Coordinates
            The difference in 3D cell coordinates between the corrected and original coordinates.
        """
        # generate labels image after applying the movements
        displacements_from_vol1 = coords - self.coord_proof
        labels_image_interp, mask_image_interp = self.move_cells(self.updated_subregions, self.updated_proofed_segmentation,
                                                                 displacements_from_vol1.interp, boundary_ids)
        labels_image = labels_image_interp[:, :, self.z_slice_original_labels]
        mask_image = mask_image_interp[:, :, self.z_slice_original_labels]

        # remove the overlapped regions from each label (marker for watershed)
        labels_image[mask_image > 1] = 0
        positions_of_new_centers = ndm.center_of_mass(prob_img, labels_image,
                                                      range(1, self.final_segmentation.max() + 1))
        positions_of_new_centers = np.asarray(positions_of_new_centers)

        lost_cells = np.isnan(positions_of_new_centers[:, 0])
        positions_of_new_centers[lost_cells, :] = coords.raw[lost_cells, :]

        corrected_coords = Coordinates(positions_of_new_centers, self.interpolation_factor, self.voxel_size,
                                       dtype="raw")
        delta_coords = corrected_coords - coords

        return corrected_coords, delta_coords

    def save_merged_labels(self, corrected_labels_image: ndarray, raw_img: ndarray, t: int, cmap_colors):
        """
        Save the merged labels, which is an overlay of the labels and raw images.

        Parameters
        ----------
        corrected_labels_image : ndarray
            The corrected labels image.
        raw_img : ndarray
            The raw images.
        t : int
            The time point at which the merged labels are saved.
        """
        merged_labels, merged_labels_xz = mix_img_labels(
            corrected_labels_image, raw_img, self.interpolation_factor, alpha=0.6, cmap_colors=cmap_colors)

        (self.results_folder / MERGED_LABELS).mkdir(parents=True, exist_ok=True)
        (self.results_folder / MERGED_LABELS_XZ).mkdir(parents=True, exist_ok=True)
        merged_labels.save(str(self.results_folder / MERGED_LABELS / ("merged_labels_t%06d.png" % t)))
        merged_labels_xz.save(
            str(self.results_folder / MERGED_LABELS_XZ / ("merged_labels_xz_t%06d.png" % t)))


def save_tracked_labels(results_folder: Path, labels_xyz: ndarray, t: int, use_8_bit: bool):
    """
    Save tracked label images to the specified folder.

    Parameters
    ----------
    results_folder : Path
        The path to the folder where the tracked labels will be saved.
    labels_xyz : ndarray
        A 3D NumPy array containing the label images.
    t : int
        The time step of the tracked labels.
    use_8_bit : bool
        If True, the images will be saved as 8-bit images; otherwise, as 16-bit images.
    """
    tracked_labels_path = results_folder / TRACK_RESULTS / LABELS
    tracked_labels_path.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if use_8_bit else np.uint16

    for z in range(1, labels_xyz.shape[2] + 1):
        img2d = labels_xyz[:, :, z - 1].astype(dtype)
        Image.fromarray(img2d).save(str(tracked_labels_path / ("track_results_t%06i_z%04i.tif" % (t, z))))


def extract_cells_after_gaussian_filter(label_image, interpolation_factor=10, smooth_sigma=5) -> \
        List[Tuple[Tuple[slice, slice, slice], ndarray]]:
    """
    Generate interpolated/smoothed cell regions.

    Parameters
    ----------
    label_image : ndarray
        A 3D NumPy array containing the label image.
    interpolation_factor : int, default=10
        The factor of interpolations along the z-axis.
    smooth_sigma : float, default=5
        The sigma used for Gaussian blur.

    Returns
    -------
    subregions : List[Tuple[Tuple[slice, slice, slice], ndarray]]
        A list of tuples, where each tuple contains a tuple of slices (representing the bounding box)
        and a NumPy array of the smoothed label image for the corresponding cell.
    """
    bboxes: List[Tuple[slice, slice, slice]] = ndm.find_objects(label_image)
    subregions = []

    for label in range(1, np.max(label_image) + 1):
        print(f"Interpolating... cell:{label}", end="\r")
        bbox = bboxes[label - 1]
        sub_img = (label_image[bbox] == label).astype(np.float32)
        repeated_sub_img = np.pad(np.repeat(sub_img, interpolation_factor, axis=2), PAD_WIDTH)
        img_smooth = skf.gaussian(repeated_sub_img,
                                  sigma=smooth_sigma, mode='constant')
        percentage = 1 - np.count_nonzero(sub_img) / (img_smooth.size/interpolation_factor)
        threshold = np.percentile(img_smooth, percentage * 100)
        interpolated_bbox = (slice(bbox[0].start - PAD_WIDTH,
                                   bbox[0].stop + PAD_WIDTH,
                                   None),
                             slice(bbox[1].start - PAD_WIDTH,
                                   bbox[1].stop + PAD_WIDTH,
                                   None)
                             ,
                             slice(bbox[2].start * interpolation_factor - PAD_WIDTH,
                                   bbox[2].stop * interpolation_factor + PAD_WIDTH,
                                   None)
                             )
        subregions.append((interpolated_bbox, img_smooth > threshold))
    return subregions


def fix_labeling_errors(segmentation: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Relabel the separated cells that were incorrectly labeled as the same one.

    Parameters
    ----------
    segmentation : ndarray
        A 3D NumPy array containing the input label image with separated cells that were labeled as the same one.

    Returns
    -------
    new_segmentation : ndarray
        A 3D NumPy array containing the corrected label image.
    was_corrected : bool
        True if the segmentation was corrected; False otherwise.
    """
    num_cells = np.size(np.unique(segmentation)) - 1
    new_segmentation = skm.label(segmentation, connectivity=3)
    was_corrected = num_cells != np.max(new_segmentation)
    if was_corrected:
        print(f"WARNING: The number of cells in the manually labeled segmentation ({num_cells}) does not match "
              f"the number of separated cells found by the program ({np.max(new_segmentation)}). "
              f"The program has corrected the segmentation accordingly.")
    return new_segmentation, was_corrected


def move_cells(
    subregions: List[Tuple[Tuple[slice, slice, slice], ndarray]],
    segmentation_t1: ndarray,
    interpolation_factor: int,
    movements_nx3: ndarray = None,
    cells_missed: List[int] = None
) -> Tuple[ndarray, ndarray]:
    """
    Parameters
    ----------
    subregions : List[Tuple[Tuple[slice, slice, slice], ndarray]]
        A list of tuples, each containing a bounding box and the corresponding subimage.
    segmentation_t1 : ndarray
        A 3D array representing the segmentation to be moved.
    interpolation_factor : int
        A factor by which the z-axis of the segmentation is interpolated.
    movements_nx3 : ndarray, optional
        Movements of each cell, by default None.
    cells_missed : List[int], optional
        A list of cell indices that were missed during the tracking process, by default None.

    Returns
    -------
    output : numpy.ndarray
        The new image with moved cells.
    mask : numpy.ndarray
        The new image with the overlapping regions.
    """
    # Initialize movements_nx3 and cells_missed if they are None
    if movements_nx3 is None:
        movements_nx3 = np.zeros((len(subregions), 3))
    if cells_missed is None:
        cells_missed = set()

    # Initialize output images
    output_img = np.repeat(np.zeros_like(segmentation_t1), interpolation_factor, axis=2)
    mask = output_img.copy()

    # Calculate interpolated shape
    siz_x, siz_y, siz_z = segmentation_t1.shape
    interp_shape = (siz_x, siz_y, siz_z * interpolation_factor)

    # Iterate over subregions and apply movements
    for i, (bbox, subimage) in enumerate(subregions):
        label = i + 1
        if label in cells_missed:
            continue
        bbox_moved, partial_bbox = add_bbox_with_movements(bbox, movements_nx3[i], interp_shape)
        output_img[bbox_moved] += (subimage * label).astype(output_img.dtype)[partial_bbox]
        mask[bbox_moved] += (subimage * 1).astype(mask.dtype)[partial_bbox]

    return output_img, mask


def add_bbox_with_movements(bbox: Tuple[slice, slice, slice], movements: ndarray, image_shape: tuple):
    """
    Add movements to the start and stop indices of the given slices and return the updated slices
    and the partial slices indicating how to clip the bbox when the target is out of range.

    Parameters
    ----------
    bbox : Tuple[slice, slice, slice]
        A tuple of slices representing the bounding box.
    movements : ndarray
        A 1x3 NumPy array representing the movements to apply to the bounding box.
    image_shape : tuple
        A tuple of integers representing the shape of the 3D image.

    Returns
    -------
    new_bbox : Tuple[slice, slice, slice]
        The updated bounding box with applied movements.
    partial_bbox : Tuple[slice, slice, slice]
        The partial slices indicating how to clip the bounding box when the target is out of range.
    """
    if len(bbox) != 3 or len(movements) != 3 or len(image_shape) != 3:
        raise ValueError("bbox, movements_1x3 and image_shape must be (3,) shape")

    new_bbox = []
    partial_bbox = []
    for s, c, size in zip(bbox, movements, image_shape):
        new_start_ = s.start + int(c)
        new_start = max(new_start_, 0)
        partial_start = new_start - new_start_
        new_stop_ = s.stop + int(c)
        new_stop = min(new_stop_, size)
        partial_stop = (s.stop - s.start) - (new_stop_ - new_stop)
        new_bbox.append(slice(new_start, new_stop, None))
        partial_bbox.append(slice(partial_start, partial_stop, None))
        if new_start >= new_stop:
            raise ValueError(f"Slices are out of range for image of size {image_shape}")

    return tuple(new_bbox), tuple(partial_bbox)
