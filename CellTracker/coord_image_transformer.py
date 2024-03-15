from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Tuple, List, Set

import numpy as np
import scipy.ndimage.measurements as ndm
import skimage.filters as skf
import skimage.measure as skm
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from numpy import ndarray
from skimage.segmentation import relabel_sequential
from tifffile import imread

from CellTracker.stardistwrapper import lbl_cmap, load_2d_slices_at_time
from CellTracker.watershed import recalculate_cell_boundaries

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


class CoordsToImageTransformer:
    """
    A class to transform cell coordinates into an image with the cells represented as labeled regions.
    """
    auto_corrected_segmentation: ndarray
    subregions: List[Tuple[Tuple[slice, slice, slice], ndarray]]
    proofed_segmentation: ndarray
    z_slice_original_labels: slice
    coord_vol1: Coordinates
    use_8_bit: bool

    def __init__(self, results_folder: str, voxel_size: tuple):
        """
        Initialize the CoordsToImageTransformer with a specified results folder and voxel size.

        Parameters
        ----------
        results_folder : str
            Path to the folder where the results will be saved.
        voxel_size : tuple
            Voxel size in the format (x, y, z).
        """
        self.voxel_size = np.asarray(voxel_size)
        self.results_folder = Path(results_folder)

    def load_segmentation(self, manual_vol_start_path: str) -> None:
        """
        Load the proofed segmentation from a directory containing image slices.

        Parameters
        ----------
        manual_vol_start_path : str
            The path to the image slices of the manually corrected segmentation.
        """
        # Get list of paths to image slices
        slice_paths = sorted(glob(manual_vol_start_path))
        if len(slice_paths) == 0:
            # Raise an error if no image slices are found in the specified directory
            raise FileNotFoundError(f"No image in {manual_vol_start_path} was found")

        # Load the proofed segmentation and relabel it to sequential integers
        proofed_segmentation = imread(slice_paths).transpose((1, 2, 0))
        self.proofed_segmentation, _, _ = relabel_sequential(proofed_segmentation)

        # Print a message confirming that the proofed segmentation has been loaded and its shape
        print(
            f"Loaded the proofed segmentations at vol 1 with {np.count_nonzero(np.unique(proofed_segmentation))} cells")

    def interpolate(self, interpolation_factor: int, smooth_sigma: float = 2.5, t_start=1) -> None:
        """
        Interpolate the images along z axis and save the results in "track_results_xxx" folder

        Parameters
        ----------
        interpolation_factor : int
            The factor by which to interpolate the images along the z-axis.
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
            self.subregions = gaussian_interpolation_3d(
                segmentation, interpolation_factor=interpolation_factor, smooth_sigma=smooth_sigma)

            # Get interpolated labels image
            interpolated_labels, cell_overlaps_mask = self.move_cells(movements_nx3=None)

            # Recalculate cell boundaries
            auto_corrected_segmentation = recalculate_cell_boundaries(
                interpolated_labels[:, :, self.z_slice_original_labels],
                cell_overlaps_mask[:, :, self.z_slice_original_labels],
                sampling_xy=self.voxel_size[:2])
            return self.subregions, auto_corrected_segmentation

        if interpolation_factor <= 0:
            raise ValueError("Interpolation factor must be greater than zero.")

        print("Interpolating images along z-axis...")

        # Interpolate layers in z-axis
        self.interpolation_factor = interpolation_factor

        # Calculate the original z-slice labels
        self.z_slice_original_labels = slice(interpolation_factor // 2,
                                             interpolation_factor * self.proofed_segmentation.shape[2],
                                             interpolation_factor)

        _, smoothed_labels = extract_regions(self.proofed_segmentation)

        # Fix segmentation errors
        corrected_segmentation, _ = fix_labeling_errors(smoothed_labels)

        self.subregions, self.auto_corrected_segmentation = extract_regions(corrected_segmentation)

        # Check if 8-bit is needed for saving the image
        self.use_8_bit = self.auto_corrected_segmentation.max() <= 255
        print(
            f"The interpolated segmentations at vol 1 contains {np.count_nonzero(np.unique(self.auto_corrected_segmentation))} cells")

        # Save labels in the first volume (interpolated)
        save_tracked_labels(self.results_folder, self.auto_corrected_segmentation, t=t_start, use_8_bit=self.use_8_bit)

        print("Calculating coordinates of cell centers...")
        # Calculate coordinates of cell centers at t=1
        coord_vol1 = ndm.center_of_mass(
            self.auto_corrected_segmentation > 0,
            self.auto_corrected_segmentation,
            range(1, self.auto_corrected_segmentation.max() + 1)
        )
        self.coord_vol1 = Coordinates(np.asarray(coord_vol1), interpolation_factor, self.voxel_size, dtype="raw")
        coords_real_path = self.results_folder / TRACK_RESULTS / COORDS_REAL
        coords_real_path.mkdir(parents=True, exist_ok=True)
        np.save(str(coords_real_path / ("coords%06d.npy" % t_start)), self.coord_vol1.real)

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
        interpolated_labels, cell_overlaps_mask = self.move_cells(movements_nx3=movements_nx3,
                                                                  cells_missed=cells_missed)
        return recalculate_cell_boundaries(
            interpolated_labels[:, :, self.z_slice_original_labels],
            cell_overlaps_mask[:, :, self.z_slice_original_labels],
            sampling_xy=self.voxel_size[:2], print_message=False)

    def move_cells(self, movements_nx3: ndarray = None, cells_missed: Set[int] = None):
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

        if movements_nx3 is None:
            movements_nx3 = np.zeros((len(self.subregions), 3))
        else:
            assert movements_nx3.shape[0] == len(self.subregions)
        if cells_missed is None:
            cells_missed = []

        output_img = np.repeat(np.zeros_like(self.proofed_segmentation), self.interpolation_factor, axis=2)
        mask = output_img.copy()
        siz_x, siz_y, siz_z = self.proofed_segmentation.shape
        interp_shape = (siz_x, siz_y, siz_z * self.interpolation_factor)
        for i, (bbox, subimage) in enumerate(self.subregions):
            label = i + 1
            if label in cells_missed:
                continue
            bbox_moved, partial_bbox = add_bbox_with_movements(bbox, movements_nx3[i], interp_shape)
            output_img[bbox_moved] += (subimage * label).astype(output_img.dtype)[partial_bbox]
            mask[bbox_moved] += (subimage * 1).astype(mask.dtype)[partial_bbox]
        return output_img, mask

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
                            max_repetition: int = 20, format = "prob%06d.npy"):
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
        prob_map = np.load(str(self.results_folder / SEG / (format % t)))
        prob_map = np.repeat(np.repeat(np.repeat(prob_map, grid[1], axis=0), grid[2], axis=1), grid[0], axis=2)
        if prob_map.shape != self.proofed_segmentation.shape:
            x_lim, y_lim, z_lim = self.proofed_segmentation.shape
            prob_map = prob_map[:x_lim, :y_lim, :z_lim]

        boundary_ids = set(self.get_cells_on_boundary(coords.real, ensemble=ensemble).tolist())

        for i in range(max_repetition):
            # update positions by correction
            coords, delta_coords = self._correction_once(prob_map, coords, boundary_ids)

            # stop the repetition if correction converged
            if np.max(delta_coords.interp) < 0.5:
                break
        corrected_labels_image = self.move_cells_in_3d_image((coords - self.coord_vol1).interp, boundary_ids)
        return coords, corrected_labels_image

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
        displacements_from_vol1 = coords - self.coord_vol1
        labels_image_interp, mask_image_interp = self.move_cells(displacements_from_vol1.interp, boundary_ids)
        labels_image = labels_image_interp[:, :, self.z_slice_original_labels]
        mask_image = mask_image_interp[:, :, self.z_slice_original_labels]

        # remove the overlapped regions from each label (marker for watershed)
        labels_image[mask_image > 1] = 0
        positions_of_new_centers = ndm.center_of_mass(prob_img, labels_image,
                                                      range(1, self.auto_corrected_segmentation.max() + 1))
        positions_of_new_centers = np.asarray(positions_of_new_centers)

        lost_cells = np.isnan(positions_of_new_centers[:, 0])
        positions_of_new_centers[lost_cells, :] = coords.raw[lost_cells, :]

        corrected_coords = Coordinates(positions_of_new_centers, self.interpolation_factor, self.voxel_size,
                                       dtype="raw")
        delta_coords = corrected_coords - coords

        return corrected_coords, delta_coords

    def save_tracking_results(self, coords: Coordinates, corrected_labels_image: ndarray, tracker, t1: int, t2: int,
                              images_path: str | dict):
        """
        Save the tracking results, including coordinates, corrected labels image, and a visualization of the matching
        process.

        Parameters
        ----------
        coords : Coordinates
            The corrected coordinates of cell centers.
        corrected_labels_image : ndarray
            The corrected labels image.
        tracker : TrackerLite
            The tracker object responsible for tracking cells.
        t1 : int
            The first time point.
        t2 : int
            The second time point.
        images_path : str
            The path to the directory containing the raw images.
        """
        np.save(str(self.results_folder / TRACK_RESULTS / COORDS_REAL / ("coords%06d.npy" % t2)), coords.real)
        save_tracked_labels(self.results_folder, corrected_labels_image, t2, self.use_8_bit)
        self.save_merged_labels(corrected_labels_image, images_path, t2)

        confirmed_coord_t1 = np.load(
            str(self.results_folder / TRACK_RESULTS / COORDS_REAL / f"coords{str(t1).zfill(6)}.npy"))
        segmented_pos_t2 = tracker._get_segmented_pos(t2)
        fig = plot_prgls_prediction(confirmed_coord_t1, segmented_pos_t2.real, coords.real, t1, t2)
        fig.savefig(self.results_folder / TRACK_RESULTS / "figure" / f"matching_{str(t2).zfill(6)}.png",
                    facecolor='white')
        plt.close()

    def save_merged_labels(self, corrected_labels_image: ndarray, images_path: str | dict, t: int):
        """
        Save the merged labels, which is an overlay of the labels and raw images.

        Parameters
        ----------
        corrected_labels_image : ndarray
            The corrected labels image.
        images_path : str
            The path to the directory containing the raw images.
        t : int
            The time point at which the merged labels are saved.
        """
        labels_rgb = lbl_cmap.colors[corrected_labels_image.max(axis=2)]
        labels_rgb = Image.fromarray((labels_rgb * 255).astype(np.uint8))

        labels_rgb_xz = lbl_cmap.colors[corrected_labels_image.max(axis=0)].transpose(1, 0, 2)
        labels_rgb_xz = np.repeat(labels_rgb_xz, self.interpolation_factor, axis=0)
        labels_rgb_xz = Image.fromarray((labels_rgb_xz * 255).astype(np.uint8))

        raw_img = np.max(load_2d_slices_at_time(images_path, t=t), axis=0)
        raw_rgb = Image.fromarray((raw_img * 255 / raw_img.max()).astype(np.uint8)).convert('RGB')

        raw_img_xz = np.max(load_2d_slices_at_time(images_path, t=t), axis=1)
        raw_img_xz = np.repeat(raw_img_xz, self.interpolation_factor, axis=0)
        raw_rgb_xz = Image.fromarray((raw_img_xz * 255 / raw_img_xz.max()).astype(np.uint8)).convert('RGB')

        merged_labels = Image.blend(labels_rgb, raw_rgb, alpha=0.5)
        merged_labels_xz = Image.blend(labels_rgb_xz, raw_rgb_xz, alpha=0.5)

        (self.results_folder / TRACK_RESULTS / MERGED_LABELS).mkdir(parents=True, exist_ok=True)
        (self.results_folder / TRACK_RESULTS / MERGED_LABELS_XZ).mkdir(parents=True, exist_ok=True)
        merged_labels.save(str(self.results_folder / TRACK_RESULTS / MERGED_LABELS / ("merged_labels_t%06d.png" % t)))
        merged_labels_xz.save(
            str(self.results_folder / TRACK_RESULTS / MERGED_LABELS_XZ / ("merged_labels_xz_t%06d.png" % t)))


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
        Image.fromarray(img2d).save(str(tracked_labels_path / ("track_results_t%06i_z%04i.tif" % (t, z))),
                                    compression="tiff_lzw")


def gaussian_interpolation_3d(label_image, interpolation_factor=10, smooth_sigma=5) -> \
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
        percentage = 1 - np.count_nonzero(sub_img) / sub_img.size
        img_smooth = skf.gaussian(np.repeat(sub_img, interpolation_factor, axis=2),
                                  sigma=smooth_sigma, mode='constant')
        threshold = np.percentile(img_smooth, percentage * 100)
        interpolated_bbox = (bbox[0], bbox[1],
                             slice(bbox[2].start * interpolation_factor, bbox[2].stop * interpolation_factor,
                                   bbox[2].step))
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


def plot_prgls_prediction(ref_ptrs: ndarray, tgt_ptrs: ndarray, predicted_ref_ptrs: ndarray, t1: int, t2: int,
                          fig_width_px=1200, dpi=96):
    """
    Draws the initial matching between two sets of 3D points and their matching relationships.

    Parameters
    ----------
    ref_ptrs : ndarray
        A 2D array of shape (n, 3) containing the positions of reference points.
    tgt_ptrs : ndarray
        A 2D array of shape (n, 3) containing the positions of target points.
    predicted_ref_ptrs : ndarray
        A 2D array of shape (n, 3) containing the predicted positions of reference points.
    t1 : int
        The time step of the reference points.
    t2 : int
        The time step of the target points.
    fig_width_px : int, optional
        The width of the output figure in pixels. Default is 1200.
    dpi : int, optional
        The resolution of the output figure in dots per inch. Default is 96.

    Raises
    ------
    AssertionError
        If the inputs have invalid shapes or data types.
    """
    # Validate the inputs
    assert isinstance(ref_ptrs, ndarray) and ref_ptrs.ndim == 2 and ref_ptrs.shape[
        1] == 3, "ref_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(tgt_ptrs, ndarray) and tgt_ptrs.ndim == 2 and tgt_ptrs.shape[
        1] == 3, "tgt_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(predicted_ref_ptrs, ndarray) and predicted_ref_ptrs.ndim == 2 and predicted_ref_ptrs.shape[
        1] == 3, "predicted_ref_ptrs should be a 2D array with shape (n, 3)"

    # Plot the scatters of the ref_points and tgt_points
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, fig_width_px, ref_ptrs, tgt_ptrs, t1, t2)

    # Plot the matching relationships between the two sets of points
    for ref_ptr, tgt_ptr in zip(ref_ptrs, predicted_ref_ptrs):
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptr[1], -ref_ptr[0]])
        pt2 = np.asarray([tgt_ptr[1], -tgt_ptr[0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="C1")
        ax2.add_artist(con)

    return fig


def plot_two_pointset_scatters(dpi: float, fig_width_px: float, ref_ptrs: ndarray, tgt_ptrs: ndarray, t1: int, t2: int):
    """
    Creates a figure with two subplots showing two sets of 3D points.

    Parameters
    ----------
    dpi : float
        The resolution of the output figure in dots per inch.
    fig_width_px : float
        The width of the output figure in pixels.
    ref_ptrs : ndarray
        A 2D array of shape (n, 3) containing the positions of reference points.
    tgt_ptrs : ndarray
        A 2D array of shape (n, 3) containing the positions of target points.
    t1 : int
        The time step of the reference points.
    t2 : int
        The time step of the target points.

    Returns
    -------
    ax1 : matplotlib.axes.Axes
        The first Axes object of the subplots.
    ax2 : matplotlib.axes.Axes
        The second Axes object of the subplots.
    fig : matplotlib.figure.Figure
        The Figure object containing the two subplots.
    """
    # Calculate the figure size based on the input width and dpi
    fig_width_in = fig_width_px / dpi  # convert to inches assuming the given dpi
    fig_height_in = fig_width_in / 1.618  # set height to golden ratio
    # Determine whether to use a top-down or left-right layout based on the aspect ratio of the point sets
    ref_range_y, ref_range_x, _ = np.max(ref_ptrs, axis=0) - np.min(ref_ptrs, axis=0)
    tgt_range_y, tgt_range_x, _ = np.max(tgt_ptrs, axis=0) - np.min(tgt_ptrs, axis=0)
    top_down = ref_range_x + tgt_range_x >= ref_range_y + tgt_range_y
    # Create the figure and subplots
    if top_down:
        # print("Using top-down layout")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width_in, fig_height_in))
    else:
        # print("Using left-right layout")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in))
    # Plot the point sets on the respective subplots
    ax1.scatter(ref_ptrs[:, 1], -ref_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 1')
    ax2.scatter(tgt_ptrs[:, 1], -tgt_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 2')

    unify_xy_lims(ax1, ax2)

    # Set plot titles or y-axis labels based on the layout
    if top_down:
        ax1.set_ylabel(f"Point Set t={t1}")
        ax2.set_ylabel(f"Point Set t={t2}")
    else:
        ax1.set_title(f"Point Set t={t1}")
        ax2.set_title(f"Point Set t={t2}")
    return ax1, ax2, fig


def unify_xy_lims(ax1, ax2):
    """
    Set the x and y limits of two matplotlib axes to be the same.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        The first Axes object.
    ax2 : matplotlib.axes.Axes
        The second Axes object.
    """
    # Determine the shared x_lim and y_lim
    x_lim = [min(ax1.get_xlim()[0], ax2.get_xlim()[0]), max(ax1.get_xlim()[1], ax2.get_xlim()[1])]
    y_lim = [min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1])]
    # Set the same x_lim and y_lim on both axes
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
