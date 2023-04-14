from glob import glob
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from numpy import ndarray
import skimage.filters as skf
from skimage.segmentation import relabel_sequential
import skimage.measure as skm
import scipy.ndimage.measurements as ndm
from tifffile import imread

from CellTracker.watershed import recalculate_cell_boundaries


class CoordsToImageTransformer:

    proofed_segmentation: ndarray
    z_slice_original_labels: slice
    use_8_bit: bool

    def __init__(self, track_results_folder: str, voxel_size: tuple):
        self.voxel_size = np.asarray(voxel_size)
        self.track_results_folder = Path(track_results_folder) / "tracked_labels"
        self.track_results_folder.mkdir(parents=True, exist_ok=True)

    def load_segmentation(self, files_path: str):
        """
        Load the proofed segmentation from a directory containing image slices.
        """
        # Get list of paths to image slices
        slice_paths = sorted(glob(files_path))
        if len(slice_paths) == 0:
            # Raise an error if no image slices are found in the specified directory
            raise FileNotFoundError(f"No image in {files_path} was found")

        # Load the proofed segmentation and relabel it to sequential integers
        proofed_segmentation = imread(slice_paths).transpose((1,2,0))
        self.proofed_segmentation, _, _ = relabel_sequential(proofed_segmentation)

        # Print a message confirming that the proofed segmentation has been loaded and its shape
        print(f"Loaded the proofed segmentations at vol 1 with {np.count_nonzero(np.unique(proofed_segmentation))} cells")

    def interpolate(self, interpolation_factor: int, smooth_sigma: float = 2.5):
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

        if interpolation_factor <= 0:
            raise ValueError("Interpolation factor must be greater than zero.")

        print("Interpolating images along z-axis...")

        # Interpolate layers in z-axis
        self.interpolation_factor = interpolation_factor
        self.subregions = gaussian_interpolation_3d(
            self.proofed_segmentation, interpolation_factor=interpolation_factor, smooth_sigma=smooth_sigma)

        # Move cells and get interpolated labels
        interpolated_labels, cell_overlaps_mask = self.move_cells(movements_nx3=None)

        # Calculate the original z-slice labels
        self.z_slice_original_labels = slice(interpolation_factor // 2, interpolated_labels.shape[2], interpolation_factor)

        # Recalculate cell boundaries
        self.auto_corrected_segmentation = recalculate_cell_boundaries(
            interpolated_labels[:, :, self.z_slice_original_labels],
            cell_overlaps_mask[:, :, self.z_slice_original_labels],
            sampling_xy=self.voxel_size[:2])

        # Fix segmentation errors
        self.auto_corrected_segmentation, was_corrected = fix_labeling_errors(self.auto_corrected_segmentation)
        if was_corrected:
            # Recalculate the interpolation and cell boundaries again if segmentation errors were fixed
            self.subregions = gaussian_interpolation_3d(
                self.auto_corrected_segmentation, interpolation_factor=interpolation_factor, smooth_sigma=smooth_sigma)
            interpolated_labels, cell_overlaps_mask = self.move_cells(movements_nx3=None)
            self.auto_corrected_segmentation = recalculate_cell_boundaries(
                interpolated_labels[:, :, self.z_slice_original_labels],
                cell_overlaps_mask[:, :, self.z_slice_original_labels],
                sampling_xy=self.voxel_size[:2])

        # Check if 8-bit is needed for saving the image
        self.use_8_bit = self.auto_corrected_segmentation.max() <= 255
        print(
            f"The interpolated segmentations at vol 1 contains {np.count_nonzero(np.unique(self.auto_corrected_segmentation))} cells")

        # Save labels in the first volume (interpolated)
        filename = str(self.track_results_folder / "track_results_t%04i_z%04i.tif")
        save_tracked_cell_images(self.auto_corrected_segmentation, filename, t=1, use_8_bit=self.use_8_bit)

        print("Calculating coordinates of cell centers...")
        # Calculate coordinates of cell centers at t=1
        coord_vol1 = ndm.center_of_mass(
            self.auto_corrected_segmentation > 0,
            self.auto_corrected_segmentation,
            range(1, self.auto_corrected_segmentation.max() + 1)
        )
        self.coord_vol1 = Coordinates(np.array(coord_vol1), interpolation_factor, self.voxel_size, dtype="raw")

    def move_cells_in_3d_image(self, movements_nx3: ndarray = None, cells_missed: List[int] = None):
        interpolated_labels, cell_overlaps_mask = self.move_cells(movements_nx3=movements_nx3, cells_missed=cells_missed)
        return recalculate_cell_boundaries(
            interpolated_labels[:, :, self.z_slice_original_labels],
            cell_overlaps_mask[:, :, self.z_slice_original_labels],
            sampling_xy=self.voxel_size[:2])

    def move_cells(self, movements_nx3: ndarray = None, cells_missed: List[int] = None):
        """
        Generate a image with labels indicating the moved cells.

        Parameters
        ----------
        movements_nx3 : numpy.array
            Movements of each cell

        Returns
        -------
        output : numpy.ndarray
            The new image with moved cells
        mask : numpy.ndarray
            The new image with the
        """
        if movements_nx3 is None:
            movements_nx3 = np.zeros((len(self.subregions), 3))
        else:
            assert movements_nx3.shape[0] == len(self.subregions)
        if cells_missed is None:
            cells_missed = []
        output_img = np.repeat(np.zeros_like(self.proofed_segmentation), self.interpolation_factor, axis=2)
        mask = output_img.copy()
        for i, (bbox, subimage) in enumerate(self.subregions):
            label = i + 1
            print(f"Generating... cell image:{label}", end="\r")
            if label in cells_missed:
                continue
            bbox_moved = self.add_slices_with_change(bbox, movements_nx3[i])
            output_img[bbox_moved] += (subimage * label).astype(output_img.dtype)
            mask[bbox_moved] += (subimage * 1).astype(mask.dtype)
        return output_img, mask

    @staticmethod
    def add_slices_with_change(slices, changes):
        if len(slices) != 3 or len(changes) != 3:
            raise ValueError("Slices and changes must be 3-element tuples")

        new_slices = []
        for s, c in zip(slices, changes):
            new_start = s.start + int(c) if s.start is not None else None
            new_stop = s.stop + int(c) if s.stop is not None else None
            new_slices.append(slice(new_start, new_stop, s.step))
        return tuple(new_slices)


def gaussian_interpolation_3d(label_image, interpolation_factor=10, smooth_sigma=5) -> \
        List[Tuple[Tuple[slice, slice, slice], ndarray]]:
    """
    Generate smoothed label image of cells

    Parameters
    ----------
    label_image : numpy.ndarray
        Label image
    interpolation_factor : int
        Factor of interpolations along z axis, should be < 10
    smooth_sigma : float
        sigma used for making Gaussian blur

    Returns
    -------
    output_img : numpy.ndarray
        Generated smoothed label image
    mask : numpy.ndarray
        Mask image indicating the overlapping of multiple cells (0: background; 1: one cell; >1: multiple cells)
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
                             slice(bbox[2].start * interpolation_factor, bbox[2].stop * interpolation_factor, bbox[2].step))
        subregions.append((interpolated_bbox, img_smooth > threshold))
    return subregions


def fix_labeling_errors(segmentation: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Relabel the separate cells that were incorrectly labeled as the same one

    Parameters
    ----------
    segmentation :
        the input label image with separated cells that were labeled as the same one

    Returns
    -------
    new_segmentation :
        the corrected label image
    """
    num_cells = np.size(np.unique(segmentation)) - 1
    new_segmentation = skm.label(segmentation, connectivity=3)
    was_corrected = num_cells != np.max(new_segmentation)
    if was_corrected:
        print(f"WARNING: The number of cells in the manually labeled segmentation ({num_cells}) does not match "
              f"the number of separated cells found by the program ({np.max(new_segmentation)}). "
              f"The program has corrected the segmentation accordingly.")
    return new_segmentation, was_corrected


class Coordinates:
    def __init__(self, coords: np.ndarray, interpolation_factor: int, voxel_size: ndarray, dtype: str = "raw"):
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
        return Coordinates(self._raw + other._raw, self.interpolation_factor, self.voxel_size, 'raw')

    def __sub__(self, other: 'Coordinates') -> 'Coordinates':
        return Coordinates(self._raw - other._raw, self.interpolation_factor, self.voxel_size, 'raw')

    @staticmethod
    def _transform_z(coords_nx3: np.ndarray, factor_x3: ndarray) -> np.ndarray:
        """Transform the z-coordinate by scaling with factor"""
        new_coords = coords_nx3.copy()
        new_coords = new_coords * factor_x3[None, :]
        return new_coords

    @property
    def real(self) -> np.ndarray:
        return self._transform_z(self._raw, self.voxel_size)

    @property
    def interp(self) -> np.ndarray:
        return np.round(
            self._transform_z(
                self._raw, np.asarray((1, 1, self.interpolation_factor))
            )
        ).astype(np.int32)

    @property
    def raw(self):
        return np.round(self._raw).astype(np.int32)

    @property
    def cell_num(self) -> int:
        return self._raw.shape[0]


def save_tracked_cell_images(labels_xyz: ndarray, folder_path: str, t: int, use_8_bit: bool=True):
    """
    Save a 3D image at time t as 2D image sequence

    Parameters
    ----------
    labels_xyz : numpy.array
        The 3D image to be saved
    folder_path : str
        The path of the image files to be saved.
        It should use formatted string to indicate volume number and then layer number, e.g. "xxx_t%04d_z%04i.tif"
    t : int
        The volume number for the image to be saved
    use_8_bit: bool
        The array will be transformed to 8-bit or 16-bit before saving as image.
    """
    dtype = np.uint8 if use_8_bit else np.uint16
    for z in range(labels_xyz.shape[2]):
        img2d = (labels_xyz[:, :, z]).astype(dtype)
        Image.fromarray(img2d).save(folder_path % (t, z + 1))
