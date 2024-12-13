from __future__ import annotations

import gc
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import napari
import numpy as np
from magicgui import widgets, use_app
from magicgui.types import FileDialogMode
from magicgui.widgets import Container

from CellTracker.stardist3dcustom import StarDist3DCustom
from CellTracker.stardistwrapper import create_cmap
from CellTracker.trackerlite import TrackerLite


def set_layer_position(viewer, layer, to_top=False):
    """Moves a specified layer to the top or bottom within the viewer layers."""
    if to_top:
        viewer.layers.move(viewer.layers.index(layer), len(viewer.layers) - 1)
    else:
        viewer.layers.move(viewer.layers.index(layer), 0)


def get_h5_file_path(caption="Open tracking results (.h5)", filter='*.h5'):
    start_path = get_last_directory()
    start_path = Path.home() if start_path is None else start_path
    h5_tracking_path = use_app().get_obj("show_file_dialog")(
        FileDialogMode.EXISTING_FILE,
        caption=caption,
        filter=filter,
        start_path=str(start_path),
    )
    save_last_directory(Path(h5_tracking_path))
    return h5_tracking_path


def save_last_directory(path: Path):
    with open('.last_directory.txt', 'w') as file:
        file.write(str(path))

def get_last_directory() -> Path | None:
    last_directory_file = Path('.last_directory.txt')
    if last_directory_file.exists():
        with last_directory_file.open('r') as file:
            last_directory = file.read().strip()
            return Path(last_directory)
    else:
        return None


class ToggleLayers:
    def __init__(self, layers_groups: list[list]):
        # 将所有图层组存储在一个列表中，便于迭代与索引访问
        self.layer_groups = layers_groups
        self.state = 0  # 表示当前可见的图层组索引
        self._set_visibility()

    def _set_visibility(self):
        """根据当前 state 设置对应图层的可见性。"""
        # 先全部设置为不可见
        for group in self.layer_groups:
            for layer in group:
                layer.visible = False

        # 再把当前 state 的图层组设为可见
        for layer in self.layer_groups[self.state]:
            layer.visible = True

    def toggle(self):
        """切换至下一个图层组的可见性。"""
        self.state = (self.state + 1) % len(self.layer_groups)
        self._set_visibility()


class ToggleProjections:
    MIP_VIEW = True
    STACK_VIEW = False
    def __init__(self, view: SegmentationView, raw_img, neuropal_img, prob_img, labels_img, labels_mip):
        self.state = self.STACK_VIEW
        self.raw_img = raw_img
        self.raw_mip = raw_img.max(axis=0)
        self.prob_img = prob_img
        self.prob_mip = prob_img.max(axis=0)
        self.labels_img = labels_img
        self.labels_mip = labels_mip
        self.neuropal_img = neuropal_img
        if neuropal_img is not None:
            self.neuropal_mip = neuropal_img.max(axis=0)
        self.z = 0
        self.view = view

    def update_labels(self, labels_img, labels_mip):
        self.labels_img = labels_img
        self.labels_mip = labels_mip

    def toggle(self):
        """Toggle visibility between the two groups of layers."""
        # Set visibility based on the current state
        # Switch state
        self.state = not self.state

        if self.state == self.STACK_VIEW:
            self.view.wba_image_layer.data = self.raw_img
            self.view.probmap_layer.data = self.prob_img
            self.view.seg_layer.data = self.labels_img
            if self.neuropal_img is not None:
                self.view.neuropal_image_layer.data = self.neuropal_img
            self.view.viewer.dims.set_current_step(0, self.z)
        else:
            self.z = self.view.viewer.dims.current_step[0]
            self.view.wba_image_layer.data = self.raw_mip
            self.view.probmap_layer.data = self.prob_mip
            self.view.seg_layer.data = self.labels_mip
            if self.neuropal_img is not None:
                self.view.neuropal_image_layer.data = self.neuropal_mip


class SegmentationView:
    RFP_LOWER, RFP_UPPER = 0, 1
    TITLE = 'Segmentation Fixer'

    def __init__(self, inspector: SegmentationInspector):
        self.viewer = napari.Viewer()
        self.viewer.title = self.TITLE
        self.ins = inspector
        self.message = widgets.TextEdit(value="", label="Message")

        container_import = Container(widgets=[self.message,])
        self.viewer.window.add_dock_widget(container_import, name="Import data", area="right")

    def show_message(self, msg: str):
        self.message.value = msg + f"Current number of cells: {self.ins.cell_edges_mip.max()}"

    def update_labels_layers(self, seg_labels_zyx):
        cmap = self.modify_label_cmap(seg_labels_zyx.transpose(1, 2, 0))
        if 'Labels (t0)' in self.viewer.layers:
            self.seg_layer.data = seg_labels_zyx
            self.seg_layer.color = cmap
        else:
            self.seg_layer = self.viewer.add_labels(seg_labels_zyx, name='Labels (t0)', opacity=1, color=cmap)

    def modify_label_cmap(self, seg_labels_yxz):
        new_cmap = create_cmap(seg_labels_yxz, is_transparent=True)
        color_dict = {i: new_cmap(i) for i in range(new_cmap.N)}
        return color_dict

    def add_wba_nuclei_layers(self, img_wba_t0_zyx):
        img_mip = np.max(img_wba_t0_zyx, axis=0)
        contrast = (np.percentile(img_mip, 50), np.percentile(img_mip, 99.5))
        self.wba_image_layer = self.viewer.add_image(img_wba_t0_zyx, name='Nuclei image (wba)',
                                                     colormap='gray', contrast_limits=contrast)
        set_layer_position(self.viewer, self.wba_image_layer)

    def add_neuropal_nuclei_layers(self, img_neuropal_zyx):
        if img_neuropal_zyx is not None:
            img_mip = np.max(img_neuropal_zyx, axis=0)
            contrast = (np.percentile(img_mip, 50), np.percentile(img_mip, 99.5))
            self.neuropal_image_layer = self.viewer.add_image(img_neuropal_zyx, name='Nuclei image (neuropal)',
                                                     colormap='gray', contrast_limits=contrast)
            set_layer_position(self.viewer, self.neuropal_image_layer)

    def update_probmap_layers(self, probmap_zyx):
        if 'Probmap (t0)' in self.viewer.layers:
            self.probmap_layer.data = probmap_zyx
        else:
            self.probmap_layer = self.viewer.add_image(probmap_zyx, name='Probmap (t0)', colormap='gray')
            set_layer_position(self.viewer, self.probmap_layer)


def sobel_1d(image, axis):
    from scipy.ndimage import convolve1d
    kernel = np.array([-1, 0, 1])
    return convolve1d(image, kernel, axis=axis)


def extract_edges(image_zyx: np.ndarray):
    dx = sobel_1d(image_zyx, axis=-1)  # x 方向梯度
    dy = sobel_1d(image_zyx, axis=-2)  # y 方向梯度
    edges = np.abs(dx) + np.abs(dy)
    cell_edges = np.where(edges > 0, image_zyx, 0)  # 非边缘置为 0
    return cell_edges

def extract_edges_mip(mask_zyx: np.ndarray):
    labels = np.unique(mask_zyx)
    labels = labels[labels > 0]  # 排除背景标签（假设背景为 0）

    # 创建一个空的 2D 图像用于保存结果
    y, x = mask_zyx.shape[1], mask_zyx.shape[2]
    result_2d = np.zeros((y, x), dtype=mask_zyx.dtype)

    # 遍历每个物体
    for label in labels:
        # 提取当前物体的掩膜
        object_mask = np.where(mask_zyx == label, mask_zyx, 0)

        # 对物体进行最大投影
        projection = np.max(object_mask, axis=0)  # 投影到 XY 平面

        # 提取边缘
        dx = sobel_1d(projection, axis=1)  # x 方向梯度
        dy = sobel_1d(projection, axis=0)  # y 方向梯度
        edges = np.abs(dx) + np.abs(dy)
        cell_edges = np.where(edges > 0, projection, 0)  # 非边缘置为 0

        # 将边缘添加到结果图像
        result_2d = np.maximum(result_2d, cell_edges)
    return result_2d


class SegmentationResult:
    def __init__(self, h5_file_path: str, stardist: "StarDist3DCustom"):
        self.stardist = stardist
        self.h5_file_path = h5_file_path
        self.h5_file = h5py.File(h5_file_path, "r+")

        # fixed data
        self.prob_map = self.h5_file["prob_map_zyx"][:]
        self.dist_map_reduced_zyxr = self.h5_file["dist_map_reduced_zyxr"]
        self.img_shape = self.h5_file["img_shape"][:]

        # can be modified
        self.seg = self.h5_file["seg_labels_zyx"][:]
        self.prob_n = self.h5_file["prob_n"][:]
        self.dist_n = self.h5_file["dist_n"][:]
        self.coords_nx3 = self.h5_file["coords_nx3"][:]

    def write_to_h5(self):
        # Write only the modified data
        del self.h5_file["seg_labels_zyx"]
        self.h5_file.create_dataset("seg_labels_zyx", data=self.seg)

        del self.h5_file["prob_n"]
        self.h5_file.create_dataset("prob_n", data=self.prob_n)

        del self.h5_file["dist_n"]
        self.h5_file.create_dataset("dist_n", data=self.dist_n)

        del self.h5_file["coords_nx3"]
        self.h5_file.create_dataset("coords_nx3", data=self.coords_nx3)

    def _re_segment(self, coords_nx3, dist_n, prob_n):
        self.seg, res_dict = self.stardist._instances_from_prediction_simple(
            self.img_shape,
            prob_n,
            dist_n,
            points=coords_nx3,
            nms_thresh=None
        )
        self.prob_n = res_dict["prob"]
        self.dist_n = res_dict["dist"]
        self.coords_nx3 = res_dict["points"]
        assert self.seg.max() == self.prob_n.shape[0]

    def delete_cell(self, cell: int):
        prob_n = np.delete(self.prob_n, cell - 1, axis=0)
        dist_n = np.delete(self.dist_n, cell - 1, axis=0)
        coords_nx3 = np.delete(self.coords_nx3, cell - 1, axis=0)
        self._re_segment(coords_nx3, dist_n, prob_n)
        self.write_to_h5()

    def add_a_cell(self, coordinate: np.ndarray):
        prob_n = np.append(self.prob_n, 1)
        scaled_point = (coordinate / np.asarray(self.stardist.config.grid)).astype(np.int16)
        dist_n = np.vstack([self.dist_n, self.dist_map_reduced_zyxr[tuple(scaled_point)]])
        coords_nx3 = np.vstack([self.coords_nx3, np.asarray(coordinate).reshape((1, 3))])
        self._re_segment(coords_nx3, dist_n, prob_n)
        self.write_to_h5()


class SegmentationInspector:
    def __init__(self, tracker: "TrackerLite", neuropal_loader_type: str):
        self.view = SegmentationView(self)
        self.selected_cell = None
        self.selected_zyx = None
        self.tracker = tracker
        self.neuropal_loader = NeuropalImageLoaderFactory.get_loader(neuropal_loader_type)
        napari.run()

    def load_segmentation_results(self, tracking_result_path=None, neuropal_image_path=None):
        self.model = TrackingResultsLoader(tracking_result_path)
        self.neuropal_nuclei_zyx = self.neuropal_loader.load_nuclei(neuropal_image_path)
        if self.model.h5_seg_path:
            self.segmentation_result = SegmentationResult(
                self.model.h5_seg_path, self.tracker.seg.stardist_model)
            self.update_segmentation()
            self.view.show_message("")

            @self.view.viewer.window.qt_viewer.destroyed.connect
            def cleanup():
                self.segmentation_result.h5_file.close()
                gc.collect()
                print("HDF5 file closed")

    def update_segmentation(self):
        self.cell_edges_zyx = extract_edges(self.segmentation_result.seg)
        self.cell_edges_mip = extract_edges_mip(self.segmentation_result.seg)
        self.view.update_labels_layers(self.cell_edges_zyx)
        self.view.update_probmap_layers(self.segmentation_result.prob_map)

    def load_raw_t0(self, path=None):
        self.img_wba_t0_zyx = self.model.read_nuclei_image_t0(
            self.model.raw_dset, self.model.T_Initial,
            self.model.raw_channel_nuclei, path)
        self.view.add_wba_nuclei_layers(self.img_wba_t0_zyx)
        self.view.add_neuropal_nuclei_layers(self.neuropal_nuclei_zyx)
        if self.neuropal_nuclei_zyx is not None:
            self.toggle_raw_prob: ToggleLayers = ToggleLayers([
                [self.view.wba_image_layer, self.view.probmap_layer, self.view.seg_layer],
                [self.view.neuropal_image_layer],
            ])
        self.toggle_stack_mip: ToggleProjections = ToggleProjections(
            self.view,
            self.img_wba_t0_zyx,
            self.neuropal_nuclei_zyx,
            self.segmentation_result.prob_map,
            self.cell_edges_zyx,
            self.cell_edges_mip
        )
        self.key_bindings()

    def key_bindings(self):
        @self.view.viewer.bind_key('w')
        def toggle_prob(viewer):
            if self.neuropal_nuclei_zyx is not None:
                self.toggle_raw_prob.toggle()

        @self.view.viewer.bind_key('q')
        def toggle_projection(viewer):
            self.toggle_stack_mip.toggle()

        @self.view.viewer.mouse_double_click_callbacks.append
        def select_cell(viewer, event):
            z, y, x = int(event.position[0]), int(event.position[1]), int(event.position[2])
            self.selected_zyx = (z, y, x,)
            cell = self.segmentation_result.seg[z, y, x]
            if cell == 0:
                self.selected_cell = None
            else:
                self.selected_cell = cell

        @self.view.viewer.bind_key("d")
        def delete_a_cell(viewer):
            if self.selected_cell is not None:
                self.segmentation_result.delete_cell(self.selected_cell)
                self.update_segmentation()
                self.toggle_stack_mip.update_labels(self.cell_edges_zyx,self.cell_edges_mip)
                self.view.show_message("Deleted a cell\n")
            else:
                self.view.show_message("Cannot delete because no cell was selected\n")

        @self.view.viewer.bind_key("a")
        def add_a_cell(viewer):
            if self.selected_zyx is not None:
                if self.selected_cell is None:
                    self.segmentation_result.add_a_cell(self.selected_zyx)
                    self.update_segmentation()
                    self.toggle_stack_mip.update_labels(self.cell_edges_zyx, self.cell_edges_mip)
                    self.view.show_message("Added a cell\n")
                else:
                    self.view.show_message("There is already a cell at the selected coordinates, "
                                           "Do you want to add one anyway?")
            else:
                self.view.show_message("Cannot add because no coordinate was selected\n")


class TrackingResultsLoader:
    def __init__(self, path=None):
        if path is None:
            self.h5_seg_path = get_h5_file_path(caption="Open segmentation results at t0 (.h5)")
        else:
            self.h5_seg_path = path
        with h5py.File(self.h5_seg_path, 'r') as segmentation_h5_file:
            self.T_Initial = segmentation_h5_file.attrs["t_initial"]
            self.raw_dset = segmentation_h5_file.attrs["raw_dset"]
            self.raw_channel_nuclei = segmentation_h5_file.attrs["raw_channel_nuclei"]

    def read_nuclei_image_t0(self, raw_dset: str, t_initial: int, raw_channel_nuclei: int, path=None):
        if path is None:
            self.h5_raw_path = use_app().get_obj("show_file_dialog")(
                FileDialogMode.EXISTING_FILE,
                caption="Open raw images (.h5)",
                filter='*.h5'
            )
        else:
            self.h5_raw_path = path
        with h5py.File(self.h5_raw_path, 'r') as f:
            return f[raw_dset][t_initial - 1, :, raw_channel_nuclei, :, :]


class NeuroPALImageLoader(ABC):
    """Abstract base class for image loaders"""

    @abstractmethod
    def load_nuclei(self, file_path: str) -> np.ndarray | None:
        """Load nuclei image and convert to standard 3D NumPy array"""
        pass


class NWBLoader(NeuroPALImageLoader):
    """Loader for NWB file"""
    def load_nuclei(self, file_path: str) -> np.ndarray:
        # Open image from file path
        if isinstance(file_path, str):
            with h5py.File(file_path, "r") as file:
                channel_RFP = file["acquisition/NeuroPALImageRaw/RGBW_channels"][3]
                neuropal_img = file["acquisition/NeuroPALImageRaw/data"]
                neuropal_rfp_img_zyx = neuropal_img[..., channel_RFP].transpose((2,0,1))
                print(f"{neuropal_rfp_img_zyx.shape=}")
        else:
            raise TypeError("Input must be file path")

        # Convert to NumPy array
        return np.asarray(neuropal_rfp_img_zyx)


class NoneImageLoader(NeuroPALImageLoader):
    def load_nuclei(self, file_path: str) -> None:
        return None


class NeuropalImageLoaderFactory:
    """Factory for creating image loaders"""
    _loaders = {
        'nwb': NWBLoader,
        'none': NoneImageLoader
    }

    @classmethod
    def get_loader(cls, loader_type: str) -> NeuroPALImageLoader:
        """Get specific image loader"""
        loader_class = cls._loaders.get(loader_type.lower())
        if not loader_class:
            raise ValueError(f"Unsupported loader type: {loader_type}")
        return loader_class()

