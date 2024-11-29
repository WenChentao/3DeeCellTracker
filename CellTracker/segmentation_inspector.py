from __future__ import annotations

from pathlib import Path

import h5py
import napari
import numpy as np
from magicgui import widgets, use_app
from magicgui.types import FileDialogMode
from magicgui.widgets import Container
from scipy import ndimage

from CellTracker.stardistwrapper import create_cmap


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
    def __init__(self, layers1: list, layers2: list):
        self.state = False
        self.layers1 = layers1
        self.layers2 = layers2
        self.toggle()

    def toggle(self):
        """Toggle visibility between the two groups of layers."""
        # Switch state
        self.state = not self.state
        # Set visibility based on the current state
        for layer in self.layers1:
            layer.visible = self.state
        for layer in self.layers2:
            layer.visible = not self.state


class ToggleProjections:
    MIP_VIEW = True
    STACK_VIEW = False
    def __init__(self, view: SegmentationView, raw_img, prob_img, labels_img, labels_mip):
        self.state = self.STACK_VIEW
        self.raw_img = raw_img
        self.raw_mip = raw_img.max(axis=0)
        self.prob_img = prob_img
        self.prob_mip = prob_img.max(axis=0)
        self.labels_img = labels_img
        self.labels_mip = labels_mip
        self.view = view

    def toggle(self):
        """Toggle visibility between the two groups of layers."""
        # Set visibility based on the current state
        # Switch state
        self.state = not self.state

        if self.state == self.STACK_VIEW:
            self.view.image_layer.data = self.raw_img
            self.view.probmap_layer.data = self.prob_img
            self.view.seg_layer.data = self.labels_img
        else:
            self.view.image_layer.data = self.raw_mip
            self.view.probmap_layer.data = self.prob_mip
            self.view.seg_layer.data = self.labels_mip


class SegmentationView:
    RFP_LOWER, RFP_UPPER = 0, 1
    TITLE = 'Segmentation Fixer'

    def __init__(self, inspector: SegmentationInspector):
        self.viewer = napari.Viewer()
        self.viewer.title = self.TITLE
        self.ins = inspector
        # Buttons to load data from h5 file
        self.btn_load_label_h5 = widgets.PushButton(text="Load segmentation results (h5)")
        self.btn_load_raw_h5 = widgets.PushButton(text="Load raw image (h5)")

        container_import = Container(widgets=[self.btn_load_label_h5, self.btn_load_raw_h5])
        self.viewer.window.add_dock_widget(container_import, name="Import data", area="right")

        # Buttons for loading raw images and tracking results
        self.btn_load_raw_h5.clicked.connect(self.ins.load_raw_t0)
        self.btn_load_label_h5.clicked.connect(self.ins.load_segmentation_results)

    def add_labels_layers(self, seg_labels_zyx):
        self.viewer.layers.clear()
        cmap = self.modify_label_cmap(seg_labels_zyx.transpose(1, 2, 0))
        self.seg_layer = self.viewer.add_labels(seg_labels_zyx, name='Labels (t0)', opacity=1, color=cmap)

    def modify_label_cmap(self, seg_labels_yxz):
        new_cmap = create_cmap(seg_labels_yxz, is_transparent=True)
        color_dict = {i: new_cmap(i) for i in range(new_cmap.N)}
        return color_dict

    def add_image_layers(self, img_wba_t0_zyx):
        img_mip = np.max(img_wba_t0_zyx, axis=0)
        contrast = (np.percentile(img_mip, 50), np.percentile(img_mip, 99.5))
        self.image_layer = self.viewer.add_image(img_wba_t0_zyx, name='Raw image (t0)',
                                            colormap='gray', contrast_limits=contrast)
        set_layer_position(self.viewer, self.image_layer)

    def add_probmap_layers(self, probmap_zyx):
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


class SegmentationInspector:
    def __init__(self):
        self.view = SegmentationView(self)
        napari.run()

    def load_segmentation_results(self, path=None):
        self.model = Model(path)
        if self.model.h5_seg_path:
            self.T_Initial = self.model.segmentation_results.attrs["t_initial"]
            self.raw_dset = self.model.segmentation_results.attrs["raw_dset"]
            self.raw_channel_nuclei = self.model.segmentation_results.attrs["raw_channel_nuclei"]
            self.set_segmentation(seg_h5=self.model.segmentation_results)

            @self.view.viewer.window.qt_viewer.destroyed.connect
            def cleanup():
                self.model.segmentation_results.close()
                print("HDF5 file closed")

    def set_segmentation(self, seg_h5):
        self.segmentation_results = seg_h5["seg_labels_zyx"][:]
        self.cell_edges_zyx = extract_edges(self.segmentation_results)
        self.cell_edges_mip = extract_edges_mip(self.segmentation_results)
        self.view.add_labels_layers(self.cell_edges_zyx)
        self.prob_map = seg_h5["prob_map_zyx"][:]
        self.view.add_probmap_layers(self.prob_map)

    def load_raw_t0(self, path=None):
        self.img_wba_t0_zyx = self.model.read_nuclei_image_t0(
            self.raw_dset, self.T_Initial, self.raw_channel_nuclei, path)
        self.view.add_image_layers(self.img_wba_t0_zyx)
        self.toggle_raw_prob: ToggleLayers = ToggleLayers([self.view.image_layer], [self.view.probmap_layer])
        self.toggle_stack_mip: ToggleProjections = ToggleProjections(
            self.view,
            self.img_wba_t0_zyx,
            self.prob_map,
            self.cell_edges_zyx,
            self.cell_edges_mip
        )
        self.key_bindings()

    def key_bindings(self):
        @self.view.viewer.bind_key('w')
        def toggle_prob(viewer):
            self.toggle_raw_prob.toggle()

        @self.view.viewer.bind_key('q')
        def toggle_projection(viewer):
            self.toggle_stack_mip.toggle()


class Model:
    def __init__(self, path=None):
        if path is None:
            self.h5_seg_path = get_h5_file_path(caption="Open segmentation results at t0 (.h5)")
        else:
            self.h5_seg_path = path
        self.segmentation_results = h5py.File(self.h5_seg_path, 'r')

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


def main():
    # Create a Napari viewer
    viewer = SegmentationInspector()


if __name__ == '__main__':
    main()
