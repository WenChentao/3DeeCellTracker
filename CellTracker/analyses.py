"""
A module for analyses after tracking
Author: Chentao Wen

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tifffile import imread

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def get_activities(raw_path: str, tracked_labels_path: str, volume_num: int, layer_num: int):
    """
    Get activities of all cells

    Parameters
    ----------
    raw_path : str
        The path of the images for extracting activities
    tracked_labels_path : str
        The path of the tracked labels
    volume_num : int
        The number of the volumes
    layer_num : int
        the number of layers in the raw images and the tracked labels

    Returns
    -------
    activities : numpy.ndarray
        The extracted activities with shape (volume, label)
    """
    images_label, images_raw = _read_image(1, layer_num, raw_path, tracked_labels_path)
    cell_num = np.max(images_label)
    activities = np.zeros((volume_num, cell_num))
    discard_ratio = 0.1
    for frame in range(1, volume_num + 1):

        print("t=%i"%frame, end="\r")

        # read raw images and labels
        if frame>=2:
            images_label, images_raw = _read_image(frame, layer_num, raw_path, tracked_labels_path)

        # calculate mean intensities of each cell of top 90% area
        for label in range(1, cell_num + 1):
            intensity_label_i = images_raw[images_label==label]
            threshold = np.floor(np.size(intensity_label_i) * discard_ratio).astype(int)
            sorted_intensity_idx = np.argsort(intensity_label_i)
            activities[frame-1, label-1] = np.mean(intensity_label_i[sorted_intensity_idx[threshold:]])
    return activities


def _read_image(frame, layer_num, path_raw, path_tracked):
    """Read 3D images of raw activities and tracked labels"""
    images_raw = []
    images_label = []
    for z in range(1, layer_num + 1):
        images_raw.append(imread(path_raw % (frame, z)))
        images_label.append(imread(path_tracked % (frame, z)))
    images_raw = np.array(images_raw)
    images_label = np.array(images_label)
    return images_label, images_raw


def optimize_row_column(duration, n_signals, figsize):
    """
    Return the proper number of row, column for visualization

    Parameters
    ----------
    duration : int
        Number of time points for each signal.
    n_signals : int
        Number of signals.
    figsize : tuple
        size of the figure to show activities

    Returns
    -------
    row_n : int
        Number of row
    column_n : int
        Number of column

    Notes
    -----
    The row_n and colume_n are designed to make sure the xy_ratio of each subplot propotional with duration.

    Examples
    --------
    >>> optimize_row_column(duration=200, n_signals=100, figsize=(40, 20))
    (14, 8)
    """
    width_hight_ratio = figsize[0]/figsize[1]
    total_length = duration * n_signals
    row_n = int((total_length / (50 * width_hight_ratio)) ** 0.5)
    column_n = int(np.ceil(n_signals / row_n))
    return row_n, column_n


def draw_signals(signals, ylim_upper=None, ylim_lower=None, figsize=(20, 10)):
    """
    Draw signals in multiple subplots

    Parameters
    ----------
    signals : numpy.ndarray
        N Signals with T time points with shape (T, N)
    ylim_upper : float
        ylim upper bound. If None, set it to the highest value.
    ylim_lower : float
        ylim lower bound. If None, set it to the lowest value.
    figsize : tuple
        Size of the figure

    Returns
    -------
    fig : matplotlib.figure
    axes : array of matplotlib.axes.Axes
    """
    row_n, column_n = optimize_row_column(signals.shape[0], signals.shape[1], figsize)
    fig, axes = plt.subplots(row_n, column_n, figsize=figsize)
    for row in range(row_n):
        for column in range(column_n):
            n = row * column_n + column
            if n >= signals.shape[1]:
                break
            axes[row, column].plot(signals[:, n], lw=2)
            upper_sig_n, lower_sig_n = np.nanmax(signals[:, n]), np.nanmin(signals[:, n])
            if ylim_upper is not None:
                upper_sig_n = ylim_upper
            if ylim_lower is not None:
                lower_sig_n = ylim_lower
            axes[row, column].set_ylim(lower_sig_n, upper_sig_n)
            axes[row, column].set_title("N%d" % (n + 1), va="top")
            if row<row_n-1:
                axes[row, column].get_xaxis().set_visible(False)
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.2, hspace=0.2)

