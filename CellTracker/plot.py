from __future__ import division, absolute_import, print_function, unicode_literals, annotations

from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from numpy import ndarray


def custom_tab20_cmap(map_index_to_tab20: List[int]):
    n = len(map_index_to_tab20)
    x = np.linspace(0, 1, 20)
    x = np.concatenate((x[0:20:2], x[1:20:2]))
    tab20_colors = plt.cm.tab20b(x)
    custom_colors = [tab20_colors[map_index_to_tab20[i]] for i in range(n)]
    return custom_colors


def plot_initial_matching(ref_ptrs: ndarray, tgt_ptrs: ndarray, pairs_px2: ndarray,
                          t1_name: int | str, t2_name: int | str, ref_ptrs_confirmed: ndarray=None,
                          fig_height_px=1500, dpi=96, ids_ref=None, ids_tgt=None, show_3d: bool = False, display_fig=True, top_down=True,
                          show_ids=False):
    """Draws the initial matching between two sets of 3D points and their matching relationships.

    Args:
        ref_ptrs (ndarray): A 2D array of shape (n, 3) containing the reference points.
        tgt_ptrs (ndarray): A 2D array of shape (n, 3) containing the target points.
        pairs_px2 (ndarray): A 2D array of shape (m, 2) containing the pairs of matched points.
        fig_height_px (int): The width of the output figure in pixels. Default is 1200.
        fig_size (tuple): Ignore fig_width_px if this is not None
        dpi (int): The resolution of the output figure in dots per inch. Default is 96.

    Raises:
        AssertionError: If the inputs have invalid shapes or data types.
    """
    if show_3d:
        fig = plot_matching_with_arrows_3d_plotly(ref_ptrs, tgt_ptrs, pairs_px2, ids_ref, ids_tgt)
        return fig

    # Plot the scatters of the ref_points and tgt_points
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, ref_ptrs, tgt_ptrs, t1_name, t2_name, ids_ref, ids_tgt, fig_height_px=fig_height_px,
                                               show_ids=show_ids)
    equal_layout(ax1, ax2)

    # Plot the matching relationships between the two sets of points
    for ref_index, tgt_index in pairs_px2:
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptrs[ref_index, 1], ref_ptrs[ref_index, 0]])
        pt2 = np.asarray([tgt_ptrs[tgt_index, 1], tgt_ptrs[tgt_index, 0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="C1")
        ax2.add_artist(con)
    if ref_ptrs_confirmed is not None:
        ax1.scatter(ref_ptrs_confirmed[:, 1], ref_ptrs_confirmed[:, 0], facecolors='r', edgecolors='r', label='Confirmed cells')

    if display_fig:
        plt.pause(0.1)
    return fig


def plot_pairs_and_movements(ref_ptrs: ndarray, tgt_ptrs: ndarray, t1: int, t2: int, ref_ptrs_confirmed: ndarray,
                             ref_ptrs_tracked: ndarray, fig_height_px=1500, dpi=96, ids_ref=None, ids_tgt=None, display_fig=True,
                             show_ids=True):
    # Plot the scatters of the ref_points and tgt_points
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, ref_ptrs, tgt_ptrs, t1, t2, ids_ref, ids_tgt, fig_height_px=fig_height_px,
                                               show_ids=show_ids)

    # Plot the predicted pairs and movements
    n = ref_ptrs_confirmed.shape[0]
    for i in range(n):
        # Get the coordinates of the matched points in the two point sets
        pt1 = ref_ptrs_confirmed[i, [1, 0]]
        pt2 = ref_ptrs_tracked[i, [1, 0]]

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="C1")
        ax2.add_artist(con)
    if ref_ptrs_confirmed is not None:
        ax1.scatter(ref_ptrs_confirmed[:, 1], ref_ptrs_confirmed[:, 0], facecolors='r', edgecolors='r', label='Confirmed cells')
    if display_fig:
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


def plot_matching_with_arrows_3d_plotly(points_t1, points_t2, pairs, ids_t1, ids_t2):
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

    # draw motion paths
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
        # Add arrow (cone) at the end of each line
        cone_size = 0.01  # adjust this parameter to fit your plot
        cone = go.Cone(
            x=[-points_t2[end, 0]], y=[-points_t2[end, 1]], z=[points_t2[end, 2]],
            u=[-(-points_t1[start, 0] - (-points_t2[end, 0]))],
            v=[-(-points_t1[start, 1] - (-points_t2[end, 1]))],
            w=[-(points_t1[start, 2] - points_t2[end, 2])],
            sizemode='absolute',
            sizeref=cone_size,
            anchor='tip',
            showscale=False,
            colorscale=[[0, 'black'], [1, 'black']]
        )
        traces.append(cone)

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
                          dpi=96):
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

    # Calculate best figure height
    x_min = min(np.min(ref_ptrs[:, 1]), np.min(tgt_ptrs[:, 1]))
    x_max = max(np.max(ref_ptrs[:, 1]), np.max(tgt_ptrs[:, 1]))
    y_min = min(np.min(ref_ptrs[:, 0]), np.min(tgt_ptrs[:, 0]))
    y_max = max(np.max(ref_ptrs[:, 0]), np.max(tgt_ptrs[:, 0]))
    x_range = x_max - x_min
    y_range = y_max - y_min
    actual_aspect_ratio = x_range / y_range
    fig_height_in = fig_width_in / actual_aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    plt.scatter(ref_ptrs[:, 1], ref_ptrs[:, 0], facecolors='b', edgecolors='b', label=str(t1))
    plt.scatter(tgt_ptrs[:, 1], tgt_ptrs[:, 0], marker="x", facecolors='r', edgecolors='r', label=str(t2))
    ax.invert_yaxis()

    # Plot the matching relationships between the two sets of points
    for ref_index, tgt_index in pairs_px2:
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptrs[ref_index, 1], ref_ptrs[ref_index, 0]])
        pt2 = np.asarray([tgt_ptrs[tgt_index, 1], tgt_ptrs[tgt_index, 0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], lw=1, color="C1")
        plt.legend()
    plt.axis('equal')
    return fig


def plot_matching_2d_with_plotly(neuropal_ptrs: np.ndarray, wba_ptrs: np.ndarray, pairs_px2: np.ndarray,
                                 ids_ref=None, ids_tgt=None, shift=(0, 0, 0)):
    """
    Draws the initial matching between two sets of 3D points and their matching relationships using Plotly.

    Args:
        neuropal_ptrs (np.ndarray): A 2D array of shape (n, 3) containing the reference points.
        wba_ptrs (np.ndarray): A 2D array of shape (n, 3) containing the target points.
        pairs_px2 (np.ndarray): A 2D array of shape (m, 2) containing the pairs of matched points.
    """
    from plotly import graph_objects as go

    # Validate the inputs
    assert isinstance(neuropal_ptrs, np.ndarray) and neuropal_ptrs.ndim == 2 and neuropal_ptrs.shape[1] == 3, \
        "ref_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(wba_ptrs, np.ndarray) and wba_ptrs.ndim == 2 and wba_ptrs.shape[1] == 3, \
        "tgt_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(pairs_px2, np.ndarray) and pairs_px2.ndim == 2 and pairs_px2.shape[1] == 2, \
        "pairs_px2 should be a 2D array with shape (n, 2)"

    neuropal_ptrs = neuropal_ptrs + np.asarray(shift)[np.newaxis, :]

    # Create the figure
    fig = go.Figure()

    # Add scatter plots for the reference and target points
    fig.add_trace(go.Scatter(x=neuropal_ptrs[:, 1], y=-neuropal_ptrs[:, 0],
                             mode='markers', name='NeuroPAL Points',
                             text=ids_ref, marker=dict(size=10, color='blue')))
    fig.add_trace(go.Scatter(x=wba_ptrs[:, 1], y=-wba_ptrs[:, 0],
                             mode='markers', name='WBA Points',
                             text=ids_tgt, marker=dict(size=10, color='red')))

    # Add lines for the matching relationships
    for ref_index, tgt_index in pairs_px2:
        fig.add_shape(type="line",
                      x0=neuropal_ptrs[ref_index, 1], y0=-neuropal_ptrs[ref_index, 0],
                      x1=wba_ptrs[tgt_index, 1], y1=-wba_ptrs[tgt_index, 0],
                      line=dict(color="green", width=2))

    # Update the layout
    fig.update_layout(xaxis_title="X", yaxis_title="Y",
                      autosize=True, hovermode="closest")

    return fig


def plot_matching_relationships(ref_ptrs, predicted_ref_ptrs, ax1=None, ax2=None, single_panel=False):
    for ref_ptr, tgt_ptr in zip(ref_ptrs, predicted_ref_ptrs):
        pt1 = np.asarray([ref_ptr[1], ref_ptr[0]])
        pt2 = np.asarray([tgt_ptr[1], tgt_ptr[0]])

        if single_panel:
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], lw=1, color="C1")
        else:
            con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                                  axesA=ax2, axesB=ax1, color="C1")
            ax2.add_artist(con)


def plot_predicted_movements(ref_ptrs: ndarray, tgt_ptrs: ndarray, predicted_ref_ptrs: ndarray, t1: int, t2: int,
                             fig_height_px=1500, dpi=96):
    # Validate the inputs
    validate_inputs(ref_ptrs, tgt_ptrs, predicted_ref_ptrs)

    # Plot the scatters of the ref_points and tgt_points
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, ref_ptrs, tgt_ptrs, t1, t2, fig_height_px=fig_height_px)

    # Plot the matching relationships between the two sets of points
    plot_matching_relationships(ref_ptrs, predicted_ref_ptrs, ax1, ax2, single_panel=False)
    equal_layout(ax1, ax2)
    plt.pause(0.1)
    return fig, (ax1, ax2)


def plot_predicted_movements_one_panel(ref_ptrs: ndarray, tgt_ptrs: ndarray, predicted_ref_ptrs: ndarray, t1: int,
                                       t2: int,
                                       fig_width_px=1800, dpi=96):
    # Validate the inputs
    validate_inputs(ref_ptrs, tgt_ptrs, predicted_ref_ptrs)

    # Plot the scatters of the ref_points and tgt_points
    fig_width_in = fig_width_px / dpi  # convert to inches assuming the given dpi
    fig_height_in = fig_width_in / 1.618  # set height to golden ratio
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    plt.scatter(ref_ptrs[:, 1], -ref_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 1')
    plt.scatter(tgt_ptrs[:, 1], -tgt_ptrs[:, 0], marker="x", facecolors='r', edgecolors='r', label='Set 2')

    # Plot the matching relationships between the two sets of points
    plot_matching_relationships(ref_ptrs, predicted_ref_ptrs, single_panel=True)
    plt.axis('equal')
    return fig


def plot_two_pointset_scatters(dpi: float, ref_ptrs: ndarray, tgt_ptrs: ndarray,
                               t1_name: int | str, t2_name: int | str,
                               ids_ref: list = None, ids_tgt: list = None,
                               fig_height_px: float = 1500, fig_width_px: float = 2000, show_ids=True):
    """
    Creates a figure with two subplots showing two sets of 3D points.

    Parameters
    ----------
    dpi : float
        The resolution of the output figure in dots per inch.
    fig_height_px : float
        The height limit of the output figure in pixels.
    fig_width_px: float
        The width limit of the output figure in pixels.
    ref_ptrs : ndarray
        A 2D array of shape (n, 3) containing the positions of reference points.
    tgt_ptrs : ndarray
        A 2D array of shape (n, 3) containing the positions of target points.
    t1_name : int
        The time step of the reference points.
    t2_name : int
        The time step of the target points.
    ids_ref : list, optional
        A list of strings containing the IDs of the reference points. Default is None.
    ids_tgt : list, optional
        A list of strings containing the IDs of the target points. Default is None.

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
    fig_height_in = fig_height_px / dpi  # convert to inches assuming the given dpi

    # Determine whether to use a top-down or left-right layout based on the aspect ratio of the point sets
    ptrs_combined = np.vstack((ref_ptrs, tgt_ptrs))
    range_y, range_x, _ = np.max(ptrs_combined, axis=0) - np.min(ptrs_combined, axis=0)
    ratio = range_x / range_y

    fig_width_in = fig_height_in * ratio / 2
    if fig_width_in > fig_width_px / dpi:
        fig_width_in = fig_width_px / dpi
        fig_height_in = fig_width_in * 2 / ratio

    ids_ref = range(1, ref_ptrs.shape[0]+1) if ids_ref is None else ids_ref
    ids_tgt = range(1, tgt_ptrs.shape[0]+1) if ids_tgt is None else ids_tgt

    fig_size = (fig_width_in, fig_height_in)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size)

    # Plot the point sets on the respective subplots
    ax1.scatter(ref_ptrs[:, 1], ref_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 1')
    ax1.scatter(tgt_ptrs[:, 1], tgt_ptrs[:, 0], alpha=0)
    ax1.invert_yaxis()
    ax2.scatter(tgt_ptrs[:, 1], tgt_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 2')
    ax2.invert_yaxis()
    if show_ids:
        for i, txt in enumerate(ids_ref):
            ax1.annotate(txt, (ref_ptrs[i, 1], ref_ptrs[i, 0]))
        for i, txt in enumerate(ids_tgt):
            ax2.annotate(txt, (tgt_ptrs[i, 1], tgt_ptrs[i, 0]))

    equal_layout(ax1, ax2)

    plt.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.02)

    # Set plot titles or y-axis labels based on the layout
    ax1.set_ylabel(f"Point Set t={t1_name}")
    ax2.set_ylabel(f"Point Set t={t2_name}")
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
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    # Determine the shared x_lim and y_lim
    x_lim = [min(ax1.get_xlim()[0], ax2.get_xlim()[0]), max(ax1.get_xlim()[1], ax2.get_xlim()[1])]
    y_lim = [min(ax1.get_ylim()[1], ax2.get_ylim()[1]), max(ax1.get_ylim()[0], ax2.get_ylim()[0])]
    range_x = x_lim[1] - x_lim[0]
    range_y = y_lim[1] - y_lim[0]
    if range_y > range_x:
        x_mean = (x_lim[1] + x_lim[0]) / 2
        x_lim = [x_mean - range_y / 2, x_mean + range_y / 2]
    else:
        y_mean = (y_lim[1] + y_lim[0]) / 2
        y_lim = [y_mean - range_x / 2, y_mean + range_x / 2]

    # Set the same x_lim and y_lim on both axes
    ax1.set_xlim(x_lim[0], x_lim[1])
    ax1.set_ylim(y_lim[1], y_lim[0])
    ax2.set_xlim(x_lim[0], x_lim[1])
    ax2.set_ylim(y_lim[1], y_lim[0])


def equal_layout(ax1, ax2):
    """
    Set the x and y scale of two matplotlib axes to be the same.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        The first Axes object.
    ax2 : matplotlib.axes.Axes
        The second Axes object.
    """
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    # ax1.axis("equal")
    # ax2.axis("equal")
    # ratio = (ax1.get_ylim()[0] - ax1.get_ylim()[1]) / (ax1.get_xlim()[1] - ax1.get_xlim()[0])
    #
    # # Determine the shared x_lim and y_lim
    # x_lim = [min(ax1.get_xlim()[0], ax2.get_xlim()[0]), max(ax1.get_xlim()[1], ax2.get_xlim()[1])]
    # y_lim = [min(ax1.get_ylim()[1], ax2.get_ylim()[1]), max(ax1.get_ylim()[0], ax2.get_ylim()[0])]
    # range_x = x_lim[1] - x_lim[0]
    # range_y = y_lim[1] - y_lim[0]
    # if range_y > range_x * ratio:
    #     x_mean = (x_lim[1] + x_lim[0]) / 2
    #     x_lim = [x_mean - range_y / ratio / 2, x_mean + range_y / ratio / 2]
    # else:
    #     y_mean = (y_lim[1] + y_lim[0]) / 2
    #     y_lim = [y_mean - range_x * ratio / 2, y_mean + range_x * ratio / 2]
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    # Set the same x_lim and y_lim on both axes
    ax2.set_xlim(x_lim[0], x_lim[1])
    ax2.set_ylim(y_lim[0], y_lim[1])


def validate_inputs(ref_ptrs: ndarray, tgt_ptrs: ndarray, predicted_ref_ptrs: ndarray):
    assert isinstance(ref_ptrs, ndarray) and ref_ptrs.ndim == 2 and ref_ptrs.shape[1] == 3, \
        "ref_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(tgt_ptrs, ndarray) and tgt_ptrs.ndim == 2 and tgt_ptrs.shape[1] == 3, \
        "tgt_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(predicted_ref_ptrs, ndarray) and predicted_ref_ptrs.ndim == 2 and predicted_ref_ptrs.shape[
        1] == 3, \
        "predicted_ref_ptrs should be a 2D array with shape (n, 3)"


def set_unique_xlim(ax1, ax2):
    x1_min, x1_max = ax1.get_xlim()
    x2_min, x2_max = ax2.get_xlim()
    ax1.set_xlim(min((x1_min, x2_min)), max((x1_max, x2_max)))
    ax2.set_xlim(min((x1_min, x2_min)), max((x1_max, x2_max)))
