import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from numpy import ndarray
from plotly import graph_objects as go

from CellTracker.utils import set_unique_xlim


def plot_initial_matching(ref_ptrs: ndarray, tgt_ptrs: ndarray, pairs_px2: ndarray, t1: int, t2: int, fig_width_px=1800,
                          dpi=96, ids_ref=None, ids_tgt=None, show_3d: bool = False):
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
    if show_3d:
        fig = plot_matching_with_arrows_3d_plotly(ref_ptrs, tgt_ptrs, pairs_px2, ids_ref, ids_tgt)
        return fig
    # Plot the scatters of the ref_points and tgt_points
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, fig_width_px, ref_ptrs, tgt_ptrs, t1, t2, ids_ref, ids_tgt)

    # Plot the matching relationships between the two sets of points
    for ref_index, tgt_index in pairs_px2:
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptrs[ref_index, 1], ref_ptrs[ref_index, 0]])
        pt2 = np.asarray([tgt_ptrs[tgt_index, 1], tgt_ptrs[tgt_index, 0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="C1")
        ax2.add_artist(con)
    # ax1.axis('equal')
    # ax2.axis('equal')
    set_unique_xlim(ax1, ax2)
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
                          dpi=96, ids_ref=None, ids_tgt=None):
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
    fig_height_in = fig_width_in / 1.618  # set height to golden ratio
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    plt.scatter(ref_ptrs[:, 1], -ref_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 1')
    plt.scatter(tgt_ptrs[:, 1], -tgt_ptrs[:, 0], marker="x", facecolors='r', edgecolors='r', label='Set 2')

    # Plot the matching relationships between the two sets of points
    for ref_index, tgt_index in pairs_px2:
        # Get the coordinates of the matched points in the two point sets
        pt1 = np.asarray([ref_ptrs[ref_index, 1], -ref_ptrs[ref_index, 0]])
        pt2 = np.asarray([tgt_ptrs[tgt_index, 1], -tgt_ptrs[tgt_index, 0]])

        # Draw a connection between the matched points in the two subplots using the `ConnectionPatch` class
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], lw=1, color="C1")
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
                             fig_width_px=1800, dpi=96):
    # Validate the inputs
    validate_inputs(ref_ptrs, tgt_ptrs, predicted_ref_ptrs)

    # Plot the scatters of the ref_points and tgt_points
    ax1, ax2, fig = plot_two_pointset_scatters(dpi, fig_width_px, ref_ptrs, tgt_ptrs, t1, t2)

    # Plot the matching relationships between the two sets of points
    plot_matching_relationships(ref_ptrs, predicted_ref_ptrs, ax1, ax2, single_panel=False)
    set_unique_xlim(ax1, ax2)
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


def plot_two_pointset_scatters(dpi: float, fig_width_px: float, ref_ptrs: ndarray, tgt_ptrs: ndarray, t1: int, t2: int,
                               ids_ref: list = None, ids_tgt: list = None):
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
    fig_width_in = fig_width_px / dpi  # convert to inches assuming the given dpi
    fig_height_in = fig_width_in / 1.618  # set height to golden ratio
    # Determine whether to use a top-down or left-right layout based on the aspect ratio of the point sets
    ref_range_y, ref_range_x, _ = np.max(ref_ptrs, axis=0) - np.min(ref_ptrs, axis=0)
    tgt_range_y, tgt_range_x, _ = np.max(tgt_ptrs, axis=0) - np.min(tgt_ptrs, axis=0)
    top_down = ref_range_x + tgt_range_x >= ref_range_y + tgt_range_y

    ids_ref = range(1, ref_ptrs.shape[0]+1) if ids_ref is None else ids_ref
    ids_tgt = range(1, tgt_ptrs.shape[0]+1) if ids_tgt is None else ids_tgt

    # Create the figure and subplots
    if top_down:
        # print("Using top-down layout")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width_in, fig_height_in))
    else:
        # print("Using left-right layout")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in))

    # Plot the point sets on the respective subplots
    ax1.scatter(ref_ptrs[:, 1], ref_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 1')
    for i, txt in enumerate(ids_ref):
        ax1.annotate(txt, (ref_ptrs[i, 1], ref_ptrs[i, 0]))

    ax2.scatter(tgt_ptrs[:, 1], tgt_ptrs[:, 0], facecolors='b', edgecolors='b', label='Set 2')
    for i, txt in enumerate(ids_tgt):
        ax2.annotate(txt, (tgt_ptrs[i, 1], tgt_ptrs[i, 0]))

    ax1.invert_yaxis()
    ax2.invert_yaxis()

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


def validate_inputs(ref_ptrs: ndarray, tgt_ptrs: ndarray, predicted_ref_ptrs: ndarray):
    assert isinstance(ref_ptrs, ndarray) and ref_ptrs.ndim == 2 and ref_ptrs.shape[1] == 3, \
        "ref_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(tgt_ptrs, ndarray) and tgt_ptrs.ndim == 2 and tgt_ptrs.shape[1] == 3, \
        "tgt_ptrs should be a 2D array with shape (n, 3)"
    assert isinstance(predicted_ref_ptrs, ndarray) and predicted_ref_ptrs.ndim == 2 and predicted_ref_ptrs.shape[
        1] == 3, \
        "predicted_ref_ptrs should be a 2D array with shape (n, 3)"
