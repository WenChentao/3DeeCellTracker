import csv
from typing import List, Tuple

import shutil
import h5py
import numpy as np
import plotly.graph_objects as go


def _read_hdf5(hdf5_path: str):
    view_data = dict()
    with h5py.File(hdf5_path, 'r') as file:
        view_data["affine_aligned_coords_t1"] = file["affine_aligned_coords_t1"][:]
        view_data["neuropal_coords_norm_t2"] = file["neuropal_coords_norm_t2"][:]
        view_data["match_seg_t1_seg_t2"] = file["match_seg_t1_seg_t2"][:]
        view_data["ids_neuropal"] = file["neuropal_coords_norm_t2"].attrs["ids_neuropal"].tolist()
        view_data["ids_wba"] = file["affine_aligned_coords_t1"].attrs["ids_wba"].tolist()
    return view_data


def correct_links(init_matching_file: str, modify_pairs: List[Tuple[any, any]]=None, new_results_name: str = None):
    view_data = _read_hdf5(init_matching_file)
    match_nx2 = view_data["match_seg_t1_seg_t2"].copy()
    ids_wba: list = view_data["ids_wba"]
    ids_neuropal: list = view_data["ids_neuropal"]

    print("Initial matching 3D:")
    plot_matching_with_arrows_3d_plotly(view_data["affine_aligned_coords_t1"],
                                view_data["neuropal_coords_norm_t2"],
                                match_nx2, view_data["ids_wba"], view_data["ids_neuropal"])

    for wba_id, neuropal_id in modify_pairs:
        if wba_id is None and neuropal_id is None:
            pass
        elif wba_id is None:
            try:
                idx = ids_neuropal.index(neuropal_id)
                match_nx2 = remove_row_by_neuopal_id(ids_wba, idx, match_nx2, neuropal_id)
            except ValueError:
                print(f"Warning: Neuron ID {neuropal_id} was not found in neuropal image!")
        elif neuropal_id is None:
            try:
                idx = ids_wba.index(wba_id)
                match_nx2 = remove_row_by_wba_id(ids_neuropal, idx, match_nx2, wba_id)
            except ValueError:
                print(f"Warning: Neuron ID {wba_id} was not found in wba image!")
        else:
            try:
                idx_neuropal = ids_neuropal.index(neuropal_id)
            except ValueError:
                print(f"Warning: Neuron ID {neuropal_id} was not found in neuropal image!")
                continue
            try:
                idx_wba = ids_wba.index(wba_id)
            except ValueError:
                print(f"Warning: Neuron ID {wba_id} was not found in wba image!")
                continue
            match_nx2 = remove_row_by_neuopal_id(ids_wba, idx_neuropal, match_nx2, neuropal_id, with_warning=False)
            match_nx2 = remove_row_by_wba_id(ids_neuropal, idx_wba, match_nx2, wba_id, with_warning=False)
            match_nx2 = np.vstack((match_nx2, (idx_wba, idx_neuropal)))
            print(f"Add Link: from {wba_id} to {neuropal_id}")

    print("Modified matching 3D:")
    plot_matching_with_arrows_3d_plotly(view_data["affine_aligned_coords_t1"],
                                view_data["neuropal_coords_norm_t2"],
                                match_nx2, view_data["ids_wba"], view_data["ids_neuropal"])

    if new_results_name is not None:
        shutil.copy(init_matching_file, new_results_name+".h5")
        with h5py.File(new_results_name+".h5", 'r+') as new_file:
            del new_file["match_seg_t1_seg_t2"]
            dataset_match = new_file.create_dataset("match_seg_t1_seg_t2", data=match_nx2)

        match = np.asarray([(ids_wba[i], ids_neuropal[j]) for i, j in match_nx2])
        sorted_match = match[match[:, 0].astype(int).argsort()]

        with open(new_results_name+".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write each row of the array to the CSV file
            for row in sorted_match:
                writer.writerow(row)


def remove_row_by_wba_id(ids_neuropal, idx, match_nx2, wba_id, with_warning=True):
    row = np.nonzero(match_nx2[:, 0] == idx)[0]
    if len(row) == 1:
        idx_neuropal = match_nx2[row[0], 1]
        match_nx2 = np.delete(match_nx2, row[0], axis=0)
        print(f"Delete Link: from {wba_id} to {ids_neuropal[idx_neuropal]}")
    else:
        if with_warning:
            print(f"Warning: Link from {wba_id} did not exist")
    return match_nx2


def remove_row_by_neuopal_id(ids_wba, idx, match_nx2, neuropal_id, with_warning=True):
    row = np.nonzero(match_nx2[:, 1] == idx)[0]
    if len(row) == 1:
        idx_wba = match_nx2[row[0], 0]
        match_nx2 = np.delete(match_nx2, row[0], axis=0)
        print(f"Delete Link: from {ids_wba[idx_wba]} to {neuropal_id}")
    else:
        if with_warning:
            print(f"Warning: Link to {neuropal_id} did not exist")
    return match_nx2


def plot_matching_with_arrows_3d_plotly(points_t1, points_t2, pairs, ids_t1, ids_t2):
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