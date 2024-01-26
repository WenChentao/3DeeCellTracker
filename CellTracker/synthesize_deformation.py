from functools import partial
from typing import Tuple, Optional, List, Union, Generator, Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from skimage.transform import estimate_transform
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, NearestNeighbors

NUM_FEATURES = 4

WIDTH_DEFORM = 0.3

MV_FACTOR = 0.2

RATIO_SEG_ERROR = 0.15

RANDOM_MOVEMENTS_FACTOR = 0.001

NUM_POINT_SET = 100
K_NEIGHBORS = 20  # number of neighbors for calculating relative coordinates
NUMBER_FEATURES = K_NEIGHBORS * 4 + 8


class DataGeneratorFPN:
    degrees_rotate_raw: Tuple[int] = (-135, -90, -45, 0, 45, 90, 135, 180)

    def __init__(self, points_normalized_nx3: ndarray, num_rotations: int, num_deformations: int,
                 range_rotation_tgt: float, movement_factor: float = 0.2):
        """
        This class is used to generate training data for FPN model.

        Parameters
        ----------
        points_normalized_nx3: ndarray
            The normalized points set
        num_rotations: int
            The number of rotations
        num_deformations: int
            The number of deformations
        range_rotation_tgt: tuple
            The range of rotation degrees for target points set
        movement_factor: float
            The movement factor for deformations
        """
        self.num_rotate = num_rotations
        self.num_deform = num_deformations
        self.points_normalized_nx3 = points_normalized_nx3
        self.train_data_gen = self.generator_train_data(points_normalized_nx3,
                                                        range_rotation_tgt=range_rotation_tgt,
                                                        movement_factor=movement_factor)

    @staticmethod
    def generator_train_data(points_nx3: ndarray, range_rotation_tgt: float,
                             batch_size: int = 128, movement_factor: float = 0.2) -> Generator[Tuple[ndarray, ndarray], None, None]:
        """Generate training data for FPN model

        Parameters
        ----------
        points_nx3: ndarray, shape (n, 3)
            The normalized points set
        batch_size: int
            The batch size
        range_rotation_tgt: float
            The range of rotation degrees for target points set
        movement_factor: float
            The movement factor for deformations

        Yields
        ------
        xy_bxkxfx2: ndarray, shape (batch_size, num_neighbors + 2, num_features, 2)
            The features of reference points and target points in each batch
        y_bx1: ndarray, shape (batch_siz, 1)
            The label of the matching of reference points and target points in each batch
        """
        n = points_nx3.shape[0]
        num_batch_per_set = n * 2
        num_sets = 20
        num_batch = num_batch_per_set * num_sets
        assert num_batch > batch_size, "The batch size is too large"

        x_bxkp2xfx2 = np.empty((num_batch, K_NEIGHBORS + 2, NUM_FEATURES, 2), dtype=np.float32)
        y_bx1 = np.empty((num_batch, 1), dtype=np.bool_)

        random_indexes = np.arange(num_batch)

        # Generate training data
        while True:
            # Generate a relative large number of data than the batch size
            updated_points_nx3 = points_nx3.copy()
            for i in range(num_sets):
                rotvec_ref = DataGeneratorFPN.random_rotvec((-np.pi, np.pi))
                rotvec_tgt = DataGeneratorFPN.random_rotvec((-np.pi*range_rotation_tgt/180, np.pi*range_rotation_tgt/180))
                longest_distance = get_longest_distance(updated_points_nx3)
                points_ref_nx3, points_tgt_nx3 = generate_points_pair(updated_points_nx3, rotvec_ref, rotvec_tgt,
                                                                      longest_distance, mv_factor=movement_factor)
                points_tgt_with_errors, replaced_indexes = add_seg_errors(points_tgt_nx3)
                point_set_s_ = slice(i * num_batch_per_set, (i + 1) * num_batch_per_set)
                x_bxkp2xfx2[point_set_s_, ...], y_bx1[point_set_s_, :] = \
                    points_to_features(points_ref_nx3, points_tgt_with_errors, replaced_indexes)
                updated_points_nx3 = points_tgt_with_errors.copy()

            # Yield small batches from the generated data set in a shuffled order
            np.random.shuffle(random_indexes)
            for i in range(num_batch // batch_size):
                batch_i_s_ = np.s_[random_indexes[i * batch_size:(i + 1) * batch_size],:]
                yield x_bxkp2xfx2[batch_i_s_], y_bx1[batch_i_s_]

    @staticmethod
    def random_rotvec(vec_alpha: tuple):
        phi = np.random.uniform(0, np.pi)  # 与z轴的角度
        theta = np.random.uniform(0, 2 * np.pi)  # x-y平面上与x轴的角度
        alpha = np.random.uniform(vec_alpha[0], vec_alpha[1])  # 旋转角度

        # 先创建基于phi和theta的单位向量
        unit_vector = np.array([np.sin(phi) * np.cos(theta),
                                np.sin(phi) * np.sin(theta),
                                np.cos(phi)])

        # 然后将这个向量的每个分量乘以alpha
        rotation_ref = unit_vector * alpha
        return rotation_ref


# def train_data_rotation(points_normalized_nx3: ndarray, num_ptr_set: int = NUM_POINT_SET) -> ndarray:
#     degrees_mx3 = np.zeros((num_ptr_set, 3))
#     degrees_mx3[:, 2] = np.hstack((np.linspace(0, 40, num_ptr_set//2),
#                                    np.linspace(-40, -10, num_ptr_set - num_ptr_set//2)))
#     return rotate(points_normalized_nx3, degrees_mx3, num_ptr_set)


def train_data_deformation(points_normalized_mxnx3: ndarray, longest_distance: float, num_ptr_set: int = NUM_POINT_SET) \
        -> ndarray:
    new_points_mxmxnx3 = np.empty(((num_ptr_set), *points_normalized_mxnx3.shape))
    for i, points_nx3 in enumerate(points_normalized_mxnx3):
        new_points_mxmxnx3[i, ...] = random_deform(points_nx3, longest_distance, num_ptr_set)
    new_points_kxnx3 = new_points_mxmxnx3.reshape((-1, points_normalized_mxnx3.shape[1], 3))
    return new_points_kxnx3


def points_to_features(points_raw_nx3: ndarray, points_gen_nx3: ndarray,
                       replaced_indexes: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Convert the raw points and generated points to input data required by FPN model

    Parameters
    ----------
    points_raw_nx3: ndarray, shape (n, 3)
        The raw points
    points_gen_nx3: ndarray, shape (n, 3)
        The generated points
    replaced_indexes: ndarray, shape (n, )
        The indexes of replaced points in the generated points set
    k_ptrs: int
        The number of nearest neighbors for each point

    Returns
    -------
    ndarray, shape (2n, k_ptrs + 2, 4, 2)
        The spherical features of reference points and target points
    labels_2nx1: ndarray, shape (2n, 1)
        The labels indicating whether the points in the reference points set and target points set are matched
    """
    k_ptrs = K_NEIGHBORS
    # fit knn models for raw and generated points
    knn_model_raw = NearestNeighbors(n_neighbors=k_ptrs + 1).fit(points_raw_nx3)
    knn_model_generated = NearestNeighbors(n_neighbors=k_ptrs + 1).fit(points_gen_nx3)

    # shuffle the generated points
    shuffled_points_gen_nx3 = shuffle_points(points_gen_nx3)

    # generate spherical features for raw points, generated points and shuffled generated points
    points_raw_features_nxkp2x4 = spherical_features_of_points(points_raw_nx3, points_raw_nx3, k_ptrs, knn_model_raw)
    points_gen_features_nxkp2x4 = spherical_features_of_points(points_gen_nx3, points_gen_nx3, k_ptrs, knn_model_generated)
    points_gen_no_match_features_nxkp2x4 = spherical_features_of_points(points_gen_nx3, shuffled_points_gen_nx3, k_ptrs, knn_model_generated)

    # generate two point sets of reference points and target points with matched and unmatched points
    points_ref_2nxkp2x4x1 = np.vstack((points_raw_features_nxkp2x4, points_raw_features_nxkp2x4))[:, :, :, np.newaxis]
    points_tgt_2nxkp2x4x1 = np.vstack((points_gen_features_nxkp2x4, points_gen_no_match_features_nxkp2x4))[:, :, :, np.newaxis]

    # generate labels indicating whether the points in the reference points set and target points set are matched
    n = points_raw_nx3.shape[0]
    labels_2nx1 = np.ones((n * 2, 1), dtype=np.bool_)
    labels_2nx1[:n][replaced_indexes] = False
    labels_2nx1[n:] = False

    # randomly swap the reference points set and target points set to increase the diversity of the training data
    if np.random.rand() > 0.5:
        return np.concatenate((points_ref_2nxkp2x4x1, points_tgt_2nxkp2x4x1), axis=3), labels_2nx1
    else:
        return np.concatenate((points_tgt_2nxkp2x4x1, points_ref_2nxkp2x4x1), axis=3), labels_2nx1


def shuffle_points(points_nx3: ndarray) -> ndarray:
    """
    Generate a new points set by shuffling the points_nx3 so that each point in the new set is
    a different point in the original set

    Parameters
    ----------
    points_nx3: ndarray
        The points set

    Returns
    -------
    shuffled_points_nx3: ndarray
        The shuffled points set
    """
    n = points_nx3.shape[0]
    random_indexes = np.arange(n)
    np.random.shuffle(random_indexes)
    shuffled_points_nx3 = np.zeros_like(points_nx3)
    for i in range(n):
        if random_indexes[i] == i:
            no_match_index = random_indexes[i - 1]
        else:
            no_match_index = random_indexes[i]
        shuffled_points_nx3[i, :] = points_nx3[no_match_index, :]
    return shuffled_points_nx3


def add_seg_errors(points_normalized_nx3: ndarray, ratio: float = RATIO_SEG_ERROR, bandwidth: float = 0.1) -> Tuple[ndarray, ndarray]:
    """
    Add segmentation errors to points set by replacing some points with new points sampled from KDE model

    Parameters
    ----------
    points_normalized_nx3: ndarray
        The normalized points set
    ratio: float
        The ratio of points to be replaced by new points
    bandwidth: float
        The bandwidth of KDE model for sampling new points

    Returns
    -------
    new_points_nx3: ndarray
        The new points set with segmentation errors
    """
    if ratio <= 0 or ratio >= 1:
        raise ValueError(f"ratio should be set between 0 and 1 but = {ratio}")

    # Randomly select points to be replaced
    num_points = points_normalized_nx3.shape[0]
    num_replaced_points = int(np.ceil(num_points * ratio))
    points_indexes = np.arange(num_points)
    np.random.shuffle(points_indexes)
    replaced_indexes = points_indexes[:num_replaced_points]

    # Sample new points from KDE model
    kde_model = KernelDensity(bandwidth=bandwidth)
    kde_model.fit(points_normalized_nx3)
    new_points_nx3 = points_normalized_nx3.copy()
    new_points_nx3[replaced_indexes, :] = kde_model.sample(num_replaced_points)

    return new_points_nx3, replaced_indexes


def generate_points_pair(points_nx3: ndarray, rotvec_ref_3: ndarray, rotvec_tgt_3: ndarray,
                         longest_distance: Optional[float] = None,
                         mv_factor: float = None, width: float = None):
    """
    Generate a pair of points set with rotation and deformation

    Parameters
    ----------
    points_nx3: ndarray
        The normalized points set
    rotvec_ref_3: ndarray
        degree of rotation for reference points set
    rotvec_tgt_3: ndarray
        degree of rotation for target points set
    longest_distance: float
        The longest distance of points set
    mv_factor: float
        The strength of deformations
    width: float
        The spatial range of deformations

    Returns
    -------
    points_ref_nx3: ndarray
        The reference points set
    points_tgt_nx3: ndarray
        The target points set
    """
    if longest_distance is None:
        longest_distance = get_longest_distance(points_nx3)
    points_ref_nx3 = rotate_one_point_set(points_nx3, rotvec_ref_3)
    points_tgt_nx3 = rotate_one_point_set(points_ref_nx3, rotvec_tgt_3)
    points_ref_nx3 = random_deform(points_ref_nx3, longest_distance, num_ptr_set=1, mv_factor=mv_factor, width=width)[0, :, :]
    points_tgt_nx3 = random_deform(points_tgt_nx3, longest_distance, num_ptr_set=1, mv_factor=mv_factor, width=width)[0, :, :]
    points_tgt_nx3 += (np.random.rand(*points_tgt_nx3.shape) - 0.5) * 4 * RANDOM_MOVEMENTS_FACTOR
    return points_ref_nx3, points_tgt_nx3


# def rotate(points_nx3: ndarray, rad_mx3: ndarray, num_ptr_set: int) -> ndarray:
#     new_points_mxnx3 = np.empty((num_ptr_set, points_nx3.shape[0], 3))
#     for i in range(num_ptr_set):
#         r = Rotation.from_rotvec(rad_mx3[i, :])
#         new_points_mxnx3[i, ...] = np.dot(points_nx3, r.as_matrix())
#     return new_points_mxnx3


def rotate_one_point_set(points_nx3: ndarray, rad_3: ndarray) -> ndarray:
    r = Rotation.from_rotvec(rad_3)
    return np.dot(points_nx3, r.as_matrix())


def random_deform(points_normalized_nx3: ndarray, longest_distance: float, num_ptr_set: int, mv_factor: float = None,
                  width: float = None, appy_to_z: bool = True) -> ndarray:
    """Make random deformation to a point set

    Parameters
    ----------
    points_normalized_nx3 :
        The initial point set normalized by the normalize_pts function
    longest_distance :
        The longest distance in the initial point set
    num_ptr_set :
        The number of point sets to be generated
    mv_factor :
        affect the max movements along each axis: mv_factor * longest distance
    width :
        affect the range of deformation around the target point
    appy_to_z :
        If False, deformation will only be applied in x-y plane

    Notes
    -----
    A randomly selected point is applied a large movement, while other points were applied smaller movements according
    to the distances to the point.
    m: number of point sets to be generated
    n: number of points in the initial point set
    """
    if mv_factor is None:
        mv_factor = MV_FACTOR
    if width is None:
        width = WIDTH_DEFORM
    target_points_m = np.random.randint(points_normalized_nx3.shape[0], size=num_ptr_set)
    target_movement_mx1x3 = (longest_distance * mv_factor) * np.random.uniform(-1, 1, size=(num_ptr_set, 1, 3))
    if not appy_to_z:
        target_movement_mx1x3[..., 2] = 0
    return apply_deform(points_normalized_nx3, target_points_m, target_movement_mx1x3, longest_distance, width)


def apply_deform(points_normalized_nx3, target_points_m, target_movement_mx1x3, longest_distance, width):
    distances_to_targets_mxn = cdist(points_normalized_nx3[target_points_m, :], points_normalized_nx3,
                                     metric='euclidean')
    scale_factors_mxnx1 = np.exp(-0.5 * (distances_to_targets_mxn / (width * longest_distance)) ** 2)[:, :, np.newaxis]
    all_movements_mxnx3 = scale_factors_mxnx1 * target_movement_mx1x3
    new_points_mxnx3 = points_normalized_nx3[np.newaxis, :, :] + all_movements_mxnx3
    return new_points_mxnx3


def apply_deform_0_max(points_normalized_nx3: ndarray, movements: dict) -> ndarray:
    target_node = movements["node"]
    sigma = movements["sigma"]
    distances_to_targets_n = cdist(points_normalized_nx3[target_node:target_node+1, :], points_normalized_nx3, metric='euclidean')[0]
    scale_factors_nx1 = np.exp(-0.5 * (distances_to_targets_n / sigma) ** 2)[:, np.newaxis]
    factor_min = scale_factors_nx1.min()
    scale_factors_nx1 = (scale_factors_nx1 - factor_min) / (1 - factor_min)
    move_x, move_y, move_z, _ = polar2cartesian(node_with_vector=movements)
    all_movements_nx3 = scale_factors_nx1 * np.asarray([[move_x, move_y, move_z]])
    return points_normalized_nx3 + all_movements_nx3

def get_longest_distance(points_normalized_nx3: ndarray) -> float:
    """
    Get the longest distance between points in convex hull of points set

    Parameters
    ----------
    points_normalized_nx3: ndarray of shape (n, 3),
        normalized points set

    Returns
    -------
    float,
        the longest distance between points in convex hull of points set
    """
    convex_points: ndarray = get_convex_points(points_normalized_nx3)
    convex_distances: ndarray = cdist(convex_points, convex_points, metric='euclidean')
    return np.max(convex_distances)


def spherical_features_of_points(points_train_knn_nx3: ndarray, points_test_nx3: ndarray, k_neighbors: int, knn_model) -> ndarray:
    """
    Generate spherical features of points based on k neighbors and convex hull of points set

    Parameters
    ----------
    points_train_knn_nx3: ndarray
        The point set used to train the knn model
    points_test_nx3: ndarray
        The point set in which each point will be assigned neighbor points using the trained knn model
    k_neighbors: int
        The number of neighbors for each point in points_test_nx3
    knn_model: sklearn.neighbors.NearestNeighbors
        The trained knn model

    Returns
    -------
    ndarray, shape (n, k_neighbors+2, 4)
        The spherical features of points in points_test_nx3
    """
    # get convex hull of points_train_knn_nx3
    convex_points_mx3 = get_convex_points(points_train_knn_nx3)
    convex_distances_nxm = cdist(points_test_nx3, convex_points_mx3, metric='euclidean')
    convex_distances_mxm = cdist(convex_points_mx3, convex_points_mx3, metric='euclidean')
    longest_distance_index_n = np.argmax(convex_distances_nxm, axis=1)
    longest_distance_n = np.max(convex_distances_nxm, axis=1)
    longest_distance = np.max(convex_distances_mxm)

    # get spherical coordinates of neighbors of each point in points_test_nx3
    distances_nxk, neighbors_indexes_nxk = knn_model.kneighbors(points_test_nx3)
    points_neighbors_nxkx3 = points_train_knn_nx3[neighbors_indexes_nxk[:, 1:k_neighbors + 1], :]
    coordinates_relative_nxkx3 = points_neighbors_nxkx3 - points_test_nx3[:, np.newaxis, :]
    thetas_z_nxk, thetas_x_nxk, thetas_y_nxk = cart2sph(coordinates_relative_nxkx3)
    dists_nxk = distances_nxk[:, 1:] / longest_distance

    # fill features_nxkp2x4 with features based on neighbors
    features_nxkp2x4 = np.empty((points_test_nx3.shape[0], k_neighbors + 2, NUM_FEATURES))
    features_nxkp2x4[:, :-2, :]  = np.stack((dists_nxk, thetas_z_nxk, thetas_x_nxk, thetas_y_nxk), axis=2)

    # fill features_nxkp2x4 with features based on farthest point
    coordinates_farthest_nx3 = convex_points_mx3[longest_distance_index_n, :] - points_test_nx3
    theta_farthestz_nx1, theta_farthestx_nx1, theta_farthesty_nx1 = cart2sph(coordinates_farthest_nx3[:, None, :])
    longest_distance_nx1 = longest_distance_n[:, None] / longest_distance
    features_nxkp2x4[:, -2, :] = np.hstack((longest_distance_nx1, theta_farthestz_nx1, theta_farthestx_nx1, theta_farthesty_nx1))

    # fill features_nxkp2x4 with features based on centroid point
    coordinates_relative_centroid_nx3 = np.mean(points_train_knn_nx3, axis=0)[None, :] - points_test_nx3
    theta_centroidz_nx1, theta_centroidx_nx1, theta_centroidy_nx1, centroid_distance_nx1 = cart2sph(coordinates_relative_centroid_nx3[:, None, :], return_dist=True)
    centroid_distance_nx1 = centroid_distance_nx1 / longest_distance
    features_nxkp2x4[:, -1, :] = np.hstack((centroid_distance_nx1, theta_centroidz_nx1, theta_centroidx_nx1, theta_centroidy_nx1))
    return features_nxkp2x4


def get_convex_points(points: ndarray) -> ndarray:
    return points[ConvexHull(points).vertices, :]


def cart2sph(coordinates_nxkx3: ndarray, return_dist: bool = False) -> \
        Union[Tuple[ndarray, ndarray, ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]]:
    x_nxk = coordinates_nxkx3[..., 0]
    y_nxk = coordinates_nxkx3[..., 1]
    z_nxk = coordinates_nxkx3[..., 2]

    hxy_nxk = np.hypot(x_nxk, y_nxk)
    hxz_nxk = np.hypot(x_nxk, z_nxk)
    hzy_nxk = np.hypot(z_nxk, y_nxk)
    theta_z_nxk = np.arctan2(z_nxk, hxy_nxk) / np.pi
    theta_x_nxk = np.arctan2(x_nxk, hzy_nxk) / np.pi
    theta_y_nxk = np.arctan2(y_nxk, hxz_nxk) / np.pi
    if return_dist:
        r_nxk = np.hypot(z_nxk, hxy_nxk)
        return theta_z_nxk, theta_x_nxk, theta_y_nxk, r_nxk
    else:
        return theta_z_nxk, theta_x_nxk, theta_y_nxk


def cal_spring_net_acc(coords_nx3: ndarray, velocity_nx3: ndarray, connections_mx2: ndarray,
                       equilibrium_lengths_m: ndarray, node_with_force: dict = None, k=1, c=1):
    """
    Calculate the net acceleration of nodes in a spring-connected system.

    This function computes the accelerations for a set of nodes that are interconnected by springs,
    based on Hooke's law. The system is defined by the coordinates of the nodes, the connections between them
    (i.e., which nodes each spring connects), and the equilibrium lengths of the springs.

    Parameters:
    coords_nx3 (ndarray): An array of shape (n, 3), where n is the number of nodes.
                          Each row represents the x, y, z coordinates of a node.
    connections_mx2 (ndarray): An array of shape (m, 2), where m is the number of springs.
                               Each row represents a spring, with the values being indices
                               into coords_nx3 of the two nodes the spring connects.
    equilibrium_lengths_m (ndarray): A one-dimensional array of length m, where each element
                                     is the equilibrium length of the corresponding spring in connections_mx2.
    node_with_force (dict): A dictionary containing information about an external force applied to a node.
                            It should have the following keys:
                            - "node" (int): The index of the node to which the force is applied.
                            - "polar angle" (float): The polar angle (in degrees) of the force direction.
                            - "azimuthal angle" (float): The azimuthal angle (in degrees) of the force direction.
                            - "strength" (float): The magnitude of the force.

    Returns:
    ndarray: An array of shape (n, 3) containing the net acceleration of each node.
    """
    a_nx3 = np.zeros(coords_nx3.shape)
    k_m = k / equilibrium_lengths_m
    # Hooke's law: a = - k * displacement / m along spring, here m = 1 for all nodes
    vertices1_m = connections_mx2[:, 0]
    vertices2_m = connections_mx2[:, 1]
    displacements_mx3 = coords_nx3[vertices1_m, :] - coords_nx3[vertices2_m, :]
    current_lengths_m = np.linalg.norm(displacements_mx3, axis=1)
    diff_lengths_m = (current_lengths_m - equilibrium_lengths_m)
    acc = - k_m * diff_lengths_m / current_lengths_m
    ax = acc * displacements_mx3[:, 0]
    ay = acc * displacements_mx3[:, 1]
    az = acc * displacements_mx3[:, 2]
    np.add.at(a_nx3[:, 0], vertices1_m, ax)
    np.add.at(a_nx3[:, 1], vertices1_m, ay)
    np.add.at(a_nx3[:, 2], vertices1_m, az)
    np.add.at(a_nx3[:, 0], vertices2_m, -ax)
    np.add.at(a_nx3[:, 1], vertices2_m, -ay)
    np.add.at(a_nx3[:, 2], vertices2_m, -az)

    if node_with_force is not None:
        ax_force, ay_force, az_force, node = polar2cartesian(node_with_force)
        np.add.at(a_nx3[:, 0], node, ax_force)
        np.add.at(a_nx3[:, 1], node, ay_force)
        np.add.at(a_nx3[:, 2], node, az_force)

    a_nx3 -= velocity_nx3 * c

    return a_nx3


def polar2cartesian(node_with_vector: dict):
    theta = np.deg2rad(node_with_vector["polar angle"])
    phi = np.deg2rad(node_with_vector["azimuthal angle"])
    r = node_with_vector["strength"]
    node = node_with_vector["node"]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z, node


def cal_spring_net_coords(coords_nx3: ndarray, velocity_nx3: ndarray, acc_nx3: ndarray, dt: float):
    """
    Calculate the new positions and velocities of nodes in a spring-connected system using midpoint method.
    """
    new_velocity = velocity_nx3 + acc_nx3 * dt
    new_coords = coords_nx3 + 0.5 * (velocity_nx3 + new_velocity) * dt
    return new_coords, new_velocity


def get_connections(tri: Delaunay):
    connections: List[Tuple[int, int]] = []
    for pt_1, pt_2, pt_3, pt_4 in  tri.simplices:
        add_pair(pt_1, pt_2, connections)
        add_pair(pt_2, pt_3, connections)
        add_pair(pt_1, pt_3, connections)
        add_pair(pt_1, pt_4, connections)
        add_pair(pt_2, pt_4, connections)
        add_pair(pt_3, pt_4, connections)
    connections_mx2 = np.asarray(connections)
    connections_mx2 = connections_mx2[np.lexsort((connections_mx2[:, 1], connections_mx2[:, 0]))]
    return connections_mx2


def add_pair(pt_1, pt_2, connections: List[Tuple[int, int]]):
    pair = (pt_1, pt_2) if pt_1 < pt_2 else (pt_2, pt_1)
    if pair not in connections:
        connections.append(pair)


def align_rotation(ref_pos_nx3: ndarray, tgt_pos_nx3xt: ndarray):
    tform = estimate_transform("euclidean", tgt_pos_nx3xt[..., -1], ref_pos_nx3)
    aligned_tgt_pos_nx3xt = np.einsum("ndt, dl -> nlt", tgt_pos_nx3xt, tform.params.T[:3, :3]) + tform.params.T[3:4, :3, None]
    return aligned_tgt_pos_nx3xt


def cal_connections(points_nx3: ndarray):
    tri = Delaunay(points_nx3)
    triangles = []
    for tetra in tri.simplices:
        triangles.append([tetra[0], tetra[1], tetra[2]])
        triangles.append([tetra[0], tetra[1], tetra[3]])
        triangles.append([tetra[0], tetra[2], tetra[3]])
        triangles.append([tetra[1], tetra[2], tetra[3]])

    connections_mx2 = get_connections(tri)
    equilibrium_disp_m = points_nx3[connections_mx2[:, 0]] - points_nx3[connections_mx2[:, 1]]
    equilibrium_lengths_m = np.linalg.norm(equilibrium_disp_m, axis=1)
    return connections_mx2, equilibrium_lengths_m


def func_spring_network(t, x_and_v_6n: np.ndarray, connections_mx2: ndarray, equilibrium_lengths_m: ndarray):
    assert len(x_and_v_6n) % 2 == 0
    len_3n = len(x_and_v_6n) // 2
    assert len_3n % 3 == 0
    pos_n = x_and_v_6n[:len_3n].reshape((-1, 3))
    v_n = x_and_v_6n[len_3n:].reshape((-1, 3))
    v_dev = cal_spring_net_acc(coords_nx3=pos_n, velocity_nx3=v_n, connections_mx2=connections_mx2,
                       equilibrium_lengths_m=equilibrium_lengths_m, node_with_force=None,
                               k=1, c=3).flatten()
    pos_dev = x_and_v_6n[len_3n:]
    return np.concatenate((pos_dev, v_dev), axis=0)


def random_deform_spring(points_nx3: ndarray):
    movements = {"node": np.random.randint(0, points_nx3.shape[0] + 1),
                 "polar angle": np.random.uniform(0,180),
                 "azimuthal angle": np.random.uniform(0,360),
                 "strength": 0.4,
                 "sigma": np.random.uniform(0.4,0.6)}
    points_with_deform = apply_deform_0_max(points_normalized_nx3=points_nx3, movements=movements)
    pos = points_with_deform.flatten()
    v = np.zeros_like(pos)
    y = np.concatenate((pos, v), axis=0)

    connections_mx2, equilibrium_lengths_m = cal_connections(points_nx3)

    func_spring_partial = partial(func_spring_network, connections_mx2=connections_mx2, equilibrium_lengths_m=equilibrium_lengths_m)
    t_max = 15
    sol = solve_ivp(func_spring_partial, (0, t_max), t_eval=(t_max+1)**(np.arange(0, 100)/99)-1, y0=y)

    n, t = sol.y.shape
    points_updated = sol.y.reshape((n // 3, 3, t))
    aligned_points_nx3xt = align_rotation(ref_pos_nx3=points_nx3, tgt_pos_nx3xt=points_updated[:n // 6])
    displacements_mx3xt = aligned_points_nx3xt[connections_mx2[:,0], ...] - aligned_points_nx3xt[connections_mx2[:,1], ...]
    dist_mxt = np.linalg.norm(displacements_mx3xt, axis=1)
    energy_t = np.sum(0.5 * (1.0 / equilibrium_lengths_m)[:, None] * (dist_mxt - equilibrium_lengths_m[:, None])**2, axis=0)
    return aligned_points_nx3xt, sol.t, energy_t


def sample_random_deform_spring(points_nx3: ndarray):
    aligned_points_nx3xt, timings_t, energy_t = random_deform_spring(points_nx3)
    samples_timings_10 = np.zeros((10,), dtype=int)
    upper_limit = 0.9 * energy_t.max()
    lower_limit = max(0.1 * energy_t.max(), energy_t.min())
    pointer = 1
    for i, level in enumerate(np.linspace(upper_limit, lower_limit, 10)):
        samples_timings_10[i] = np.argmin(np.abs(energy_t[pointer:] - level)) + pointer
        pointer = samples_timings_10[i] + 1
    print(samples_timings_10)
    return aligned_points_nx3xt, timings_t, energy_t, samples_timings_10