from typing import Tuple, Optional, List, Union, Generator, Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
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

RANGE_ROT_REF = (-170, 180)
RANGE_ROT_TGT = (-170, 180)


class DataGeneratorFPN:
    degrees_rotate_raw: Tuple[int] = (-135, -90, -45, 0, 45, 90, 135, 180)

    def __init__(self, points_normalized_nx3: ndarray, num_rotations: int, num_deformations: int,
                 range_rotation_ref: tuple = RANGE_ROT_REF, range_rotation_tgt: tuple = RANGE_ROT_TGT, movement_factor: float = 0.2):
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
        range_rotation_ref: tuple
            The range of rotation degrees for reference points set
        range_rotation_tgt: tuple
            The range of rotation degrees for target points set
        movement_factor: float
            The movement factor for deformations
        """
        self.num_rotate = num_rotations
        self.num_deform = num_deformations
        self.points_normalized_nx3 = points_normalized_nx3
        self.train_data_gen = self.generator_train_data(points_normalized_nx3,
                                                        range_rotation_ref=range_rotation_ref,
                                                        range_rotation_tgt=range_rotation_tgt,
                                                        movement_factor=movement_factor)

    # def generate_valid_data(self):
    #     self.train_point_sets_list = self.generate_train_point_sets(self.points_normalized_nx3)
    #     self.train_data = self.generate_train_data()

    # def generate_train_point_sets(self, points_nx3: ndarray) -> List[Tuple[ndarray, ndarray]]:
    #     """Generate synthesized points sets by rotation and deformation"""
    #     point_sets_list = []
    #     longest_distance = get_longest_distance(points_nx3)
    #     for degree in self.degrees_rotate_raw:
    #         points_rotated_nx3 = rotate(points_nx3, np.array([[0, 0, degree]]), 1)[0, :, :]
    #         target_points_mxnx3 = train_data_rotation(points_rotated_nx3, self.num_rotate)
    #         target_points_mxnx3 = train_data_deformation(target_points_mxnx3, longest_distance, self.num_deform)
    #         target_points_mxnx3 += (np.random.rand(*target_points_mxnx3.shape) - 0.5) * 4 * RANDOM_MOVEMENTS_FACTOR
    #         point_sets_list.append((points_rotated_nx3, target_points_mxnx3))
    #     return point_sets_list

    # def generate_train_data(self):
    #     points_num = self.train_point_sets_list[0][0].shape[0]
    #     generated_sets_num = self.num_rotate * self.num_deform
    #     train_sample_num = len(self.train_point_sets_list) * generated_sets_num * points_num * 2
    #     x_mxf = np.empty((train_sample_num, NUMBER_FEATURES * 2), dtype=np.float32)
    #     y_mx1 = np.empty((train_sample_num, 1), dtype=np.bool_)
    #     for i, (points_raw, generated_point_sets) in enumerate(self.train_point_sets_list):
    #         base = i * generated_sets_num * points_num * 2
    #         for j, generated_points in enumerate(generated_point_sets):
    #             points_wi_seg_errors, replaced_indexes = add_seg_errors(generated_points)
    #             s_ = slice(base + j * points_num * 2, base + (j + 1) * points_num * 2)
    #             points_to_features(x_mxf[s_, :], y_mx1[s_, 0], points_raw, points_wi_seg_errors, replaced_indexes)
    #     return TensorDataset(tensor(x_mxf), tensor(y_mx1))

    @staticmethod
    def generator_train_data(points_nx3: ndarray, batch_size: int = 128,
                             range_rotation_ref: tuple = RANGE_ROT_REF, range_rotation_tgt: tuple = RANGE_ROT_TGT,
                             movement_factor: float = 0.2) -> Generator[Tuple[ndarray, ndarray], None, None]:
        """Generate training data for FPN model

        Parameters
        ----------
        points_nx3: ndarray, shape (n, 3)
            The normalized points set
        batch_size: int
            The batch size
        range_rotation_ref: tuple, shape (2,)
            The range of rotation degrees for reference points set
        range_rotation_tgt: tuple, shape (2,)
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
        longest_distance = get_longest_distance(points_nx3)

        # Generate training data
        while True:
            # Generate a relative large number of data than the batch size
            for i in range(num_sets):
                rotation_ref = np.random.randint(*range_rotation_ref, (1, 3))
                rotation_tgt = np.random.randint(*range_rotation_tgt, (1, 3))
                points_ref_nx3, points_tgt_nx3 = generate_points_pair(points_nx3, rotation_ref, rotation_tgt,
                                                                      longest_distance, mv_factor=movement_factor)
                points_tgt_with_errors, replaced_indexes = add_seg_errors(points_tgt_nx3)
                point_set_s_ = slice(i * num_batch_per_set, (i + 1) * num_batch_per_set)
                x_bxkp2xfx2[point_set_s_, ...], y_bx1[point_set_s_, :] = \
                    points_to_features(points_ref_nx3, points_tgt_with_errors, replaced_indexes)

            # Yield small batches from the generated data set in a shuffled order
            np.random.shuffle(random_indexes)
            for i in range(num_batch // batch_size):
                batch_i_s_ = np.s_[random_indexes[i * batch_size:(i + 1) * batch_size],:]
                yield x_bxkp2xfx2[batch_i_s_], y_bx1[batch_i_s_]


def train_data_rotation(points_normalized_nx3: ndarray, num_ptr_set: int = NUM_POINT_SET) -> ndarray:
    degrees_mx3 = np.zeros((num_ptr_set, 3))
    degrees_mx3[:, 2] = np.hstack((np.linspace(0, 40, num_ptr_set//2),
                                   np.linspace(-40, -10, num_ptr_set - num_ptr_set//2)))
    return rotate(points_normalized_nx3, degrees_mx3, num_ptr_set)


def train_data_deformation(points_normalized_mxnx3: ndarray, longest_distance: float, num_ptr_set: int = NUM_POINT_SET) \
        -> ndarray:
    new_points_mxmxnx3 = np.empty(((num_ptr_set), *points_normalized_mxnx3.shape))
    for i, points_nx3 in enumerate(points_normalized_mxnx3):
        new_points_mxmxnx3[i, ...] = deform(points_nx3, longest_distance, num_ptr_set)
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


def generate_points_pair(points_nx3: ndarray, degree_ref_1x3: ndarray, degree_tgt_1x3: ndarray,
                         longest_distance: Optional[float] = None,
                         mv_factor: float = None, width: float = None):
    """
    Generate a pair of points set with rotation and deformation

    Parameters
    ----------
    points_nx3: ndarray
        The normalized points set
    degree_ref_1x3: ndarray
        degree of rotation for reference points set
    degree_tgt_1x3: ndarray
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
    points_ref_nx3 = rotate(points_nx3, degree_ref_1x3, 1)[0, :, :]
    points_tgt_nx3 = rotate(points_ref_nx3, degree_tgt_1x3, 1)[0, :, :]
    points_tgt_nx3 = deform(points_tgt_nx3, longest_distance, num_ptr_set=1, mv_factor=mv_factor, width=width)[0, :, :]
    points_tgt_nx3 += (np.random.rand(*points_tgt_nx3.shape) - 0.5) * 4 * RANDOM_MOVEMENTS_FACTOR
    return points_ref_nx3, points_tgt_nx3


def rotate(points_nx3: ndarray, degrees_mx3: ndarray, num_ptr_set: int) -> ndarray:
    new_points_mxnx3 = np.empty((num_ptr_set, points_nx3.shape[0], 3))
    rad_mx3 = degrees_mx3 * np.pi / 180
    for i in range(num_ptr_set):
        r = Rotation.from_rotvec(rad_mx3[i, :])
        new_points_mxnx3[i, ...] = np.dot(points_nx3, r.as_matrix())
    return new_points_mxnx3


def deform(points_normalized_nx3: ndarray, longest_distance: float, num_ptr_set: int, mv_factor: float = None,
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
    distances_to_targets_mxn = cdist(points_normalized_nx3[target_points_m, :], points_normalized_nx3, metric='euclidean')
    scale_factors_mxnx1 = np.exp(-0.5 * (distances_to_targets_mxn / (width * longest_distance))**2)[:, :, np.newaxis]
    all_movements_mxnx3 = scale_factors_mxnx1 * target_movement_mx1x3
    new_points_mxnx3 = points_normalized_nx3[np.newaxis, :, :] + all_movements_mxnx3
    return new_points_mxnx3


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

