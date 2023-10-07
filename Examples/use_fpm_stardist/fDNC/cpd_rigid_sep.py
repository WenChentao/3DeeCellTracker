from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *
import numpy.matlib
import numpy as np
from cpd_p import cpd_p_sep, cpd_p
from cpd_p_color import cpd_p_color

#def register_translation(x, y, w, x_c=[], y_c=[], sigma2c=0.1, max_it=50):

def register_translation(x, y, w, max_it=50, x_c=[], y_c=[], sigma2c=0.1):
    """
    Registers Y to X using the Coherent Point Drift algorithm, in rigid fashion.
    Note: For affine transformation, t = scale*y*r'+1*t'(* is dot). r is orthogonal rotation matrix here.
    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    max_it : int
        Maximum number of iterations. The default value is 150.

    Returns
    -------
    t : ndarray
        The transformed version of y. Output shape is [n_points_y, n_dims].
    """
    # get dataset lengths and dimensions
    [n, d] = x.shape
    [m, d] = y.shape

    if isinstance(x_c, list):
        if len(x_c) and len(y_c):
            use_color = 1
        else:
            use_color = 0
    else:
        if x_c.shape[1] and y_c.shape[1]:
            use_color = 1
        else:
            use_color = 0

    if use_color:
        [nc, dc] = x_c.shape
        [mc, dc] = y_c.shape
    #      sigma2c = (mc*np.trace(np.dot(np.transpose(x_c), x_c))+ nc*np.trace(np.dot(np.transpose(y_c), y_c)) -
    #              2*np.dot(sum(x_c), np.transpose(sum(y_c))))/(mc*nc*dc)
    # t is the updated moving shape,we initialize it with y first.
    t = np.copy(y)
    # initialize sigma^2
    sigma2 = (m * np.trace(np.dot(np.transpose(x), x)) + n * np.trace(np.dot(np.transpose(y), y)) -
              2 * np.dot(sum(x), np.transpose(sum(y)))) / (m * n * d)
    iter = 0
    while (iter < max_it) and (sigma2 > 10.e-8):

        if use_color:
            [p1, pt1, px, px_c] = cpd_p_color(x, t, sigma2, w, m, n, d, x_c, y_c, sigma2c)
        else:
            [p1, pt1, px, p_dis] = cpd_p(x, t, sigma2, w, m, n, d, return_dis=True)
        # precompute
        Np = np.sum(pt1)
        mu_x = np.dot(np.transpose(x), pt1) / Np
        mu_y = np.dot(np.transpose(y), p1) / Np
        sigma2 = p_dis / (Np * d)

        ts = mu_x - mu_y
        t = y + numpy.matlib.repmat(np.transpose(ts), m, 1)
        iter = iter + 1
    # if use_color:
    # print('sigma c:{}'.format(sigma2c))

    return t, ts


def register_rigid(x, y, w, max_it=250, x_c=[], y_c=[], fix_scale=False, sigma2c=0.1):
    """
    This function only trannsform x and y coordinate.
    Registers Y to X using the Coherent Point Drift algorithm, in rigid fashion.
    Note: For affine transformation, t = scale*y*r'+1*t'(* is dot). r is orthogonal rotation matrix here.
    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    max_it : int
        Maximum number of iterations. The default value is 150.

    Returns
    -------
    t : ndarray
        The transformed version of y. Output shape is [n_points_y, n_dims].
    """
    # get dataset lengths and dimensions
    assert len(x) > 0
    assert len(y) > 0
    [n, d] = x.shape
    [m, d] = y.shape
    assert d >= 2

    if isinstance(x_c, list):
        if len(x_c) and len(y_c):
            use_color = 1
        else:
            use_color = 0
    else:
        if x_c.shape[1] and y_c.shape[1]:
            use_color = 1
        else:
            use_color = 0

    if use_color:
        [nc, dc] = x_c.shape
        [mc, dc] = y_c.shape
    #      sigma2c = (mc*np.trace(np.dot(np.transpose(x_c), x_c))+ nc*np.trace(np.dot(np.transpose(y_c), y_c)) -
    #              2*np.dot(sum(x_c), np.transpose(sum(y_c))))/(mc*nc*dc)
    # t is the updated moving shape,we initialize it with y first.
    t = np.copy(y)
    # initialize sigma^2
    sigma2 = (m * np.trace(np.dot(np.transpose(x), x)) + n * np.trace(np.dot(np.transpose(y), y)) -
              2 * np.dot(sum(x), np.transpose(sum(y)))) / (m * n * d)
    iter = 0
    while (iter < max_it) and (sigma2 > 10.e-8):
        # E step (P matrix)
        if use_color:
            [p1, pt1, px, px_c] = cpd_p_color(x, t, sigma2, w, m, n, d, x_c, y_c, sigma2c)
        else:
            [p1, pt1, p] = cpd_p_sep(x, t, sigma2, w, m, n, d)
        # precompute
        Np = np.sum(pt1)

        # M step, update transformation.
        x_2d, y_2d = x[:, :2], y[:, :2]
        x_z, y_z = x[:, 2:3], y[:, 2:3]
        d = 2
        # align z
        mu_z1 = np.dot(np.transpose(x_z), pt1) / Np
        mu_z2 = np.dot(np.transpose(y_z), p1) / Np
        t_z = mu_z1 - mu_z2
        t[:, 2:3] = y_z + t_z
        # align x and y
        mu_x = np.dot(np.transpose(x_2d), pt1) / Np
        mu_y = np.dot(np.transpose(y_2d), p1) / Np
        # solve for Rotation, scaling, translation and sigma^2
        px = np.dot(p, x_2d)
        a = np.dot(np.transpose(px), y_2d) - Np * (np.dot(mu_x, np.transpose(mu_y)))
        [u, s, v] = np.linalg.svd(a)
        s = np.diag(s)
        c = np.eye(d)
        c[-1, -1] = np.linalg.det(np.dot(u, v))
        r = np.dot(u, np.dot(c, v))
        if fix_scale:
            scale = 1
        else:
            scale = np.trace(np.dot(s, c)) / (sum(sum(y_2d * y_2d * numpy.matlib.repmat(p1, 1, d))) - Np *
                                              np.dot(np.transpose(mu_y), mu_y))
        sigma22 = np.abs(sum(sum(x_2d * x_2d * numpy.matlib.repmat(pt1, 1, d))) - Np *
                         np.dot(np.transpose(mu_x), mu_x) - scale * np.trace(np.dot(s, c))) / (Np * d)
        sigma2 = sigma22[0][0]
        #        if use_color:
        #          sigma2c = np.abs((np.sum(x_c*x_c*np.matlib.repmat(pt1, 1, dc))+np.sum(y_c*y_c*np.matlib.repmat(p1, 1, dc)) -
        #                         2*np.trace(np.dot(px_c.T, y_c)))/(Np*dc))
        # ts is translation
        ts = mu_x - np.dot(scale * r, mu_y)
        t_2d = np.dot(scale * y_2d, np.transpose(r)) + numpy.matlib.repmat(np.transpose(ts), m, 1)
        t[:, :2] = t_2d
        iter = iter + 1
    # if use_color:
    # print('sigma c:{}'.format(sigma2c))
    if x.shape[1] == 3:
        r_out = np.eye(3)
        r_out[:2, :2] = r
        ts_out = np.zeros((3, 1))
        ts_out[:2, :] = ts
        ts_out[2, 0] = t_z
    else:
        r_out = r
        ts_out = ts


    return t, r_out, ts_out, sigma2
