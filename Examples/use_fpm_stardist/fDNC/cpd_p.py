from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *
import numpy as np


def cpd_p(x, y, sigma2, w, m, n, d, return_dis=False):
    """
    E-step:Compute P in the EM optimization,which store the probobility of point n in x belongs the cluster m in y.

    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    sigma2 : float
        Gaussian distribution parameter.It will be calculated in M-step every loop.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    m : int
        x points' length. The reason that making it a parameter here is for avioding calculate it every time.
    n : int
        y points' length. The reason that making it a parameter here is for avioding calculate it every time.
    d : int
        Dataset's dimensions. Note that d should be equal for x and y.

    Returns
    -------
    p1 : ndarray
        The result of dot product of the matrix p and a column vector of all ones.
        Expected array shape is [n_points_y,1].
    pt1 : nadarray
        The result of dot product of the inverse matrix of p and a column vector of all ones. Expected array shape is
        [n_points_x,1].
    px : nadarray
        The result of dot product of the matrix p and matrix of dataset x.
    """
    # using numpy broadcasting to build a new matrix.
    g = x[:, np.newaxis, :]-y
    g = g*g
    g = np.sum(g, 2)
    if return_dis:
        x_y = np.copy(g)

    g = np.exp(-1.0/(2*sigma2)*g)
    # g1 is the top part of the expression calculating p
    # temp2 is the bottom part of expresion calculating p
    g1 = np.sum(g, 1)
    temp2 = (g1 + (2*np.pi*sigma2)**(d/2)*w/(1-w)*(float(m)/n)).reshape([n, 1])
    p = (g/temp2).T
    if return_dis:
        pxy_dis = (p * x_y.T).sum()

    p1 = (np.sum(p, 1)).reshape([m, 1])
    px = np.dot(p, x)
    pt1 = (np.sum(np.transpose(p), 1)).reshape([n, 1])
    if return_dis:
        return p1, pt1, px, pxy_dis
    else:
        return p1, pt1, px

def cpd_p_sep(x, y, sigma2, w, m, n, d):
    """
    E-step:Compute P in the EM optimization,which store the probobility of point n in x belongs the cluster m in y.

    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    sigma2 : float
        Gaussian distribution parameter.It will be calculated in M-step every loop.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    m : int
        x points' length. The reason that making it a parameter here is for avioding calculate it every time.
    n : int
        y points' length. The reason that making it a parameter here is for avioding calculate it every time.
    d : int
        Dataset's dimensions. Note that d should be equal for x and y.

    Returns
    -------
    p1 : ndarray
        The result of dot product of the matrix p and a column vector of all ones.
        Expected array shape is [n_points_y,1].
    pt1 : nadarray
        The result of dot product of the inverse matrix of p and a column vector of all ones. Expected array shape is
        [n_points_x,1].
    px : nadarray
        The result of dot product of the matrix p and matrix of dataset x.
    """
    # using numpy broadcasting to build a new matrix.
    g = x[:, np.newaxis, :]-y
    g = g*g
    g = np.sum(g, 2)
    g = np.exp(-1.0/(2*sigma2)*g)
    # g1 is the top part of the expression calculating p
    # temp2 is the bottom part of expresion calculating p
    g1 = np.sum(g, 1)
    temp2 = (g1 + (2*np.pi*sigma2)**(d/2)*w/(1-w)*(float(m)/n)).reshape([n, 1])
    p = (g/temp2).T
    p1 = (np.sum(p, 1)).reshape([m, 1])
    #px = np.dot(p, x)
    pt1 = (np.sum(np.transpose(p), 1)).reshape([n, 1])
    return p1, pt1, p


