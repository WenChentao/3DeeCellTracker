from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import numpy as np


def cpd_p_color(x, y, sigma2, w, m, n, d, x_c=[], y_c=[], sigma2_c=0.2, return_p=False):
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
    
  # Additional part to include color information

    if len(x_c.shape) > 1:
      d_c = x_c.shape[1]
    else:
      x_c = x_c.reshape(-1,1)
      y_c = y_c.reshape(-1,1)
      d_c = 1
    
    g_c = x_c[:, np.newaxis, :]-y_c
    g_c = g_c*g_c
    g_c = np.sum(g_c, 2)
    g_c = np.exp(-1.0/(2*sigma2_c)*g_c)
    
    # using numpy broadcasting to build a new matrix.
    g = x[:, np.newaxis, :]-y
    g = g*g
    g = np.sum(g, 2)
    g = np.exp(-1.0/(2*sigma2)*g)      
    g = g * g_c
    # g1 is the top part of the expression calculating p
    # temp2 is the bottom part of expresion calculating p
    g1 = np.sum(g, 1)
    temp2 = (g1 + (2*np.pi*sigma2)**(d/2)*(2*np.pi*sigma2_c)**(d_c/2)*w/(1-w)*(float(m)/n)).reshape([n, 1])
    p = (g/temp2).T
    p1 = (np.sum(p, 1)).reshape([m, 1])
    px = np.dot(p, x)
    px_c = np.dot(p, x_c)
    pt1 = (np.sum(np.transpose(p), 1)).reshape([n, 1])
    if return_p:
        return p1, pt1, px, px_c, p
    else:
        return p1, pt1, px, px_c
