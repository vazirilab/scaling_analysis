"""
Fit predictive models between multi-dimensionsal datasets.
"""


import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.utils.extmath import randomized_svd


def canonical_cov(Y, X, lam, npc=512, **kwargs):
    """Canonical covariance analysis to predict Y from X,
    based on originally `CanonCor2 <https://github.com/MouseLand/stringer-pachitariu-et-al-2018a>`_
    by `Stringer et al. 2019 <https://github.com/MouseLand/stringer-pachitariu-et-al-2018a/blob/master/utilities/CanonCor2.m>`_.

    After fitting a "sort of" canonical covariance analysis between
    two sets of data X and Y. The approximation of Y based on n
    projections is given by :code:`a[:, :n] @ b[:, :n].T @ X.T`.

    Parameters
    ----------
    Y : array_like of shape (m, t)
        Data matrix
    X : array_like of shape (n, t)
        Data matrix
    lam : float
        Regularization parameter
    npc : int, default=512
        Number of projections to consider

    Returns
    -------
    a : array_like of shape (m, npc)
        Projections of Y
    b : array_like of shape (n, npc)
        Projections of X
    R2 : array_like of shape (npc,)
        Proportion of total variance of Y explained by each projection
    v : array_like of shape (n, npc)
        Actual value of each linear combination of X
    """

    Xn = X.shape[1]
    Yn = Y.shape[1]

    C = np.cov(np.hstack((X, Y)).T)

    CXX = C[:Xn, :Xn] + lam * np.eye(Xn)
    CYY = C[Xn:, Xn:]
    CYX = C[Xn:, :Xn]

    CXXMH = fractional_matrix_power(CXX, -0.5)

    M = CYX @ CXXMH
    M[np.isnan(M)] = 0

    # do SVD
    u, s, vt = randomized_svd(M, n_components=npc, random_state=None, **kwargs)
    v = vt.T
    s = np.diag(s)

    b = CXXMH @ v

    a = u @ s

    R2 = (np.diag(s) ** 2) / np.sum(np.var(Y), axis=0)

    return a, b, R2, v
