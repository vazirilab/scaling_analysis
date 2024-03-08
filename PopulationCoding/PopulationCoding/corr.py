"""
Measuring and identifying significant correlations.
"""

import numpy as np


def corr(A, B):
    """
    Calculates the correlation matrix between variables in two arrays.

    Parameters
    ----------
    A : array_like of shape (m_features, t_observations)
    B : array_like of shape (n_features, t_observations)

    Returns
    -------
    C : ndarray of shape (m_features, n_features)
        Correlation matrix
    """

    Am = A - np.mean(A, axis=1)[:, None]
    Bm = B - np.mean(B, axis=1)[:, None]

    ssA = np.sum(Am**2, axis=1)
    ssB = np.sum(Bm**2, axis=1)

    C = np.dot(Am, Bm.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    return C


def sig_stim_corr(X, y, thresh=3, nshuff=100, random=False):
    """
    Identify timeseries in X that are significantly correlated with another "stimulus" y.

    Parameters
    ----------
    X : array_like of shape (n_features, t_observations)
        Timeseries data
    y : array_like of shape (1, t_observations)
        Stimulus vector
    thresh : float, default=3
        Number of standard deviations above mean required for significance
    nshuff : int, default=100
        Number of shuffling tests
    random : bool, default=False
        If false, shuffling is done by circular permutations of y.
        If true,  shuffling is done by creating new shuffles of repeated stimuli in y
        (assuming y is 0 or a value)

    Returns
    -------
    C : ndarray of shape (n_features,)
        Correlations of each feature with the stimulus y
    idx_sig : tuple
        Indices of significantly correlated variables in X,
        as a tuple where idx_sig[0] are positively-correlated and idx_sig[1] negatively
    c_thresh : tuple
        Thresholds (upper, lower) on correlation
    C_shuff : ndarray of shape (n_features, nshuff)
        Shuffled correlations
    """

    if len(y.shape) < 2:
        y = y.reshape((1, len(y)))

    C = corr(X, y)

    C_shuff = np.zeros((C.shape[0], nshuff))

    if random:
        nstim = int(np.sum(np.abs(np.diff(y > 0))) / 2)
        d = np.diff((y > 0).astype(np.float32))
        idx_start = np.where(d > 0)[1][0]
        idx_end = np.where(d < 0)[1][0]
        kernel = y[0, idx_start:idx_end]

    for i in range(nshuff):
        if random:
            yp = np.zeros(y.shape)
            yp[0, np.random.permutation(np.arange(y.shape[1]))[:nstim]] = 1
            yp = np.convolve(
                yp.reshape(
                    -1,
                ),
                kernel,
                mode="same",
            ).reshape(1, -1)
        else:
            yp = np.roll(y, np.random.randint(y.shape[1], size=(1,)), axis=1)
        C_shuff[:, i] = corr(X, yp)[:, 0]

    c_thresh = (
        np.mean(C_shuff, axis=1) + thresh * np.std(C_shuff, axis=1),
        np.mean(C_shuff, axis=1) - thresh * np.std(C_shuff, axis=1),
    )

    idx_sig = (np.where(C[:, 0] > c_thresh[0])[0], np.where(C[:, 0] < c_thresh[1])[0])

    return C, idx_sig, c_thresh, C_shuff
