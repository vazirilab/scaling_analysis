"""
Miscellaneous tools.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import warnings


### ML TOOLS


def classify_continuous(y, labels):
    """Take continuous input and classify it into the closest label."""
    return np.asarray([labels[np.argmin(np.abs(x - labels))] for x in y])


def train_test_split_idx(N, train_frac=0.5, time=False, interleave=0):

    if interleave > 0:
        indx = np.ceil(np.arange(N) / interleave)
        Nblocks = int(np.max(indx))
        irand = np.random.permutation(Nblocks)
        Ntrain = int(np.ceil(train_frac * Nblocks))
        Ntest = Nblocks - Ntrain
        itrain = np.where(np.isin(indx, np.sort(irand[:Ntrain])))[0]
        itest = np.where(np.isin(indx, np.sort(irand[Ntrain:])))[0]

        return itrain, itest
    elif time:
        test_start = np.random.randint(
            int(N * (1 - train_frac)), int(N * train_frac) + 1
        )
        return (
            np.asarray(
                list(range(test_start))
                + list(range(test_start + int(N * (1 - train_frac)), N))
            ),
            np.arange(test_start, test_start + int(N * (1 - train_frac))),
        )
    else:
        return train_test_split(np.arange(N), train_size=train_frac)


### MISC.


def bin2d(x, tbin, idim=0):
    """Bins data."""

    if idim == 1:
        x = x.T

    x = np.reshape(
        np.mean(
            np.reshape(
                x[: int(np.floor(x.shape[0] / tbin) * tbin), :], (-1, tbin, x.shape[1])
            ),
            axis=1,
        ),
        (-1, x.shape[1]),
    )
    if idim == 1:
        x = x.T

    return x


def logdet(X):
    """Computes the log of the determinant."""
    sign, lgdet = np.linalg.slogdet(X)

    if sign < 0:
        warnings.warn("Negative sign in logdet!")

    return lgdet


def generate_gaussian_data(A, T=10**4, mean_subtract=True):
    """Generates auto-regressive Gaussian test data given a connectivity matrix.

    Parameters
    ----------
    A : array_like of shape (N, N)
        Desired connectivity matrix
    T : int, default=10**4
        Number of time points to simulate
    mean_subtract : bool, default=True
        Whether or not to mean subtract the time series

    Returns
    -------
    X : array_like of shape (N, T)
        Timeseries data
    cov : array_like of shape (N, N)
        Empirical covariance matrix
    """

    N = A.shape[0]

    ## generate random gaussian time series X
    X = np.zeros((N, T))
    X[:, 0] = np.reshape(np.random.randn(N, 1), (N,))
    for t in range(1, T):
        E = np.reshape(np.random.randn(N, 1), (N,))
        X[:, t] = np.dot(A, X[:, t - 1]) + E

    ## mean subtract
    if mean_subtract:
        for i in range(N):
            X[i, :] = X[i, :] - np.mean(X[i, :])

    ## generate covariance matrix
    cov = np.dot(X, X.T) / (X.shape[0] - 1)

    return X, cov


def get_consecutive_chunks(t):
    """Finds chunks of consecutive integers in a vector t."""
    diff = np.diff(t)
    shifts = np.where(diff > 1)[0] + 1
    shifts = np.concatenate((shifts, [len(t)]), axis=0)

    chunks = []

    for i in range(len(shifts)):
        if i == 0:
            start = 0
        else:
            start = shifts[i - 1]

        chunks.append(t[start : shifts[i]])

    return chunks
