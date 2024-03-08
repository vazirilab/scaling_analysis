"""
Information metrics for neuronal populations.
"""

import numpy as np


def dprimesq(X1, X2, shuffle=False, diagonal=False):
    """
    Calculates the squared sensitivity index (d') between two distributions.
    Modeled after Rumyantsev et al. 2021 [1]_.

    Parameters
    ----------
    X1 : array_like of shape (n, t1_observations)
    X2 : array_like of shape (n, t2_observations)
    shuffle : bool, default=False
        Whether to shuffle observations
    diagonal : bool, default=False
        Whether to assume diagonal covariance matrix

    Returns
    -------
    d2 : float
        Value of squared sensitivity index
    cov : array_like of shape (n, n)
        Covariance matrix

    References
    ----------
    .. [1] Rumyantsev, O. I., Lecoq, J. A., Hernandez, O., Zhang, Y., Savall, J.,
           Chrapkiewicz, R., ... & Schnitzer, M. J. (2020). Fundamental bounds on
           the fidelity of sensory cortical coding. Nature, 580(7801), 100-105.
           https://doi.org/10.1038/s41586-020-2130-2.
    """

    if shuffle:
        X1 = X1[:, np.random.permutation(X1.shape[1])]
        X2 = X2[:, np.random.permutation(X2.shape[1])]

    cov = (
        np.cov(X1) + np.cov(X2)
    ) / 2  # assume covariance is same across distributions

    dmu = np.mean(X1, axis=1) - np.mean(X2, axis=1)

    if diagonal:
        cov_diag = np.diag(np.diag(cov))
        d2 = (dmu @ np.linalg.inv(cov_diag) @ dmu.T) ** 2 / (
            dmu @ np.linalg.inv(cov_diag) @ cov @ np.linalg.inv(cov_diag) @ dmu.T
        )

        return d2, cov_diag
    elif shuffle:
        cov_diag = np.diag(np.diag(cov))
        d2 = dmu @ np.linalg.inv(cov_diag) @ dmu.T

        return d2, cov_diag
    else:
        d2 = dmu @ np.linalg.inv(cov) @ dmu.T

        return d2, cov


### INFORMATION THEORY


def entropy(cov):
    """Computes entropy under the Gaussian assumption, given a covariance matrix."""

    n = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)

    if sign < 0:
        print("negative sign in logdet during entropy!")

    return 1 / 2 * logdet + 1 / 2 * n * np.log(2 * np.pi * np.exp(1))


def conditional_cov(covX, covXY, covY=None):
    """Computes conditional covariance, given single and joint covariance matrices."""

    if covY is None:
        covY = covX

    return covX - np.dot(np.linalg.lstsq(covY.T, covXY.T, rcond=None)[0].T, covXY.T)


def mutual_information(covX, covXY, covY):
    """Computes mutual information under the Gaussian assumption.
    Mutual information is given as the estimated reduction of uncertainty (entropy) about
    system X given system Y.
    """

    return entropy(covX) - entropy(conditional_cov(covX, covXY, covY))
