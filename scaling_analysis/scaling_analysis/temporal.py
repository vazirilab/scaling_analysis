"""
Analysis of the temporal characteristics of SVCs.
"""

import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import zscore
import sys
import warnings

from scaling_analysis.experiment import Experiment
from scaling_analysis.svca import run_SVCA_partition
from scaling_analysis.utils import get_nneur


def autocorr(x):
    """Computes the autocorrelation of x."""
    result = np.correlate(x, x, mode="full")
    return result[result.size // 2 :]


def fit_exponential_decay(x, t):
    """
    Fits an exponential decay to the signal x(t).

    Parameters
    ----------
    x : array_like of length N
        Signal
    t : array_like of length N
        Timestamps

    Returns
    -------
    tau : float
        Estimated decay timescale
    error : float
        Error associated with estimate
    """

    A = np.max(x)
    exp_decay = lambda x, t: A * np.exp(-x / t)

    warnings.filterwarnings("ignore")
    # `curve_fit` can be noisy; up to the user to inspect its results
    params, cov = curve_fit(exp_decay, t, x)
    warnings.filterwarnings("default")

    return params[0], np.sqrt(cov[0][0])


def compute_timescales(svcs, t):
    """
    Computes the autocorrelation timescale associated with each column of svcs.

    Parameters
    ----------
    svcs : array_like of shape (T, N)
    t : array_like of length T
        Timestamps associated with svcs

    Returns
    -------
    acorrs : ndarray of shape (T, N)
        Autocorrelation of each column of svcs
    timescales : ndarray of shape (N,)
        Autocorrelation timescale estimated for each column of svcs
    timescales_perr : ndarray of shape (N,)
        Error associated with estimate of timescales
    """

    acorrs = np.zeros(svcs.shape)
    timescales = np.zeros((svcs.shape[1],))
    timescales_perr = np.zeros((svcs.shape[1],))

    for i in range(svcs.shape[1]):
        acorrs[:, i] = autocorr(svcs[:, i])
        tau, err = fit_exponential_decay(acorrs[:, i], t)
        timescales[i] = tau
        timescales_perr[i] = err

    return acorrs, timescales, timescales_perr


def find_svca_temporal(neurons, t, nneur, nsamplings=10, nsvc=4096, out=None, **kwargs):
    """
    First computes SVCA on neurons with nneur sampled neurons.
    Then finds the autocorrelation timescale of each SVC.

    Parameters
    ----------

    neurons : array_like of shape (N, T)
        Neuronal activity matrix; assumed N ≥ nneur
    t : array_like of length T
        Timestamps
    nneur : int
        Number of neurons to sample
    nsamplings : int, default=10
       Number of samplings and rounds of SVCA/prediction to perform
    nsvc : int, default=4096
        Number of SVCs to compute (if not None, uses randomized SVD approximation)
        i.e., n_randomized in PopulationCoding.dimred.SVCA
    out : str, optional
        Path to optionally save results
    kwargs : passed to run_SVCA_partition

    Returns
    -------
    cov_neur : ndarray of shape (nsvc, nsamplings)
        Reliable covariance of each SVC on held-out testing timepoints
    var_neur : ndarray of shape (nsvc, nsamplings)
        Reliable variance of each SVC on held-out testing timepoints
    timescales : ndarray of shape (nsvc, nsamplings)
        Autocorrelation timescales of each SVC
    timescales_perr : ndarray of shape (nsvc, nsamplings)
        Error associated with estimate of timescales
    """

    cov_neur = np.zeros((nsvc, nsamplings))
    var_neur = np.zeros((nsvc, nsamplings))

    timescales = np.zeros((nsvc, nsamplings))
    timescales_perr = np.zeros((nsvc, nsamplings))

    for s in range(nsamplings):
        print("SAMPLING", s + 1, "OUT OF", nsamplings)

        sneur, varneur, u, v, ntrain1, ntest1, itrain, itest, pca = run_SVCA_partition(
            neurons, nneur, n_randomized=nsvc, **kwargs
        )

        cov_neur[: len(sneur), s] = sneur
        var_neur[: len(sneur), s] = varneur

        if pca is not None:
            trainX = pca["train_projs"]
        else:
            trainX = neurons[:, ntrain1]

        svcs = trainX @ u

        print("    COMPUTING TIMESCALES")
        acorrs, timescales_curr, timescales_perr_curr = compute_timescales(svcs, t)
        timescales[:, s] = timescales_curr
        timescales_perr[:, s] = timescales_perr_curr

    if out is not None:
        np.savez(
            os.path.join(out, "temporal_nneur" + str(nneur)),
            cov_neur=cov_neur,
            var_neur=var_neur,
            timescales=timescales,
            timescales_perr=timescales_perr,
            projs1=svcs,
            acorrs1=acorrs,
            u1=u,
            v1=v,
            ntrain1=ntrain1,
            ntest1=ntest1,
            itrain1=itrain,
            itest1=itest,
            **kwargs
        )

    return cov_neur, var_neur, timescales, timescales_perr


def main(path, nneur, nsamplings=10, shuffled=False):
    """
    CLI for scaling_analysis.temporal.

    For example, run 10 samplings of SVCA on 512 neurons
    and compute their autocorrelation timescales:

    :code:`python -m scaling_analysis.temporal /path/to/file.h5 512 10 0`

    Parameters
    ----------
    path : str
        Path to dataset loaded by Experiment
    nneur : int
        Number of neurons to randomly sample
    nsamplings : int, default=1
        Number of random samplings to perform
    shuffled : bool, default=False
        Whether or not to shuffle the timeseries by randomly permuting
        two second chunks
    """

    nneur = int(nneur)
    nsamplings = int(nsamplings)
    shuffled = bool(int(shuffled))  # in case user provides "0" to CLI

    expt = Experiment(path)

    neurons = zscore(expt.T_all.astype("single"))

    nneur = get_nneur(neurons.shape[1], nneur)

    if shuffled:
        print("shuffling!")
        # Note: this shuffling procedure is distinct from the rest of the analyses
        Xshuff = neurons
        for i in range(neurons.shape[1]):
            nchunks = int(Xshuff.shape[0] / (expt.fhz * 2))
            tshuff = np.array_split(
                np.roll(
                    np.arange(Xshuff.shape[0]),
                    np.random.permutation(Xshuff.shape[0])[0],
                ),
                nchunks,
            )
            tshuff = [tshuff[i] for i in np.random.permutation(len(tshuff))]
            tshuff = [x for y in tshuff for x in y]
            Xshuff[:, i] = Xshuff[:, i][tshuff]
        neurons = Xshuff

    out = os.path.join(expt.out, "temporal_svca")

    if shuffled:
        out += "_shuffled"
    print("saving temporal autocorrelation results to", out)
    if not os.path.exists(out):
        os.mkdir(out)

    prePCA = nneur > 131072

    find_svca_temporal(
        neurons,
        expt.t,
        nneur,
        nsamplings=nsamplings,
        nsvc=min(int(nneur / 2), 2048),
        out=out,
        shuffle=shuffled,
        centers=expt.centers,
        checkerboard=250,
        interleave=int(72 * expt.fhz),
        prePCA=prePCA,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
