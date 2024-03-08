"""
Session permutation method
for computing reliable variance noise floors
"""

import numpy as np
import os
from scipy.stats import zscore
import sys
from tqdm import tqdm

from PopulationCoding.dimred import SVCA
from PopulationCoding.utils import train_test_split_idx
from scaling_analysis.experiment import Experiment
from scaling_analysis.utils import get_nneur


def svca_across_sessions(
    neurons1,
    neurons2,
    nneur,
    nsamplings=10,
    nsvc=4096,
    out=None,
    interleave=400,
    **kwargs
):
    """
    Runs SVCA with each neural set sampled from different data matrix (e.g. different sessions),
    i.e. performing something like the so-called session permutation method [1]_.

    By sampling neurons from completely different recordings, the neural autocorrelations are
    NOT obfuscated, but the neural subsets are known a priori to not be synchronized.

    Parameters
    ----------
    neurons1 : array_like of shape (T1, N1)
    neurons2 : array_like of shape (T2, N2)
    nneur : int
        Number of neurons to sample
    nsamplings : int, default=10
       Number of samplings and rounds of SVCA to perform
    nsvc : int, default=4096
        Number of SVCs to compute (if not None, uses randomized SVD approximation).
        i.e., n_randomized in PopulationCoding.dimred.SVCA
    out : str, optional
        Path to optionally save results
    interleave : int, default=400
        Number of timepoints which are chunked together before randomly assigned chunks to train or test.
        If interleave=0, each timepoint is independently assigned to train or test
        (not recommended due to autocorrelations)

    Returns
    -------
    sneur : ndarray of shape (nsvc, nsamplings)
        Reliable covariance of each cross-session SVC on held-out testing timepoints
    varneur : ndarray of shape (nsvc, nsamplings)
        Reliable variance of each cross-session SVC on held-out testing timepoints

    References
    ----------
    .. [1] Harris, K. D. (2020). Nonsense correlations in neuroscience. bioRxiv, 2020-11.
           https://doi.org/10.1101/2020.11.29.402719.
    """

    cov_neur = np.zeros((nsvc, nsamplings))
    var_neur = np.zeros((nsvc, nsamplings))

    tmin = min(neurons1.shape[0], neurons2.shape[0])
    neurons1 = neurons1[:tmin, :]
    neurons2 = neurons2[:tmin, :]

    for s in tqdm(range(nsamplings)):
        idx_neur1 = np.random.permutation(neurons1.shape[1])[:nneur]
        idx_neur2 = np.random.permutation(neurons2.shape[1])[:nneur]

        X = np.hstack([neurons1[:, idx_neur1], neurons2[:, idx_neur2]])
        ntrain = np.arange(len(idx_neur1))
        ntest = np.arange(len(idx_neur1), len(idx_neur1) + len(idx_neur2))

        itrain, itest = train_test_split_idx(
            X.shape[0], train_frac=0.5, interleave=interleave
        )

        result = SVCA(
            X.T,
            ntrain=ntrain,
            ntest=ntest,
            itrain=itrain,
            itest=itest,
            n_randomized=nsvc,
            **kwargs
        )

        cov_neur[: len(result[0]), s] = result[0]
        var_neur[: len(result[0]), s] = result[1]

    if out is not None:
        np.savez(
            out,
            cov_neur=cov_neur,
            var_neur=var_neur,
            nneur=nneur,
            nsamplings=nsamplings,
            nsvc=nsvc,
            interleave=interleave,
            **kwargs
        )

    return cov_neur, var_neur


def main(path1, path2, out, nneur, nsamplings=10):
    """
    CLI for scaling_analysis.session_permuation.

    For example, run SVCA with session permutation for 10 samplings of 512 neurons:

    :code:`python -m scaling_analysis.session_permuation file1.h5 file2.h5 /out/ 512 10`

    Parameters
    ----------
    path1 : str
        Path to first dataset loaded by Experiment
    path2 : str
        Path to second dataset loaded by Experiment
    out : str
        Path to save session permutation results
    nneur : int
        Number of neurons to randomly sample
    nsamplings : int, default=10
        Number of random samplings to perform
    """

    nneur = int(nneur)
    nsamplings = int(nsamplings)

    expt1 = Experiment(path1)
    neurons1 = expt1.T_all.astype("single")
    neurons1 = neurons1[:, :10000]
    neurons1 = zscore(neurons1)
    fhz = expt1.fhz  # assuming they are same volume rates!

    expt2 = Experiment(path2)
    neurons2 = expt2.T_all.astype("single")
    neurons2 = neurons2[:, :10000]
    neurons2 = zscore(neurons2)
    del expt1, expt2

    nneur = get_nneur(min(neurons1.shape[1], neurons2.shape[1]), nneur)

    def filename(path):
        return os.path.basename(os.path.splitext(path)[0])

    out_path = os.path.join(out, filename(path1) + "_" + filename(path2))
    out_file = os.path.join(out_path, "nneur" + str(nneur) + ".npz")
    print("saving session permutation results to", out_file)
    if not os.path.exists(out):
        os.mkdir(out)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    prePCA = (
        nneur > min(neurons1.shape[0], neurons2.shape[0]) * 10
    )  # prePCA only worth it when # neurons >> # timepoints

    svca_across_sessions(
        neurons1,
        neurons2,
        nneur,
        nsamplings=nsamplings,
        out=out_file,
        interleave=int(fhz * 72),
        prePCA=prePCA,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
