"""
Functions for running PCA on various numbers of randomly sampled neurons.
"""


import os
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
import sys
from tqdm import tqdm

from PopulationCoding.utils import train_test_split_idx
from scaling_analysis.experiment import Experiment
from scaling_analysis.utils import get_nneur


MAX_N_AUTO_SOLVER = 65536


def run_pca_nneur(
    neurons, nneur, nsamplings=10, npc=4096, out=None, t_train_frac=1, interleave=400
):
    """
    Samples a given number of neurons and performs PCA.

    Parameters
    ----------
    neurons : array_like of shape (T, N)
        Neuronal timeseries
    nneur : int
        Number of neurons to sample
    nsamplings : int, default=10
       Number of samplings and rounds of PCA to perform
    npc : int, default=4096
        Number of PCs to compute
    out : str, optional
        Path to optionally save results
    t_train_frac: float, default=1
        Fraction of timepoints to use
    interleave : int, default=400
        Number of timepoints which are chunked together before randomly assigned chunks to train or test.
        If interleave=0, each timepoint is independently assigned to train or test (not recommended due to autocorrelations).
        interleave has no effect when t_train_frac=1

    Returns
    -------
    var_neur : ndarray of shape (npc, nsamplings)
        Variance explained by each PC
    """

    npc = min([npc, nneur, neurons.shape[0], neurons.shape[1]])

    var_neur = np.zeros((npc, nsamplings))

    for s in tqdm(range(nsamplings)):

        if t_train_frac > 0.999:  # just use all
            itrain = np.arange(neurons.shape[0])
        else:
            itrain, _ = train_test_split_idx(
                neurons.shape[0], train_frac=t_train_frac, interleave=interleave
            )

        ntrain = np.random.permutation(neurons.shape[1])[:nneur]

        if nneur >= MAX_N_AUTO_SOLVER:
            pca = PCA(
                n_components=min([npc, len(itrain), len(ntrain)]),
                svd_solver="randomized",
                random_state=None,
            )
        else:
            pca = PCA(n_components=min([npc, len(itrain), len(ntrain)]))

        if out is not None and s == 0:
            # Save one example set of projections, components
            projs1 = pca.fit_transform(neurons[:, ntrain][itrain, :])
            u1 = pca.components_
            ntrain1 = ntrain
            itrain1 = itrain
        else:
            pca.fit(neurons[:, ntrain][itrain, :])

        var_neur[: len(pca.explained_variance_), s] = pca.explained_variance_

    if out is not None:
        np.savez(
            os.path.join(out, "pca_nneur" + str(nneur)),
            var_neur=var_neur,
            nneur=nneur,
            projs1=projs1,
            u1=u1,
            ntrain1=ntrain1,
            itrain1=itrain1,
        )

    return var_neur


def main(path, nneur, nsamplings=1):
    """
    CLI for scaling_analysis.pca.

    For example, run 1 sampling of PCA on 512 neurons:

    :code:`python -m scaling_analysis.pca /path/to/file.h5 512 1`

    Parameters
    ----------
    path : str
        Path to dataset loaded by Experiment
    nneur : int
        Number of neurons to randomly sample
    nsamplings : int, default=1
        Number of random samplings to perform
    """

    nneur = int(nneur)
    nsamplings = int(nsamplings)

    expt = Experiment(path)

    neurons = expt.T_all.astype("single")
    neurons = zscore(neurons)

    nneur = get_nneur(neurons.shape[1], nneur)

    out = os.path.join(expt.out, "pca_vs_nneur")
    print("saving PCA results to", out)
    if not os.path.exists(out):
        os.mkdir(out)

    run_pca_nneur(neurons, nneur, nsamplings=nsamplings, out=out)


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
