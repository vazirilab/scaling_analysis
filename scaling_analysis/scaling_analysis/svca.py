"""
Functions for running SVCA on various samplings
of neurons from large-scale neural recordings.
"""

import datetime
import numpy as np
import os
from scipy.stats import zscore
import sys
from tqdm import tqdm

from PopulationCoding.dimred import SVCA
from PopulationCoding.utils import bin2d, train_test_split_idx
from scaling_analysis.experiment import Experiment
from scaling_analysis.utils import get_nneur


def run_SVCA_partition(
    X,
    n,
    checkerboard=None,
    centers=None,
    FOV=None,
    n_randomized=4096,
    t_train_frac=0.5,
    shuffle=False,
    prePCA=True,
    interleave=400,
    **kwargs
):
    """
    Samples a given number of neurons and performs SVCA.

    Parameters
    ----------
    X : array_like of shape (T, N)
        Neuronal timeseries
    n : int
        Number of neurons to sample
    checkerboard : float, optional
        Size of lateral "checkerboarding" pattern used to split neurons into two sets
    centers : array_like of shape (d, N), optional
       Position of neurons. First two dimensions are used for checkerboarding, i.e. d ≥ 2
    FOV : str "expand" | float, optional
        If None, neurons are sampled randomly.
        If "expand", neurons are sampled in order of their distance from the center of the volume.
        If float, neurons are sampled from a randomly placed lateral FOV of given size
    n_randomized : int, default=4096
        Number of SVCs to compute (if not None, uses randomized SVD approximation)
    t_train_frac : float, default=0.5
        Fraction of timepoints to use as training data
    shuffle : bool, default=False
        Whether to shuffle by swapping the train and test timepoints for one neural subset
        (recommended if not using the session permutation method)
    prePCA : bool, optional, default=False
        Whether to peform PCA on each subset independently before computing SVCs
        (see articularly useful when N >> T, so this is ignored and PCA is not performed if N ≤ T * 10
    interleave : int, default=400
        Number of timepoints which are chunked together before randomly assigned chunks to train or test.
        If interleave=0, each timepoint is independently assigned to train or test
        (not recommended due to autocorrelations)
    kwargs : Passed onto PopulationCoding.dimred.SVCA

    Returns
    -------
    sneur : ndarray
        Reliable covariance of each SVC on held-out testing timepoints
    varneur : ndarray
        Reliable variance of each SVC on held-out testing timepoints
    u : ndarray
        SVC vectors for the first neural subset, ntrain
    v : ndarray
        SVC vectors for the second neural subset, ntest
    ntrain : ndarray
        Indices of neurons in the first neural subset
    ntest : ndarray
        Indices of neurons in the second neural subset
    itrain : ndarray
        Indices of training timepoints
    itest : ndarray
        Indices of testing timepoints
    pca : None | dict
        If prePCA, return a dictionary containing the projections and principal components for both
        neural sets with the keys `train_projs`, `test_projs`, `train_vecs`, `test_vecs`

    See Also
    --------
    PopulationCoding.dimred.SVCA :
        `A basic implementation of SVCA <https://scaling-analysis.readthedocs.io/projects/PopulationCoding/en/latest/PopulationCoding.html#PopulationCoding.dimred.SVCA>`_
        which is utilized here.
    """

    ### SAMPLE NEURONS

    # determine FOV from which to sample neurons
    if FOV is not None:  # if FOV is None, sample from entire volume
        if centers is None:
            raise ValueError("if FOV is not None, must input centers as kwarg!")

        if FOV == "expand":
            # sample from an expanding FOV
            # pick the smallest FOV that contains n neurons!
            idx = expanding_FOV(centers, n)
        else:
            # pick neurons in a lateral region of size FOV
            idx = random_lateral_FOV(centers, FOV)

        X = X[:, idx]
        centers = centers[:, idx]

        if n > len(idx):
            print("WARNING: n neurons > # in FOV")

    # split neurons into two sets
    if checkerboard is not None:
        # split neurons into two sets in a checkerboard pattern of size checkerboard
        # prevents neurons in separate sets from being on top of each other, in case of axial crosstalk
        if centers is None:
            raise ValueError(
                "if checkerboard is not None, must input centers as kwarg!"
            )

        XX = X.T
        idx1, idx2 = checkerboard_centers(centers, checkerboard)

        # randomly sample n/2 neurons from each checkerboard set
        ntrain = idx1[np.random.permutation(len(idx1))[: int(n / 2)]]
        ntest = idx2[np.random.permutation(len(idx2))[: int(n / 2)]]

    else:
        # randomly split neurons into two sets regardless of location
        XX = X[:, np.random.permutation(X.shape[1])[:n]].T
        ntrain, ntest = train_test_split_idx(XX.shape[0])

    # split training and testing timepoints
    itrain, itest = train_test_split_idx(
        XX.shape[1], train_frac=t_train_frac, interleave=interleave
    )

    ### RUN SVCA

    n_randomized = min(len(ntrain), len(ntest), n_randomized, len(itrain), len(itest))

    if XX.shape[1] * 10 < XX.shape[0]:
        # prePCA only particularly useful when N >> T
        prePCA = False

    sneur, varneur, u, v, pca = SVCA(
        XX,
        ntrain,
        ntest,
        itrain,
        itest,
        n_randomized=n_randomized,
        flip_traintest=shuffle,
        prePCA=prePCA,
        **kwargs
    )

    if FOV is not None:
        # convert FOV-specific indices back to original indices from X
        ntrain = idx[ntrain]
        ntest = idx[ntest]

    return sneur, varneur, u, v, ntrain, ntest, itrain, itest, pca


def expanding_FOV(centers, n):
    """Return the indices of the n positions in centers closest to the middle of FOV"""
    center = (
        (np.max(centers[0, :]) - np.min(centers[0, :])) / 2,
        (np.max(centers[1, :]) - np.min(centers[1, :])) / 2,
    )

    dists = np.sum((centers[:2, :] - np.asarray(center).reshape(-1, 1)) ** 2, axis=0)

    idx = np.argsort(dists)[:n]
    return idx


def random_lateral_FOV(centers, FOV):
    """
    Select neurons within a randomly placed field of view with a lateral size FOV.

    Parameters
    ----------
    centers : array_like of shape (d, N)
       Position of neurons. First two dimensions are used for FOV placement, i.e. d ≥ 2
    FOV : float
       Lateral size of FOV, in units of centers

    Returns
    -------
    idx : ndarray
       Indices of neurons within a randomly placed lateral region
       of size FOV
    """

    if np.min(centers[0, :]) >= np.max(centers[0, :]) - FOV:
        print("WARNING: REQUESTED X FOV IS TOO BIG")
    if np.min(centers[1:]) >= np.max(centers[1, :]) - FOV:
        print("WARNING: REQUESTED Y FOV IS TOO BIG")

    # Randomly place center of FOV
    x = np.random.randint(
        np.min(centers[0, :]),
        np.max([np.max(centers[0, :]) - FOV, np.min(centers[0, :]) + 1]),
    )
    y = np.random.randint(
        np.min(centers[1, :]),
        np.max([np.max(centers[1, :]) - FOV, np.min(centers[1, :]) + 1]),
    )

    idx = np.where(
        np.all(
            [
                centers[0, :] > x,
                centers[0, :] < x + FOV,
                centers[1, :] > y,
                centers[1, :] < y + FOV,
            ],
            axis=0,
        )
    )[0]
    return idx


def checkerboard_centers(centers, checkerboard):
    """
    Returns indices of variables split into two sets according to a
    checkerboard pattern based on the positions located in centers.

    Parameters
    ----------
    centers : array_like of shape (d, N)
       Position of neurons. First two dimensions are used for checkerboarding, i.e. d ≥ 2
    checkerboard : float
        Size of square in checkerboard

    Returns
    -------
    idx1 : ndarray
        Indices of first set
    idx2: ndarray
        Indices of second set
    """

    nbin_x = int(
        np.round((np.max(centers[0, :]) - np.min(centers[0, :])) / checkerboard)
    )

    if nbin_x < 2:
        nbin_x = 2

    nbin_y = int(
        np.round((np.max(centers[1, :]) - np.min(centers[1, :])) / checkerboard)
    )

    if nbin_y < 2:
        nbin_y = 2

    bin_x = np.linspace(
        np.min(centers[0, :]) - 1, np.max(centers[0, :] + 1), num=nbin_x + 1
    )
    bin_y = np.linspace(
        np.min(centers[1, :]) - 1, np.max(centers[1, :] + 1), num=nbin_y + 1
    )

    idx1 = []
    idx2 = []

    def is_odd(num):
        return num & 0x1

    for i in range(len(bin_x) - 1):
        for j in range(len(bin_y) - 1):
            ixx = np.where(
                np.all(
                    [
                        centers[0, :] > bin_x[i],
                        centers[0, :] < bin_x[i + 1],
                        centers[1, :] > bin_y[j],
                        centers[1, :] < bin_y[j + 1],
                    ],
                    axis=0,
                )
            )[0]
            if is_odd(i) == is_odd(j):
                idx1.append(ixx)
            else:
                idx2.append(ixx)

    idx1 = np.asarray([x for y in idx1 for x in y])
    idx2 = np.asarray([x for y in idx2 for x in y])

    return idx1, idx2


def main(
    path,
    nneur,
    nsamplings=10,
    checkerboard=250,
    FOV=None,
    n_randomized=4096,
    shuffle=False,
    save_indiv=True,
    tbin=1,
):
    """
    CLI for scaling_analysis.svca.

    In particular, this script saves the SVC coefficients associated
    with each individual sampling, something that
    scaling_analysis.predict.predict_from_behavior
    does not do.

    For example, save SVCs for 10 samplings of 512 neurons:

    :code:`python -m scaling_analysis.svca /path/to/file.h5 512 10`

    Parameters
    ----------
    path : str
        Path to dataset loaded by Experiment
    nneur : int
        Number of neurons to randomly sample
        checkerboard : float, optional
        Size of lateral "checkerboarding" pattern used to split neurons into two sets
    nsamplings : int, default=1
        Number of random samplings to perform
    checkerboard : float, optional
        Size of lateral "checkerboarding" pattern used to split neurons into two sets
    FOV : str "expand" | float, optional
        If None, neurons are sampled randomly.
        If "expand", neurons are sampled in order of their distance from the center of the volume.
        If float, neurons are sampled from a randomly placed lateral FOV of given size
    n_randomized : int, default=4096
        Number of SVCs to compute (if not None, uses randomized SVD approximation)
    shuffle : bool, default=False
        Whether to shuffle by swapping the train and test timepoints for one neural subset
        (recommended if not using the session permutation method)
    save_indiv : bool, default=True
        Whether or not to save the SVC coefficients for each individual round of sampling
    tbin : int, default=1
        Number of frames which are binned together
    """

    ### PARSE INPUTS
    nneur = int(nneur)
    nsamplings = int(nsamplings)
    checkerboard = float(checkerboard)
    shuffle = bool(int(shuffle))
    if FOV == "0":
        FOV = None
    elif FOV is not None and FOV != "expand":
        FOV = float(FOV)
    tbin = int(tbin)
    save_indiv = bool(int(save_indiv))
    n_randomized = int(n_randomized)

    ### LOAD EXPERIMENT
    expt = Experiment(path)

    neurons = zscore(expt.T_all.astype("single"))
    nneur = get_nneur(neurons.shape[1], nneur)

    if tbin > 1:
        # bin data
        neurons = bin2d(neurons, tbin)

    out = os.path.join(expt.out, "svca")
    if shuffle:
        out += "shuffled"
    print("saving SVCA results to", out)
    if not os.path.exists(out):
        os.mkdir(out)

    prePCA = nneur > 131072

    if save_indiv and not os.path.exists(os.path.join(out, "indiv")):
        os.mkdir(os.path.join(out, "indiv"))

    ### RUN MANY SVCA SAMPLINGS AND SAVE RESULTS

    sneur = np.zeros((nsamplings, n_randomized)) + np.nan
    varneur = np.zeros((nsamplings, n_randomized)) + np.nan

    for i in tqdm(range(nsamplings)):

        result = run_SVCA_partition(
            neurons,
            nneur,
            checkerboard=checkerboard,
            centers=expt.centers,
            FOV=FOV,
            n_randomized=min(n_randomized, nneur),
            t_train_frac=0.5,
            shuffle=shuffle,
            interleave=int(72 * expt.fhz),
            prePCA=prePCA,
        )

        curr_sneur, curr_varneur, u, v, ntrain, ntest, itrain, itest, pca = result

        if pca is not None:
            # convert u and v from PC space to neuron space
            u = pca["train_vecs"].T @ u
            v = pca["test_vecs"].T @ v

        sneur[i, : len(curr_sneur)] = curr_sneur
        varneur[i, : len(curr_varneur)] = curr_varneur

        if save_indiv:
            now = datetime.datetime.now()
            date = now.strftime("%Y%m%d%H%M%S%f")
            indiv_file = os.path.join(out, "indiv", str(nneur) + "_" + date)
            np.save(indiv_file + "_sneur.npy", curr_sneur)
            np.save(indiv_file + "_varneur.npy", curr_varneur)
            np.save(indiv_file + "_u.npy", u)
            np.save(indiv_file + "_v.npy", v)
            np.save(indiv_file + "_ntrain.npy", ntrain)
            np.save(indiv_file + "_ntest.npy", ntest)
            np.save(indiv_file + "_itrain.npy", itrain)
            np.save(indiv_file + "_itest.npy", itest)

    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d%H%M%S%f")

    np.savez(
        os.path.join(out, str(nneur) + "_" + date),
        sneur=sneur,
        varneur=varneur,
        n_neurons=nneur,
        nsamplings=nsamplings,
        path=path,
        out=out,
        checkerboard=checkerboard,
        FOV=FOV,
        shuffle=shuffle,
        prePCA=prePCA,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
