"""
Analysis of the spatial properties of SVCs.
"""

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


def hoyer_sparsity(X):
    """Computes the Hoyer sparsity of the column vectors of X."""
    k = np.sqrt(X.shape[0])
    return (k - np.linalg.norm(X, ord=1, axis=0) / np.linalg.norm(X, ord=2, axis=0)) / (
        k - 1
    )


def gini_index(X):
    """Computes the Gini index of the column vectors of X."""
    N = X.shape[0]
    return 1 - 2 * np.sum(
        np.sort(X, axis=0)
        / np.linalg.norm(X, ord=1, axis=0)
        * (N - np.arange(N) + 1.5).reshape(-1, 1)
        / N,
        axis=0,
    )


def local_homogeneity(
    values,
    centers,
    dist_threshes=np.arange(20, 501, 20),
    ntodo=None,
):
    """
    Computes the local homogeneity of SVCs, which defines the average % of neighbors
    (within some radius) of a participating neuron that are *also* participating in the same SVC.

    Parameters
    ----------
    values : array_like of length N
        Binary vector indicating which neurons are participating in that SVC
    centers : array_like of shape (d, N)
        Position of neurons
    dist_threshes : array_like, default=np.arange(20, 501, 20)
        Radii at which to evaluate local homogeneity
    ntodo : int, optional
        Number of participating neurons whose local homogeneity should be evaluated.
        If not provided, :code:`ntodo = min(ntodo, np.sum(values))`

    Returns
    -------
    local_homogeneity : ndarary of shape (ntodo, len(dist_threshes))
    """

    # Determine number of participating neurons
    n_participating = int(np.sum(values))

    if n_participating == 0:
        return np.zeros((1, len(dist_threshes)))

    # Determine number of neurons to evaluate local homogeneity for
    if ntodo is not None:
        ntodo = min(ntodo, n_participating)
        idx_todo = np.where(values)[0]
        np.random.shuffle(idx_todo)
        idx_todo = idx_todo[:ntodo]
    else:
        idx_todo = np.where(values)[0]
        ntodo = n_participating

    participating_centers = centers[:, idx_todo]

    # Compute pairwise distances between participating neurons and all other neurons
    participating_dists = cdist(participating_centers.T, centers.T)

    # Ignore distance between neuron and itself
    for i in range(len(idx_todo)):
        participating_dists[i, idx_todo[i]] = np.nan

    # Initialize output matrix of local homogeneity
    indiv_homogeneity = np.zeros((participating_dists.shape[0], len(dist_threshes)))

    # Compute local homogeneity for each individual neuron
    for i in range(participating_dists.shape[0]):
        # Loop through distance thresholds
        curr_dists = participating_dists[i, :]

        for j in range(len(dist_threshes)):
            # Find neighbors within distance threshold
            neighbors = np.where(curr_dists < dist_threshes[j])[0]

            # Compute local homogeneity with a given distance threshold
            indiv_homogeneity[i, j] = np.mean(values[neighbors])

    return indiv_homogeneity


def reorder_svcs(svc, neuron_idx):
    """
    Reorder SVCs according to neuron_idx such that it aligns with the
    ordering in the original data. This is necessary because run_SVCA_partition
    randomly samples neurons, so ntrain and ntest can be random permutations
    of the neuron indices which are distinct across samplings.

    Parameters
    ----------
    svc : array_like of shape (nsamplings, nneur, nsvc)
        SVCs across many samplings to be reordered
    neuron_idx : array_like of shape (nsamplings, #nneur)
        Indices of neurons in each sampling with respect to original data matrix.
        Created by concatenating ntrain and ntest

    Returns
    -------
    svc_ordered : ndarray of shape (nsamplings, nneur, nsvc)
        Reordered SVCs, where svc_ordered[:,i,:] contains all the coefficients
        for the same neuron, i.e. the i-th neuron in the original data matrix
    """

    # Determine maximum number of neurons
    max_neurons = np.max(neuron_idx)

    # Preallocate array for reordered SVCs
    svc_ordered = np.empty(
        (svc.shape[0], max_neurons + 1, svc.shape[2]), dtype=np.float32
    )
    svc_ordered.fill(np.nan)

    # Assign values from svc to svc_ordered based on neuron_idx
    print("Reordering shuffled SVCs into original neuron order")
    for sample_i in tqdm(range(svc.shape[0])):
        svc_ordered[sample_i, neuron_idx[sample_i, :], :] = svc[sample_i, :, :].copy()

    return svc_ordered
