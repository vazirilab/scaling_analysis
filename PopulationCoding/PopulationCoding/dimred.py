"""
Dimensionality reduction tools!
"""


from math import floor
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils.extmath import randomized_svd
import warnings

from .utils import train_test_split_idx, get_consecutive_chunks


MAX_N_AUTO_SOLVER = 300000


### DIMENSIONALITY REDUCTION METHODS


def SVCA(
    X,
    ntrain=None,
    ntest=None,
    itrain=None,
    itest=None,
    n_randomized=None,
    shuffle=False,
    flip_traintest=False,
    prePCA=True,
    **kwargs
):
    """
    Shared Variance Component Analysis (SVCA),
    originally described by Stringer et al. 2019 [1]_.
    SVCA is essentially a cross-validated canonical covariance
    analysis between two sets of neuronal populations.
    New features here include shuffling approaches and `prePCA`.

    Parameters
    ----------
    Ff : array_like of shape (N, T)
        Neural data matrix
    n_randomized : int, optional
        Number of SVCs to compute (if not None, uses randomized SVD approximation)
    ntrain : array_like, optional
        Indices of first neural subset
        (note this is not truly "training" data, as `ntrain` and `ntest` are both used
        to train and test SVCA)
    ntest : array_like, optional
        Indices of second neural subset
    itrain : array_like, optional
        Indices of training timepoints
    itest : array_like, optional
        Indices of test timepoints
    shuffle : bool, default=False
        Whether to shuffle by circularly permuting each neuron's timeseries
        (not recommended whenever # neurons > # timepoints!)
    flip_traintest: bool, default=False
        Whether to shuffle by swapping the train and test timepoints for one neural subset
        (recommended if not using the session permutation method [2]_)
    prePCA: bool, default=False
        Whether to peform PCA on each subset independently before computing SVCs. This is very useful when len(ntrain)xlen(ntest)
        covariance matrix does not fit into local memory and # neurons >> # timepoints. All PCs are kept, such that the
        resulting SVCA decomposition is mathematically equivalent to running SVCA on the original data matrix

    Returns
    -------
    sneur : ndarray
        Shared variance of each covariance component
    vneur : ndarray
        Total variance of each covariance component
    u : ndarray
        Left eigenvectors of covariance matrix between ntrain and ntest during itrain timepoints
    v : ndarray
        Right eigenvectors of covariance matrix between ntrain and ntest during itrain timepoints
    pca : None | dict
        If `prePCA=True`, return a dictionary containing the projections and principal components
        for both neural sets with the keys `train_projs`, `test_projs`, `train_vecs`, `test_vecs`

    References
    ----------
    .. [1] Stringer, C., Pachitariu, M., Steinmetz, N., Reddy, C.B., Carandini, M., & Harris, K.D.
           (2019). Spontaneous behaviors drive multidimensional, brainwide activity. Science,
           364(6437), https://doi.org/10.1126/science.aav7893.
    .. [2] Harris, K. D. (2020). Nonsense correlations in neuroscience. bioRxiv, 2020-11.
           https://doi.org/10.1101/2020.11.29.402719.
    """

    if shuffle:
        # circularly permute - DOES NOT WORK WELL FOR # NEURONS >> # TIMEPOINTS
        Xshuff = X.copy()
        for i in range(X.shape[0]):
            Xshuff[i, :] = np.roll(Xshuff[i, :], np.random.randint(X.shape[1]))
        X = Xshuff

    if ntrain is None or ntest is None:
        ntrain, ntest = train_test_split_idx(X.shape[0])

    if itrain is None or itest is None:
        itrain, itest = train_test_split_idx(X.shape[1])

    if flip_traintest:
        # swapping train and test timepoints for one set of neurons
        # thus removing temporal alignment between ntrain and ntest
        if len(itrain) != len(itest):
            itrain = itrain[: min(len(itrain), len(itest))]
            itest = itest[: min(len(itrain), len(itest))]

        # shuffle interleaved sequences to remove any global long-term trends
        trains = get_consecutive_chunks(itrain)
        tests = get_consecutive_chunks(itest)
        trains = [trains[i] for i in np.random.permutation(len(trains))]
        tests = [tests[i] for i in np.random.permutation(len(tests))]

        trains = [trains[i] for i in np.random.permutation(len(trains))]
        tests = [tests[i] for i in np.random.permutation(len(tests))]

        itrain = [x for y in trains for x in y]
        itest = [x for y in tests for x in y]

        itrain1 = itrain
        itest1 = itest
        itrain2 = itest
        itest2 = itrain
    else:
        itrain1 = itrain
        itest1 = itest
        itrain2 = itrain
        itest2 = itest

    if prePCA:
        if min(len(ntrain), len(ntest)) > MAX_N_AUTO_SOLVER:
            # an empirical observation
            solver = "randomized"
        else:
            solver = "auto"

        pca = PCA(
            n_components=min(X.shape[1], len(ntrain)),
            svd_solver=solver,
            random_state=None,
        )
        trainPCs = pca.fit_transform(X[ntrain, :].T).T
        train_pc_vecs = pca.components_

        pca = PCA(
            n_components=min(X.shape[1], len(ntrain)),
            svd_solver=solver,
            random_state=None,
        )
        testPCs = pca.fit_transform(X[ntest, :].T).T
        test_pc_vecs = pca.components_

        cov = trainPCs[:, itrain1] @ testPCs[:, itrain2].T
    else:
        cov = X[ntrain, :][:, itrain1] @ X[ntest, :][:, itrain2].T

    if n_randomized is None:
        # Perform full SVD
        u, s, vt = np.linalg.svd(cov)
    else:
        # Approximate SVD with randomized SVD
        u, s, vt = randomized_svd(cov, n_components=n_randomized, **kwargs)

    if u.shape[1] != vt.shape[0]:
        warnings.warn("SVCA: u and v different sizes!")
        print(u.T.shape)
        print(vt.shape)

        nc = min(u.shape[1], vt.shape[0])
        u = u[:, :nc]
        vt = vt[:nc, :]

    # Compute covariance and variance on held-out testing timepoints
    if prePCA:
        s1 = u.T @ trainPCs[:, itest1]
        s2 = vt @ testPCs[:, itest2]
    else:
        s1 = u.T @ X[ntrain, :][:, itest1]
        s2 = vt @ X[ntest, :][:, itest2]

    sneur = np.sum(s1 * s2, axis=1)
    varneur = np.sum(s1**2 + s2**2, axis=1) / 2

    if prePCA:
        pca = {
            "train_projs": trainPCs.T,
            "test_projs": testPCs.T,
            "train_vecs": train_pc_vecs,
            "test_vecs": test_pc_vecs,
        }
    else:
        pca = None

    return sneur, varneur, u, vt.T, pca


def lda(data, labels, d=None, classes=None):
    """
    Linear discriminant analysis (LDA)

    LDA is a generalization of Fisher's linear discriminant that finds a projection
    that minimizes the Fisher-Rao discriminant among multiple classes.
    This technique requires continuous input data and a priori known classes.

    ASSUMPTIONS:

    - For those familiar with MANOVA, the same assumptions apply here.
    - Independent variables are all normal across same levels of grouping variables (multivariate normality).
    - Covariances are equal across classes (homoscedasticity).
    - Samples are chosen independently.
    - Predictive power may decrease with increased correlations among predictor variables (multicollinearity).

    For a more thorough overview of LDA, consult e.g. McLachlan, 2005 [1]_.

    Parameters
    ----------
    data : array_like of shape (n, m)
        nxm data matrix, where columns represent m features and rows represent n samples
    labels : array_like of shape (n, )
        Vector of class labels for the given data
    d : int, optional
        Desired dimensionality after projection. d <= cardinality(labels)-1
        if None, d = cardinality(labels)-1
    classes : array_like of shape (cardinality(labels),), optional
        Class names (as in labels)

    Returns
    -------
    W : array_like of shape (m, d)
        Projection matrix to reduced dimensional space, spanned by the top
        cardinality(labels)-1 generalized eigenvectors of S_b and S_w
    proj_data : array_like of shape (n, d)
        Data after projection under W
    vals : array_like of shape (d, )
        Eigenvalues corresponding to eigenvectors in columns of W
    mu_c : array_like of shape (d, cardinality(classes))
        Matrix where each column is the class mean after projection
    S_c : array_like of shape (d, d, cardinality(classes))
        Covariance matrices after projection

    References
    ----------
    .. [1] McLachlan, G. J. (2005). Discriminant analysis and statistical pattern recognition.
           John Wiley & Sons.
    """

    # get shapes
    n, m = data.shape
    assert n == labels.shape[0]

    classes = np.unique(labels)
    c = classes.shape[0]

    if d is None:
        d = c - 1

    # compute overall means and variances
    mu = np.mean(data, axis=0).reshape((1, data.shape[1]))  # overall sample mean

    S_b = np.zeros(m)
    S_w = np.zeros(m)

    for ci in range(c):
        idx = np.asarray(np.where(labels == classes[ci]))
        n_i = idx.shape[1]
        curr_data = np.squeeze(data[idx, :])
        mu_i = np.mean(curr_data, axis=0).reshape(
            (1, data.shape[1])
        )  # class sample mean
        dev = curr_data - np.repeat(mu_i, n_i, axis=0)  # deviation from mean

        S_b = S_b + n_i * (mu_i - mu) * (mu_i - mu).T  # between class variation
        S_w = S_w + np.dot(dev.T, dev)

    # train the projection matrix
    vals, vecs = eigh(S_b, S_w)  # find generalized eigendecomposition
    idx = np.flipud(np.argsort(vals))  # find top eigenvalues

    W = vecs[:, idx[0:d]]  # retrieve projection matrix
    vals = vals[idx[0:d]]  # retrieve corresponding eigenvalues

    # compute projections and resulting means / covariances
    proj_data = data.dot(W)

    mu_c = np.zeros((d, c))
    S_c = np.zeros((d, d, c))

    for ci in range(c):
        idx = np.where(labels == classes[ci])
        mu_c[:, ci] = np.mean(np.squeeze(proj_data[idx, :]), axis=0)
        S_c[:, :, ci] = np.cov(np.squeeze(proj_data[idx, :]).T)

    return W, proj_data, vals, mu_c, S_c


### OTHER USEFUL TOOLS


def optimize_latent_dim(X, y, model, dims, cv=3, scoring="neg_mean_squared_error"):
    """
    Identify the optimal latent dimensionality via cross validation.

    Parameters
    ----------
    X : array_like of shape (n_observations, p)
        Data matrix
    y : array_like of shape (n_observations, q)
        Targets
    model : estimator, must take an n_components argument
        An sklearn model to try out
    dims : array_like
        List of integer dimensionalities to try out
    cv : int, default=3
        Cross validation fold, can be int or KFold, etc.
    scoring : str | callable, default="neg_mean_squared_error"
        Metric to use to assess prediction quality in latent space

    Returns
    -------
    d_opt : int
        Latent dimensionality that maximizes cross-validated score
    scores : array_like of shape (len(dims),)
        Cross-validated score at each dimensionality
    """

    scores = []

    for d in dims:
        model.set_params(n_components=d)

        score = cross_val_score(
            model, X, y, cv=KFold(n_splits=cv, shuffle=True), scoring=scoring
        ).mean()

        scores.append(score)

    d_opt = dims[np.argmax(scores)]

    return d_opt, scores


def estimate_id_twonn(X, plot=False, X_is_dist=False):
    """
    TWO-NN method for estimating intrinsic dimensionality
    as described by Facco et al. 2017 [1]_.
    This implementation is taken from
    https://github.com/jmmanley/two-nn-dimensionality-estimator.

    Parameters
    ----------
    X : array_like of shape (N, p)
        Matrix of N p-dimensional samples (when X_is_dist=False)
    plot : bool, default=False
        Boolean flag of whether to plot fit
    X_is_dist : bool, default=False
        Boolean flag of whether X is an (N, N) distance matrix instead

    Returns
    -------
    d : TWO-NN estimate of intrinsic dimensionality

    References
    ----------
    .. [1] Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the
           intrinsic dimension of datasets by a minimal neighborhood information.
           Scientific reports, 7(1), 12140. https://doi.org/10.1038/s41598-017-11873-y.
    """

    N = X.shape[0]

    if X_is_dist:
        dist = X
    else:
        # COMPUTE PAIRWISE DISTANCES FOR EACH POINT IN THE DATASET
        dist = squareform(pdist(X, metric="euclidean"))

    # FOR EACH POINT, COMPUTE mu_i = r_2 / r_1,
    # where r_1 and r_2 are first and second shortest distances
    mu = np.zeros(N)

    for i in range(N):
        sort_idx = np.argsort(dist[i, :])
        mu[i] = dist[i, sort_idx[2]] / dist[i, sort_idx[1]]

    # COMPUTE EMPIRICAL CUMULATE
    sort_idx = np.argsort(mu)
    Femp = np.arange(N) / N

    # FIT (log(mu_i), -log(1-F(mu_i))) WITH A STRAIGHT LINE THROUGH ORIGIN
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu[sort_idx]).reshape(-1, 1), -np.log(1 - Femp).reshape(-1, 1))

    d = lr.coef_[0][0]  # extract slope

    if plot:
        # PLOT FIT THAT ESTIMATES INTRINSIC DIMENSION

        s = plt.scatter(np.log(mu[sort_idx]), -np.log(1 - Femp), c="r", label="data")
        p = plt.plot(
            np.log(mu[sort_idx]),
            lr.predict(np.log(mu[sort_idx]).reshape(-1, 1)),
            c="k",
            label="linear fit",
        )
        plt.xlabel("$\log(\mu_i)$")
        plt.ylabel("$-\log(1-F_{emp}(\mu_i))$")
        plt.title("ID = " + str(np.round(d, 3)))
        plt.legend()

    return d


def plot_component_images(components, nplot=9, **kwargs):
    """
    After applying matrix decomposition to a set of images or other 2D data,
    plots the first `nplot` components.

    Parameters
    ----------
    components : array_like of shape (N, W, ncomp)
        Matrix of weights to be plotted
    nplot : int, default=9
        Number of components to plot
    kwargs : passed to `plt.subplots`

    Returns
    -------
    fig : Figure
    """
    row = int(np.ceil(np.sqrt(nplot)))
    col = int(np.ceil(nplot / row))

    fig, axs = plt.subplots(row, col, **kwargs)

    for i in range(row * col):
        cmax = max(np.max(components[:, :, i]), np.abs(np.min(components[:, :, i])))

        if i < nplot:
            axs[floor(i / row), i % col].matshow(
                components[:, :, i], cmap="bwr", clim=[-cmax, cmax]
            )
        axs[floor(i / row), i % col].axis("off")
    plt.tight_layout()

    return fig
