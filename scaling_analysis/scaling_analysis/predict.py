"""
Functions for running SVCA and predicting neural SVCs
from *instantaneous* behavior PCs.
"""

from joblib import Parallel, delayed
import numpy as np
import os
from scipy.stats import zscore
from sklearn.neural_network import MLPRegressor
import sys
from tqdm import tqdm

from PopulationCoding.predict import canonical_cov
from scaling_analysis.experiment import Experiment
from scaling_analysis.svca import run_SVCA_partition
from scaling_analysis.utils import correct_lag, get_nneur, DEFAULT_N_JOBS


def predict_from_behavior(
    neurons,
    nneur,
    centers,
    motion,
    ranks,
    nsamplings=10,
    lag=0,
    lams=[
        1e-3,
        5e-3,
        0.01,
        0.05,
        0.1,
        0.15,
        0.3,
    ],
    nsvc=4096,
    nsvc_predict=128,
    npc_behavior=256,
    out=None,
    checkerboard=500,
    FOV=None,
    prePCA=True,
    name_suffix="",
    MLPregressor=True,
    **kwargs
):
    """
    Samples a given number of neurons, performs SVCA, and predicts
    neural SVCs from a set of motion variables.

    Parameters
    ----------
    neurons : array_like of shape (T, N)
        Neuronal timeseries
    nneur : int
        Number of neurons to sample
    centers : array_like of shape (d, N)
       Position of neurons. First two dimensions are used for checkerboarding, i.e. d ≥ 2
    motion : array_like of shape (T, M)
       Set of motion (or other) variables used to predict neurons
    ranks : array_like
       List of ranks to use in reduced rank regression
    nsamplings : int, default=10
       Number of samplings and rounds of SVCA/prediction to perform
    lag : int, default=0
       Desired lag between neurons and motion (positive indicates behavior precedes neurons)
    lams : array_like, default=[1e-3,5e-3,0.01,0.05,0.1,0.15,0.3]
       List of regularization parameters to use in reduced rank regression
    nsvc : int, default=4096
        Mumber of SVCs to compute (if not None, uses randomized SVD approximation).
        i.e., n_randomized in PopulationCoding.dimred.SVCA
    nsvc_predict : int, default=128
        Number of SVCs to predict from motion
    npc_behavior : int, default=256
        Number of behavior PCs to use for predicting neural SVCs
    out : str, optional
        Path to optionally save results
    checkerboard : float, optional
        Size of lateral "checkerboarding" pattern used to split neurons into two sets
    FOV : str "expand" | float, optional
        If None, neurons are sampled randomly.
        If "expand", neurons are sampled in order of their distance from the center of the volume.
        If float, neurons are sampled from a randomly placed lateral FOV of given size
    prePCA : bool, default=False
        Whether to peform PCA on each subset independently before computing SVCs (see PopulationCoding.dimred.SVCA).
        Particularly useful when N >> T, so this is ignored and PCA is not performed if N ≤ T * 10
    name_suffix : str, optional
        Suffix to append to end of file name
    MLPregressor : bool, default=True
        Whether or not to additionally predict SVCs from motion using a multilayer
        perception with hidden layer of size nranks
    kwargs : passed to scaling_analysis.svca.run_SVCA_partition

    Returns
    -------
    cov_neur : ndarray of shape (nsvc, nsamplings)
        Reliable covariance of each SVC on held-out testing timepoints, across samplings
    var_neur : ndarray of shape (nsvc, nsamplings)
        Reliable variance of each SVC on held-out testing timepoints, across samplings
    cov_res_beh : ndarray of shape (nsvc, nsamplings)
        Residual SVC covariance after subtracting predictions from motion, across samplings
    cov_res_beh_mlp : ndarray of shape (nsvc, nsamplings)
        Residual SVC covariance after subtracting predictions from motion (with MLP), across samplings.
        If MLPregressor=false, all NaNs
    actual_nneurs : ndarray of shape (nsamplings,)
        Actual number of neurons used, across samplings, reported as min(len(ntrain), len(ntest))
    u : ndarray
        SVC vectors for the first neural subset, ntrain, from a single example sampling
    v : ndarray
        SVC vectors for the second neural subset, ntest, from a single example sampling
    ntrain : ndarray
        Indices of neurons in the first neural subset, from a single example sampling
    ntest : ndarray
        Indices of neurons in the second neural subset, from a single example sampling
    itrain : ndarray
        Indices of training timepoints, from a single example sampling
    itest : ndarray
        Indices of testing timepoints, from a single example sampling
    pca : None | dict
        If prePCA, return a dictionary containing the projections and principal components for both neural sets
        with the keys train_projs, test_projs, train_vecs, test_vecs
        from a single example sampling
    """

    # Initialize variables
    cov_neur = np.zeros((nsvc, nsamplings)) + np.nan
    var_neur = np.zeros((nsvc, nsamplings)) + np.nan
    cov_res_beh = np.zeros((nsvc_predict, len(ranks), len(lams), nsamplings)) + np.nan
    cov_res_beh_mlp = np.zeros((nsvc_predict, len(ranks), nsamplings)) + np.nan
    actual_nneurs = np.zeros((nsamplings,)) + np.nan

    name = (
        "nneur"
        + str(nneur)
        + "_prePCA"
        + str(prePCA)
        + "_check"
        + str(checkerboard)
        + "_lag"
        + str(lag)
        + "_nsvc"
        + str(nsvc_predict)
        + name_suffix
    )

    if FOV is not None:
        name = name + "_FOV" + str(FOV)

    if MLPregressor:
        name += "_MLP"

    if out is not None:
        file = os.path.join(out, name)
        if os.path.exists(file):
            return

    neurons, motion = correct_lag(lag, neurons, motion)
    motion = motion[:, :npc_behavior]

    # Run many samplings!
    for s in tqdm(range(nsamplings)):

        ### COMPUTE SVCS
        sneur, varneur, u, v, ntrain, ntest, itrain, itest, pca = run_SVCA_partition(
            neurons,
            nneur,
            checkerboard=checkerboard,
            centers=centers,
            FOV=FOV,
            n_randomized=nsvc,
            prePCA=prePCA,
            **kwargs
        )

        cov_neur[: len(sneur[:nsvc]), s] = sneur[:nsvc]
        var_neur[: len(sneur[:nsvc]), s] = varneur[:nsvc]

        if pca is not None:
            assert prePCA
            trainX = pca["train_projs"]
            testX = pca["test_projs"]
        else:
            trainX = neurons[:, ntrain]
            testX = neurons[:, ntest]

        projs1 = trainX @ u[:, :nsvc_predict]
        projs2 = testX @ v[:, :nsvc_predict]

        actual_nneurs[s] = min(
            len(ntrain), len(ntest)
        )  # in case either are not n/2, for whatever reason

        ### PREDICT NEURAL ACTIVITY FROM BEHAVIOR
        if motion is not None:
            nrank1 = ranks[ranks <= motion.shape[1]]

            ### REDUCED RANK REGRESSIONS
            for l in range(len(lams)):  # try various regularization
                # find the linear combinations of ntrain's SVCs and the behavior variables which maximally covary
                # i.e., canonical covariance analysis
                atrain, btrain, _, _ = canonical_cov(
                    projs1[itrain, :], motion[itrain, :], lams[l], npc=max(nrank1)
                )
                # same for ntest's SVCs
                atest, btest, _, _ = canonical_cov(
                    projs2[itrain, :], motion[itrain, :], lams[l], npc=max(nrank1)
                )

                for k in range(len(nrank1)):  # try various ranks
                    vp_train = (
                        atrain[:, : nrank1[k]] @ btrain[:, : nrank1[k]].T @ motion.T
                    )
                    vp_test = atest[:, : nrank1[k]] @ btest[:, : nrank1[k]].T @ motion.T

                    s1 = projs1[itest, :].T - vp_train[:, itest]
                    s2 = projs2[itest, :].T - vp_test[:, itest]

                    # compute residual covariance between ntrain and ntest's SVCs
                    # after subtracting predictions from behavior
                    # any residual covariance reflects reliable neuronal dynamics which are no explained by behavior
                    cov_res_beh[: s1.shape[0], k, l, s] = np.sum(s1 * s2, axis=1)

            ### MULTILAYER PERCEPTRON
            if MLPregressor:
                layer_sizes = nrank1

                def test_model(model):
                    model.fit(motion[itrain, :], projs1[itrain, :])
                    vp_train = model.predict(motion[itest, :]).T

                    model.fit(motion[itrain, :], projs2[itrain, :])
                    vp_test = model.predict(motion[itest, :]).T

                    s1 = projs1[itest, :].T - vp_train
                    s2 = projs2[itest, :].T - vp_test

                    cov_res = np.sum(s1 * s2, axis=1)

                    return cov_res

                data = Parallel(n_jobs=DEFAULT_N_JOBS)(
                    delayed(test_model)(MLPRegressor(hidden_layer_sizes=(a)))
                    for a in layer_sizes
                )

                for i in range(len(data)):
                    cov_res_beh_mlp[: len(data[i]), i, s] = data[i]

    if out is not None:
        np.savez(
            file,
            nneur=nneur,
            ranks=ranks,
            lag=lag,
            lams=lams,
            npc_behavior=npc_behavior,
            nsvc_predict=nsvc_predict,
            nsvc=nsvc,
            out=out,
            cov_neur=cov_neur,
            var_neur=var_neur,
            cov_res_beh=cov_res_beh,
            cov_res_beh_mlp=cov_res_beh_mlp,
            actual_nneurs=actual_nneurs,
            u=u,
            v=v,
            ntrain=ntrain,
            ntest=ntest,
            itrain=itrain,
            itest=itest,
            projs1=projs1,
            projs2=projs2,
            pca=pca,
            **kwargs
        )

    return (
        cov_neur,
        var_neur,
        cov_res_beh,
        cov_res_beh_mlp,
        actual_nneurs,
        u,
        v,
        ntrain,
        ntest,
        itrain,
        itest,
        pca,
    )


def main(
    path,
    nneur,
    nsamplings=10,
    lag=0,
    nsvc_predict=128,
    FOV=None,
    shuffle=False,
    t_downsample=None,
    t_subsample=None,
):
    """
    CLI for scaling_analysis.predict.

    Parameters
    ----------
    neurons : array_like of shape (T, N)
        Neuronal timeseries
    nneur : int
        Number of neurons to sample
    nsamplings : int, default=10
       Number of samplings and rounds of SVCA/prediction to perform
    lag : int, default=0
       Desired lag between neurons and motion (positive indicates behavior precedes neurons)
    nsvc_predict : int, default=128
        Number of SVCs to predict from motion
    FOV : str "expand" | float, optional
        If None, neurons are sampled randomly.
        If "expand", neurons are sampled in order of their distance from the center of the volume.
        If float, neurons are sampled from a randomly placed lateral FOV of given size
    shuffle : bool, default=False
        Whether to shuffle motion (recommended if not using the session permutation method)
    t_downsample : int, optional
        If provided, downsamples the volume rate by sampling every `t_downsample` frames
    t_subsample : float, optional
        If provided, subsamples the duration of the recording by only analyzing the
        first `t_subsample` fraction of the recording
    """

    nneur = int(nneur)
    nsamplings = int(nsamplings)
    lag = int(lag)
    nsvc_predict = int(nsvc_predict)
    if FOV == "0":
        FOV = None
    elif FOV is not None and FOV != "expand":
        FOV = float(FOV)
    shuffle = bool(int(shuffle))
    if t_downsample is not None:
        t_downsample = int(t_downsample)
    if t_subsample is not None:
        t_subsample = float(t_subsample)

    expt = Experiment(path)

    neurons = zscore(expt.T_all.astype("single"))

    nneur = get_nneur(neurons.shape[1], nneur)
    nsvc_predict = min(nsvc_predict, int(nneur / 2))
    if nneur >= 65536:
        nsvc = min(neurons.shape[0], 10000)
    else:
        nsvc = 4096

    ranks = np.unique(np.round(2 ** np.arange(0, 7.5, step=0.5))).astype(int)

    motion = expt.motion.astype("single")[:, :500]
    motion = motion - np.mean(motion, axis=0)
    motion = motion / np.std(motion[:, 0])
    motion = motion * 10
    ranks = ranks[ranks < motion.shape[1]]

    if shuffle:
        # Shuffle motion
        Xshuff = np.zeros(motion.shape)
        for i in range(motion.shape[1]):
            nchunks = int(Xshuff.shape[0] / int(72 * expt.fhz))
            tshuff = np.array_split(np.arange(Xshuff.shape[0]), nchunks)
            tshuff = [tshuff[i] for i in np.random.permutation(len(tshuff))]
            tshuff = [x for y in tshuff for x in y]
            Xshuff[: len(tshuff), i] = motion[:, i][tshuff]
        motion = Xshuff
        neurons = neurons[: motion.shape[0], :]

    if FOV != None:
        if FOV == "expand":
            out = os.path.join(expt.out, "predict_vs_FOVexpand")
        else:
            out = os.path.join(expt.out, "predict_vs_FOV")
    else:
        out = os.path.join(expt.out, "predict_vs_nneur")

    suffix = ""

    if t_downsample is not None:
        # Sample every `t_downsample`-th frame
        if lag > 0:
            neurons = neurons[:-lag, :]
            if motion is not None:
                motion = motion[lag:, :]
        elif lag < 0:
            neurons = neurons[-lag:, :]
            if motion is not None:
                motion = motion[:lag, :]

        neurons = neurons[::t_downsample, :]
        if motion is not None:
            motion = motion[::t_downsample, :]
        out = os.path.join(expt.out, "predict_vs_framerate")
        suffix = "_tdownsample" + str(t_downsample)
        lag = 0

    if t_subsample is not None:
        # Sample only `t_subsample` fraction of the recording
        # starting from the start
        m = int(neurons.shape[0] * t_subsample)
        neurons = neurons[:m, :]
        if motion is not None:
            motion = motion[:m, :]
        out = os.path.join(expt.out, "predict_vs_duration")
        suffix = "_tsubsample" + str(t_subsample)

    if shuffle:
        out += "_shuffled"
    print("saving instantaneous prediction results to", out)
    if not os.path.exists(out):
        os.mkdir(out)

    predict_from_behavior(
        neurons,
        nneur,
        expt.centers,
        motion,
        ranks,
        nsamplings=nsamplings,
        lag=lag,
        out=out,
        nsvc=nsvc,
        nsvc_predict=nsvc_predict,
        FOV=FOV,
        name_suffix=suffix,
        prePCA=nneur > 131072,
        interleave=int(72 * expt.fhz),
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
