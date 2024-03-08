"""
Functions for running SVCA and predicting neural SVCs
from multi-timepoint behavior PCs (i.e. with some "history").

Note `predict_from_behavior_history` is highly similar
from `scaling_analysis.predict.predict_from_behavior`,
and they could likely be refactored.
"""

from joblib import Parallel, delayed
import numpy as np
import os
from scipy.stats import zscore
from sklearn.neural_network import MLPRegressor
import sys
from tqdm import tqdm

from PopulationCoding.predict import canonical_cov
from PopulationCoding.utils import get_consecutive_chunks
from scaling_analysis.experiment import Experiment
from scaling_analysis.svca import run_SVCA_partition
from scaling_analysis.utils import correct_lag, get_nneur, DEFAULT_N_JOBS


def predict_from_behavior_history(
    neurons,
    nneur,
    centers,
    motion,
    ranks,
    pres,
    posts,
    navg,
    nsamplings=10,
    lag=0,
    lams=[0.01, 0.1, 1],
    nsvc=2048,
    nsvc_predict=128,
    npc_behavior=256,
    out=None,
    checkerboard=250,
    FOV=None,
    prePCA=True,
    name_suffix="",
    MLPregressor=True,
    **kwargs
):
    """
    Samples a given number of neurons, performs SVCA, and predicts
    neural SVCs from a set of *multi-timepoint* motion variables,
    i.e. with some "history".

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
    pres : array_like
       List of numbers of motion frames BEFORE neural activity to include.
       Each pair of pres and posts will be tried to identify an optimal
       window of multi-timepoint behavior
    posts : array_like
       Lists of numbers of motion frames AFTER neural activity to include
    navg : int, optional
       If specified, the multi-timepoint window is binned into bins of length
       navg. This can be utilized to decrease the dimensionality of the multi-timepoint
       behavioral data, and is recommended whenever the frame rate is fast enough
       that adjacent behavioral frames are highly correlated
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
        perception with hidden layer of size nranks. Note that this only fits an MLP
        model for the instantaneous case and the *best* multi-timepoint window.
    kwargs : passed to scaling_analysis.svca.run_SVCA_partition

    Returns
    -------
    cov_neur : ndarray of shape (nsvc, len(pres), len(posts), len(ranks), len(lams), nsamplings)
        Reliable covariance of each SVC on held-out testing timepoints, across samplings.
        Note that these cov_neur, etc. must be reported for each multi-timepoint window.
        This is because the number of timepoints changes (and thus the total variance/covariance)
    var_neur : ndarray of shape (nsvc, len(pres), len(posts), len(ranks), len(lams), nsamplings)
        Reliable variance of each SVC on held-out testing timepoints, across samplings
    cov_res_beh : ndarray of shape (nsvc_predict, len(pres), len(posts), len(ranks), len(lams), nsamplings)
        Residual SVC covariance after subtracting predictions from motion, across samplings
    cov_neur_mlp : ndarray of shape (nsvc_predict, 2, len(ranks), nsamplings)
        Reliable covariance of each SVC on held-out testing timepoints, across samplings,
        for the multi-layer perceptron models. The second dimension reflects the two MLP models
        fit: the instantaneous case and the optimal multi-timepoint window
    var_neur_mlp : ndarray of shape (nsvc_predict, 2, len(ranks), nsamplings)
        Reliable variance of each SVC on held-out testing timepoints, across samplings
    cov_res_beh_mlp : (nsvc_predict, 2, len(ranks), nsamplings)
        Residual SVC covariance after subtracting predictions from motion (with MLP), across samplings.
        If MLPregressor=false, all NaNs
    actual_nneurs : ndarray of shape (nsamplings,)
        Actual number of neurons used, across samplings, reported as min(len(ntrain), len(ntest))
    """

    # Initialize variables
    cov_neur = (
        np.zeros((nsvc, len(pres), len(posts), len(ranks), len(lams), nsamplings))
        + np.nan
    )
    var_neur = (
        np.zeros((nsvc, len(pres), len(posts), len(ranks), len(lams), nsamplings))
        + np.nan
    )
    cov_res_beh = (
        np.zeros(
            (nsvc_predict, len(pres), len(posts), len(ranks), len(lams), nsamplings)
        )
        + np.nan
    )
    cov_neur_mlp = np.zeros((nsvc_predict, 2, len(ranks), nsamplings)) + np.nan
    var_neur_mlp = np.zeros((nsvc_predict, 2, len(ranks), nsamplings)) + np.nan
    cov_res_beh_mlp = np.zeros((nsvc_predict, 2, len(ranks), nsamplings)) + np.nan
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
        _, _, u, v, ntrain, ntest, itrain, itest, pca = run_SVCA_partition(
            neurons,
            nneur,
            checkerboard=checkerboard,
            centers=centers,
            FOV=FOV,
            n_randomized=nsvc,
            prePCA=prePCA,
            **kwargs
        )

        if pca is not None:
            assert prePCA
            trainX = pca["train_projs"]
            testX = pca["test_projs"]
        else:
            trainX = neurons[:, ntrain]
            testX = neurons[:, ntest]

        projs1 = nu = trainX @ u[:, :nsvc_predict]
        projs2 = nv = testX @ v[:, :nsvc_predict]

        actual_nneurs[s] = min(
            len(ntrain), len(ntest)
        )  # in case either are not n/2, for whatever reason

        ### PREDICT NEURAL ACTIVITY FROM BEHAVIOR
        if motion is not None:
            nrank1 = ranks[ranks <= motion.shape[1]]

            trains = get_consecutive_chunks(itrain)
            tests = get_consecutive_chunks(itest)

            trains = [trains[i] for i in np.random.permutation(len(trains))]
            tests = [tests[i] for i in np.random.permutation(len(tests))]

            #### TRY VARIOUS INPUT LENGTHS
            #### Parameterized by number of frames PRE neurons
            #### and number of frames POST neurons
            #### navg specifies whether the behavioral history is binned for efficiency
            for a in range(len(pres)):
                for p in range(len(posts)):
                    if a == 0 and p == 0:
                        # instantaneous!
                        navg_curr = 0
                    else:
                        navg_curr = navg

                    npre = pres[a]
                    npost = posts[p]

                    def get_history_chunks(trains, npre, npost, nu, nv, motion):
                        trainu = []
                        trainv = []
                        traints = []
                        trainmotion = []

                        for chunk in trains:
                            curr = chunk[npre : -(npost + 1)]
                            if len(curr) > 0:
                                trainu.append(nu[curr, :])
                                trainv.append(nv[curr, :])
                                traints.append(curr)
                                trainmotion.append(
                                    np.concatenate(
                                        [
                                            np.concatenate(
                                                [
                                                    np.mean(
                                                        motion[
                                                            curr[ii]
                                                            + x : curr[ii]
                                                            + x
                                                            + max(navg_curr, 1),
                                                            :,
                                                        ],
                                                        axis=0,
                                                    )
                                                    for x in range(
                                                        -npre,
                                                        npost + 1 - navg_curr,
                                                        max(navg_curr, 1),
                                                    )
                                                ]
                                            ).reshape(1, -1)
                                            for ii in range(len(curr))
                                        ],
                                        axis=0,
                                    )
                                )

                        trainu = np.concatenate(trainu)
                        trainv = np.concatenate(trainv)
                        trainmotion = np.concatenate(trainmotion)
                        return trainu, trainv, traints, trainmotion

                    trainu, trainv, traints, trainmotion = get_history_chunks(
                        trains, npre, npost, nu, nv, motion
                    )
                    testu, testv, testts, testmotion = get_history_chunks(
                        tests, npre, npost, nu, nv, motion
                    )

                    ### REDUCED RANK REGRESSIONS
                    for l in range(len(lams)):  # try various regularization
                        # find the linear combinations of ntrain's SVCs and the behavior variables which maximally covary
                        # i.e., canonical covariance analysis
                        atrain, btrain, _, _ = canonical_cov(
                            trainu, trainmotion, lams[l], npc=max(nrank1)
                        )
                        # same for ntest's SVCs
                        atest, btest, _, _ = canonical_cov(
                            trainv, trainmotion, lams[l], npc=max(nrank1)
                        )

                        for k in range(len(nrank1)):  # try various ranks
                            vp_train = (
                                atrain[:, : nrank1[k]]
                                @ btrain[:, : nrank1[k]].T
                                @ testmotion.T
                            )
                            vp_test = (
                                atest[:, : nrank1[k]]
                                @ btest[:, : nrank1[k]].T
                                @ testmotion.T
                            )

                            s1 = testu.T - vp_train
                            s2 = testv.T - vp_test

                            # compute residual covariance between ntrain and ntest's SVCs
                            # after subtracting predictions from behavior
                            # any residual covariance reflects reliable neuronal dynamics which are no explained by behavior
                            cov_res_beh[: s1.shape[0], a, p, k, l, s] = np.sum(
                                s1 * s2, axis=1
                            )
                            # must save cov_neur and var_neur for EACH pre/post/etc., since they contain different numbers of timepoints
                            cov_neur[: s1.shape[0], a, p, k, l, s] = np.sum(
                                testu * testv, axis=0
                            )
                            var_neur[: s1.shape[0], a, p, k, l, s] = (
                                np.sum(testu**2 + testv**2, axis=0) / 2
                            )

            ### MULTILAYER PERCEPTROM
            if MLPregressor:
                layer_sizes = nrank1

                # ONLY RUN MLP ON INSTANTANEOUS AND OPTIMAL HISTORY LENGTH!
                # find best pre/post history
                curr = cov_res_beh[:, :, :, :, :, s]
                curr = np.mean(np.min(np.min(curr, axis=-1), axis=-1), axis=0)
                abestneur, pbestneur = np.unravel_index(np.argmin(curr), curr.shape)
                ntodo = [(0, 0), (pres[abestneur], posts[pbestneur])]

                for ni in range(len(ntodo)):
                    npre = ntodo[ni][0]
                    npost = ntodo[ni][1]

                    if npre == 0 and npost == 0:
                        # instantaneous!
                        navg_curr = 0
                    else:
                        navg_curr = navg

                    trainu, trainv, traints, trainmotion = get_history_chunks(
                        trains, npre, npost, nu, nv, motion
                    )
                    testu, testv, testts, testmotion = get_history_chunks(
                        tests, npre, npost, nu, nv, motion
                    )

                    def test_model(model):
                        model.fit(trainmotion, trainu)
                        vp_train = model.predict(testmotion).T

                        model.fit(trainmotion, trainv)
                        vp_test = model.predict(testmotion).T

                        s1 = testu.T - vp_train
                        s2 = testv.T - vp_test

                        cov_res = np.sum(s1 * s2, axis=1)
                        cov_neur = np.sum(testu * testv, axis=0)
                        var_neur = np.sum(testu**2 + testv**2, axis=0) / 2

                        return cov_res, cov_neur, var_neur

                    data = Parallel(n_jobs=DEFAULT_N_JOBS)(
                        delayed(test_model)(MLPRegressor(hidden_layer_sizes=(aa)))
                        for aa in layer_sizes
                    )

                    for i in range(len(data)):
                        cov_res_beh_mlp[: len(data[i][0]), ni, i, s] = data[i][0]
                        cov_neur_mlp[: len(data[i][1]), ni, i, s] = data[i][1]
                        var_neur_mlp[: len(data[i][2]), ni, i, s] = data[i][2]

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
            cov_neur_mlp=cov_neur_mlp,
            var_neur_mlp=var_neur_mlp,
            cov_res_beh_mlp=cov_res_beh_mlp,
            actual_nneurs=actual_nneurs,
            pres=pres,
            posts=posts,
            abestneur=abestneur,
            pbestneur=pbestneur,
            navg=navg,
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
        cov_neur_mlp,
        var_neur_mlp,
        cov_res_beh_mlp,
        actual_nneurs,
    )


def main(path, nneur, nsamplings=10, lag=0, nsvc_predict=128, FOV=None, shuffle=False):
    """
    CLI for scaling_analysis.predict_from_history.

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

    if shuffle:
        out += "_shuffled"
    out += "_history"
    if not os.path.exists(out):
        os.mkdir(out)
    print("saving multi-timepoint prediction results to", out)

    pres = [int(np.round(x * expt.fhz)) for x in [0, 1, 3, 6]]
    posts = [int(np.round(x * expt.fhz)) for x in [0, 3, 6, 9]]
    ranks = np.unique(np.round(2 ** np.arange(2, 7))).astype(int)
    navg = int(np.round(expt.fhz))

    predict_from_behavior_history(
        neurons,
        nneur,
        expt.centers,
        motion,
        ranks,
        pres,
        posts,
        navg=navg,
        nsamplings=nsamplings,
        lag=lag,
        out=out,
        nsvc=nsvc,
        nsvc_predict=nsvc_predict,
        FOV=FOV,
        prePCA=nneur > 131072,
        interleave=int(72 * expt.fhz),
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
