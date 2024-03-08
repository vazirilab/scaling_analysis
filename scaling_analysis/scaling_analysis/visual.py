"""
Compute visual-evoked and spontaneous neuronal dimensions.
"""

import glob
import hdf5storage
import numpy as np
import os
from scipy.stats import zscore
import sys

from PopulationCoding.utils import get_consecutive_chunks
from scaling_analysis.experiment import Experiment
from scaling_analysis.svca import run_SVCA_partition
from scaling_analysis.utils import get_nneur


def _parse_stimulus_log(path):
    """
    Parse visual stimulus log file, producing stims and
    stim_times as required in compare_spont_vis_dims.
    """

    logfile = glob.glob(os.path.join(path, "vis*.log"))[0]
    log = open(logfile).readlines()

    log = log[4:-1]
    log = [x.split(":")[-1].strip() for x in log]

    stims = np.asarray(
        [int(x.split(" ")[-1]) + 1000 if x[0] == "I" else float(x) for x in log]
    )

    # load stimulus timing
    pulse_config = hdf5storage.loadmat(glob.glob(os.path.join(path, "pulse*.mat"))[0])
    stim_kernel = pulse_config["do_table"][:, 1]
    stim_times = get_consecutive_chunks(np.where(stim_kernel > 0)[0])
    stims = stims[: len(stim_times)]

    return stims, stim_times


def compare_spont_vis_dims(
    neurons,
    nneur,
    centers,
    stims,
    stim_times,
    out,
    nframes_stim=6,
    nsamplings=10,
    nsvc=4096,
    min_dt_stim=100,
    **kwargs
):
    """
    Performs SVCA during spontaneous behavior and visual stimulation, and compares the
    similarity of the resulting SVCs.

    Assumes the following experimental protocol (Figure 4D, Manley et al. 2024 [1]_):

    1. Spontaneous epoch
    2. Repeated visual epochs, each consisting of the same pattern of stimuli
    3. Spontaneous epoch

    Parameters
    ----------
    neurons : array_like of shape (N, T)
    nneur : int
        Number of neurons to sample
    centers : array_like of shape (d, N)
       Position of neurons. First two dimensions are used for checkerboarding, i.e. d ≥ 2
    stims : array_like of length nstim
        Labels identifying each stimulus.
        Each stimulus is assumed to be repeated the same number of times
        and in the same order during each repetition
    stim_times : list of length nstim
        list containing the frame indices associated with each stimulus presentation
    out : str
        Path to save results
    nframes_stim : int, default=6
        Length of visual stimulation period, in frames
    nsamplings : int, default=10
        Number of samplings and rounds of SVCA to perform
    nsvc : int, default=4096
        Number of SVCs to compute (if not None, uses randomized SVD approximation)
        i.e., n_randomized in PopulationCoding.dimred.SVCA
    min_dt_stim : int, default=100
        Number of timepoints before the start of the first visual stimulus
        and after the end of the last visual stimulus to
        ignore when selecting spontaneous epochs
    kwargs : passed to run_SVCA_partition

    References
    ----------
    .. [1] Manley, J., Lu, S., Barber, K., Demas, J., Kim, H., Meyer, D., Martínez Traub, F.,
           & Vaziri, A. (2024). Simultaneous, cortex-wide dynamics of up to 1 million neurons
           reveal unbounded scaling of dimensionality with neuron number. Neuron.
           https://doi.org/10.1016/j.neuron.2024.02.011.
    """

    # Not the most elegant way to compute and store these metrics
    # But works for this specific case!
    cosine_sim_stim_stim = np.zeros((nsvc, nsvc, nsamplings))
    cosine_sim_stim_spont = np.zeros((nsvc, nsvc, nsamplings))
    cosine_sim_spont_spont = np.zeros((nsvc, nsvc, nsamplings))
    cosine_sim_spont_shuff = np.zeros((nsvc, nsvc, nsamplings))
    cosine_sim_stim_shuff = np.zeros((nsvc, nsvc, nsamplings))

    cov_neur_spont = np.zeros((nsvc, nsamplings))
    var_neur_spont = np.zeros((nsvc, nsamplings))

    cov_neur_spont_shuff = np.zeros((nsvc, nsamplings))
    var_neur_spont_shuff = np.zeros((nsvc, nsamplings))

    cov_neur_stim = np.zeros((nsvc, nsamplings))
    var_neur_stim = np.zeros((nsvc, nsamplings))

    cov_neur_stim_shuff = np.zeros((nsvc, nsamplings))
    var_neur_stim_shuff = np.zeros((nsvc, nsamplings))

    # Identify the two spontaneous epochs
    # One before the first visual stimulus
    # One after the last visual stimulus
    # Adds a buffer of min_dt_stim frames on either side of visual stimulation
    spont_idx = np.concatenate(
        (
            np.arange(np.min(stim_times) - min_dt_stim),
            np.arange(np.max(stim_times) + min_dt_stim, neurons.shape[0]),
        )
    )

    # average visual responses across repetitions
    nstim = len(np.unique(stims))
    # nrepetitions is 5 in Manley et al. 2024
    nrepetitions = len(np.where(stims == stims[0])[0])

    # SHUFFLED activity
    # Each neuron's activity is broken up into nrepetitions chunks, similar to the visual stimulus blocks
    # These chunks are then randomly permuted for each neuron independently
    Xshuff = neurons.copy()
    for i in range(neurons.shape[1]):
        nchunks = int(Xshuff.shape[0] / nrepetitions)
        tshuff = np.array_split(
            np.roll(
                np.arange(Xshuff.shape[0]), np.random.permutation(Xshuff.shape[0])[0]
            ),
            nchunks,
        )
        tshuff = [tshuff[i] for i in np.random.permutation(len(tshuff))]
        tshuff = [x for y in tshuff for x in y]
        Xshuff[:, i] = Xshuff[:, i][tshuff]

    # Compute each neuron's average response across visual stimulation blocks
    # As well as average for shuffled data
    responses = np.zeros((nframes_stim * nstim, nrepetitions, neurons.shape[1]))
    responses_shuff = np.zeros((nframes_stim * nstim, nrepetitions, neurons.shape[1]))

    stims = [
        stims[i]
        for i in range(len(stims))
        if stim_times[i][-1] + nframes_stim < neurons.shape[0]
    ]
    stim_times = [x for x in stim_times if x[-1] + nframes_stim < neurons.shape[0]]

    for i in range(nrepetitions):
        for j in range(nstim):
            stimi = np.where(stims == stims[j])[0][i]
            nf = neurons[
                stim_times[stimi][0] : stim_times[stimi][0] + nframes_stim, :
            ].shape[0]
            responses[j * nframes_stim : j * nframes_stim + nf, i, :] = neurons[
                stim_times[stimi][0] : stim_times[stimi][0] + nframes_stim, :
            ]
            responses_shuff[j * nframes_stim : j * nframes_stim + nf, i, :] = Xshuff[
                stim_times[stimi][0] : stim_times[stimi][0] + nframes_stim, :
            ]

    avg_responses = np.mean(responses, axis=1)
    avg_responses_shuff = np.mean(responses_shuff, axis=1)

    # COMPUTE SVCs nsamplings TIMES
    for i in range(nsamplings):

        idx_neur = np.random.permutation(neurons.shape[1])[:nneur]

        print("SVCA on spontaneous")
        sneur, varneur, u, v, ntrain, ntest, _, _, _ = run_SVCA_partition(
            neurons[:, idx_neur][spont_idx, :],
            nneur,
            centers=centers[:, idx_neur],
            prePCA=False,
            n_randomized=nsvc,
            **kwargs
        )

        cov_neur_spont[: len(sneur), i] = sneur
        var_neur_spont[: len(sneur), i] = varneur

        print("SVCA on spontaneous - second time")
        sneur2, varneur2, u2, v2, ntrain2, ntest2, _, _, _ = run_SVCA_partition(
            neurons[:, idx_neur][spont_idx, :],
            nneur,
            centers=centers[:, idx_neur],
            prePCA=False,
            n_randomized=nsvc,
            **kwargs
        )

        print("SVCA on spontaneous - shuffled")
        (
            sneur_shuff,
            varneur_shuff,
            u_shuff,
            v_shuff,
            ntrain_shuff,
            ntest_shuff,
            _,
            _,
            _,
        ) = run_SVCA_partition(
            Xshuff[:, idx_neur][spont_idx, :],
            nneur,
            centers=centers[:, idx_neur],
            prePCA=False,
            n_randomized=nsvc,
            **kwargs
        )

        cov_neur_spont_shuff[: len(sneur_shuff), i] = sneur_shuff
        var_neur_spont_shuff[: len(sneur_shuff), i] = varneur_shuff

        print("SVCA on stimulus")
        (
            sneur_stim,
            varneur_stim,
            u_stim,
            v_stim,
            ntrain_stim,
            ntest_stim,
            _,
            _,
            _,
        ) = run_SVCA_partition(
            avg_responses[:, idx_neur],
            nneur,
            centers=centers[:, idx_neur],
            prePCA=False,
            n_randomized=nsvc,
            **kwargs
        )

        cov_neur_stim[: len(sneur_stim), i] = sneur_stim
        var_neur_stim[: len(sneur_stim), i] = varneur_stim

        print("SVCA on stimulus - second time")
        (
            sneur_stim2,
            varneur_stim2,
            u_stim2,
            v_stim2,
            ntrain_stim2,
            ntest_stim2,
            _,
            _,
            _,
        ) = run_SVCA_partition(
            avg_responses[:, idx_neur],
            nneur,
            centers=centers[:, idx_neur],
            prePCA=False,
            n_randomized=nsvc,
            **kwargs
        )

        print("SVCA on stimulus - shuffled")
        (
            sneur_stim_shuff,
            varneur_stim_shuff,
            u_stim_shuff,
            v_stim_shuff,
            ntrain_stim_shuff,
            ntest_stim_shuff,
            _,
            _,
            _,
        ) = run_SVCA_partition(
            avg_responses_shuff[:, idx_neur],
            nneur,
            centers=centers[:, idx_neur],
            prePCA=False,
            n_randomized=nsvc,
            **kwargs
        )

        cov_neur_stim_shuff[: len(sneur_stim_shuff), i] = sneur_stim_shuff
        var_neur_stim_shuff[: len(sneur_stim_shuff), i] = varneur_stim_shuff

        print("computing angles")
        uv = np.zeros((len(idx_neur), nsvc))
        uv[ntrain, :] = u[:, :nsvc]
        uv[ntest, :] = v[:, :nsvc]
        uv = uv / np.linalg.norm(uv, axis=0)

        uv2 = np.zeros((len(idx_neur), nsvc))
        uv2[ntrain2, :] = u2[:, :nsvc]
        uv2[ntest2, :] = v2[:, :nsvc]
        uv2 = uv2 / np.linalg.norm(uv2, axis=0)

        uv_shuff = np.zeros((len(idx_neur), nsvc))
        uv_shuff[ntrain_shuff, :] = u_shuff[:, :nsvc]
        uv_shuff[ntest_shuff, :] = v_shuff[:, :nsvc]
        uv_shuff = uv_shuff / np.linalg.norm(uv_shuff, axis=0)

        uv_stim = np.zeros((len(idx_neur), nsvc))
        uv_stim[ntrain_stim, :] = u_stim[:, :nsvc]
        uv_stim[ntest_stim, :] = v_stim[:, :nsvc]
        uv_stim = uv_stim / np.linalg.norm(uv_stim, axis=0)

        uv_stim2 = np.zeros((len(idx_neur), nsvc))
        uv_stim2[ntrain_stim2, :] = u_stim2[:, :nsvc]
        uv_stim2[ntest_stim2, :] = v_stim2[:, :nsvc]
        uv_stim2 = uv_stim2 / np.linalg.norm(uv_stim2, axis=0)

        uv_stim_shuff = np.zeros((len(idx_neur), nsvc))
        uv_stim_shuff[ntrain_stim_shuff, :] = u_stim_shuff[:, :nsvc]
        uv_stim_shuff[ntest_stim_shuff, :] = v_stim_shuff[:, :nsvc]
        uv_stim_shuff = uv_stim_shuff / np.linalg.norm(uv_stim_shuff, axis=0)

        cosine_sim_stim_stim[:, :, i] = np.abs(np.dot(uv_stim2.T, uv_stim))

        cosine_sim_stim_spont[:, :, i] = np.abs(np.dot(uv_stim.T, uv))

        cosine_sim_spont_spont[:, :, i] = np.abs(np.dot(uv2.T, uv))

        cosine_sim_spont_shuff[:, :, i] = np.abs(np.dot(uv_shuff.T, uv))

        cosine_sim_stim_shuff[:, :, i] = np.abs(np.dot(uv_stim_shuff.T, uv_stim))

    np.savez(
        os.path.join(out, "vis_spont_angles_nneur" + str(nneur)),
        cosine_sim_stim_stim=cosine_sim_stim_stim,
        cosine_sim_stim_spont=cosine_sim_stim_spont,
        cosine_sim_spont_spont=cosine_sim_spont_spont,
        cosine_sim_spont_shuff=cosine_sim_spont_shuff,
        cosine_sim_stim_shuff=cosine_sim_stim_shuff,
        cov_neur_spont=cov_neur_spont,
        var_neur_spont=var_neur_spont,
        cov_neur_spont_shuff=cov_neur_spont_shuff,
        var_neur_spont_shuff=var_neur_spont_shuff,
        cov_neur_stim=cov_neur_stim,
        var_neur_stim=var_neur_stim,
        cov_neur_stim_shuff=cov_neur_stim_shuff,
        var_neur_stim_shuff=var_neur_stim_shuff,
        nframes_stim=nframes_stim,
        nsamplings=nsamplings,
        nsvc=nsvc,
        min_dt_stim=min_dt_stim,
        out=out,
        idx_neur=idx_neur,
        ntrain=ntrain,
        ntest=ntest,
        uv=uv,
        uv2=uv2,
        uv_shuff=uv_shuff,
        uv_stim=uv_stim,
        uv_stim2=uv_stim2,
        uv_stim_shuff=uv_stim_shuff,
        **kwargs
    )

    return


def main(path, nneur=None, nsamplings=10):
    """
    CLI for scaling_analysis.visual.

    For example, run 10 samplings of spontaneous/visual SVCA
    comparisons on 512 neurons:

    :code:`python -m scaling_analysis.visual /path/to/file.h5 512 10`

    Parameters
    ----------
    path : str
        Path to dataset loaded by Experiment
    nneur : int, default is maximum
        Number of neurons to randomly sample
    nsamplings : int, default=1
        Number of random samplings to perform
    """

    nsamplings = int(nsamplings)

    expt = Experiment(path)
    neurons = zscore(expt.T_all.astype("single"))

    if nneur is None or nneur == "0":
        nneur = neurons.shape[1]
    nneur = int(nneur)
    nneur = get_nneur(neurons.shape[1], nneur)

    stims, stim_times = _parse_stimulus_log(expt.out)

    out = os.path.join(expt.out, "vis_spont_angles")
    print("saving visual results to", out)
    if not os.path.exists(out):
        os.mkdir(out)

    compare_spont_vis_dims(
        neurons,
        nneur,
        expt.centers,
        stims,
        stim_times,
        out,
        nframes_stim=int(expt.fhz * 1.3),
        nsamplings=nsamplings,
        nsvc=4096,
        min_dt_stim=100,
        interleave=int(expt.fhz * 72),
        checkerboard=250,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
