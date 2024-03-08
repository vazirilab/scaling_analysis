import matplotlib.pyplot as plt
import numpy as np
import warnings


"""
MATPLOTLIB UTILITIES
"""


def set_my_rcParams():

    # FONT
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 8
    plt.rcParams["legend.fontsize"] = 5.5
    plt.rcParams["legend.title_fontsize"] = 7
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 8

    # AXES
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    # LABELS
    plt.rcParams["axes.titlepad"] = 1
    plt.rcParams["axes.labelpad"] = 1
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.fancybox"] = True

    # FIGURE
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.facecolor"] = "w"
    plt.rcParams["figure.frameon"] = True
    plt.rcParams["figure.constrained_layout.use"] = True

    # IMAGE
    plt.rcParams["image.cmap"] = "gray"
    plt.rcParams["image.aspect"] = "equal"

    # FOR EXPORTING TEXT PROPERLY TO ADOBE ILLUSTRATOR
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def no_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def log_log(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_MIPs(Y, planes_to_show=[5, 10, 20], dorsal_depth=50, axial_spacing=16):
    for i in range(len(planes_to_show)):
        ax = plt.subplot(1, len(planes_to_show), i + 1)
        ax.matshow(Y[:, :, planes_to_show[i]])
        no_ticks(ax)
        depth = dorsal_depth + planes_to_show[i] * axial_spacing
        ax.set_title("Depth of " + str(depth) + "um")
    return


def plot_neurons_behavior(
    neurons, motion, treadmill_velocity, t, clim=[-0.3, 1.5], zspacing=10
):

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(10, 1, (1, 5))
    ax.imshow(
        neurons.T, cmap="viridis", aspect="auto", clim=clim, interpolation="bicubic"
    )
    ax.set_xticks([])
    ax.set_ylabel("Neuron #")

    ax = plt.subplot(10, 1, (6, 9))
    for i in range(motion.shape[1]):
        ax.plot(motion[:, i] - zspacing * i, color="k")
    ax.set_xlim([0, motion.shape[0]])
    no_ticks(ax)
    ax.set_ylabel("Behavior PCs")

    ax = plt.subplot(10, 1, 10)
    ax.plot(t, treadmill_velocity, color="k")
    ax.set_xlim([t[0], t[-1]])
    ax.set_yticks([])
    ax.set_ylabel("Treadmill\nvelocity")
    ax.set_xlabel("Time (sec)")

    return fig


def calc_var_expl(
    cov_neurs,
    var_neurs,
    cov_res_behs,
    model="linear",
    cumulative=False,
    cumulative_pcs=None,
):
    npc_behavior = cov_res_behs.shape[1]

    if cumulative_pcs is None:
        cumulative_pcs = np.arange(cov_res_behs.shape[1])
    cumulative_pcs = cumulative_pcs[cumulative_pcs < cov_res_behs.shape[1]]

    def fn(x):
        if cumulative:
            return np.nancumsum(x[:, cumulative_pcs, :], axis=1)
        else:
            return x

    import warnings

    warnings.simplefilter("ignore")

    ax = 2
    cov_res_behs[cov_res_behs < 0] = 0

    if model == "linear":
        var_expl = fn(
            cov_neurs[:, :npc_behavior, np.newaxis, np.newaxis, :] - cov_res_behs
        ) / fn(var_neurs[:, :npc_behavior, np.newaxis, np.newaxis, :])
        var_expl = np.nanmax(np.nanmax(var_expl, axis=ax), axis=ax)
    elif model == "MLP":
        var_expl = fn(cov_neurs[:, :npc_behavior, np.newaxis, :] - cov_res_behs) / fn(
            var_neurs[:, :npc_behavior, np.newaxis, :]
        )
        var_expl = np.nanmax(var_expl, axis=ax)
    elif model == "history linear":
        var_expl = fn(cov_neurs[:, :npc_behavior, :] - cov_res_behs) / fn(
            var_neurs[:, :npc_behavior, :]
        )
        var_expl = np.nanmax(
            np.nanmax(np.nanmax(np.nanmax(var_expl, axis=ax), axis=ax), axis=ax),
            axis=ax,
        )
    elif model == "history MLP":
        var_expl = fn(cov_neurs[:, :npc_behavior, :] - cov_res_behs) / fn(
            var_neurs[:, :npc_behavior, :]
        )
        var_expl = np.nanmax(np.nanmax(var_expl, axis=ax), axis=ax)

    warnings.simplefilter("default")

    return var_expl
