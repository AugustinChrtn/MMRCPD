import os
import numpy as np
import matplotlib.pyplot as plt
from consts import all_colors, colors, labels, markers


# =====================================================
# INTERNAL
# =====================================================

def _plot_dir(params):
    d = os.path.join(params["run_dir"], "plots")
    os.makedirs(d, exist_ok=True)
    return d


def _draw_ci(ax, x, data, agent):

    data = np.asarray(data)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    ci = 1.96 * std / np.sqrt(len(data))

    ax.plot(
        x, mean,
        color=all_colors[colors[agent]],
        marker=markers[agent],
        label=labels[agent],
        markersize=4
    )

    ax.fill_between(
        x, mean-ci, mean+ci,
        color=all_colors[colors[agent]],
        alpha=0.15
    )


def _group(df):
    out = {}
    for (agent, trial), sub in df.groupby(["agent","episode"]):
        out.setdefault(agent, []).append(sub)
    return out


# =====================================================
# CURVE
# =====================================================

def plot_metric(df, params, metric, ylabel=None, multiply=1):

    grouped = _group(df)
    fig, ax = plt.subplots()

    for agent, trials in grouped.items():

        arr = [t[metric].to_numpy()*multiply for t in trials]
        L = min(map(len, arr))
        arr = np.array([a[:L] for a in arr])

        x = np.arange(L)
        _draw_ci(ax, x, arr, agent)

    ax.set_xlabel("Step", fontweight="bold")
    ax.set_ylabel(ylabel or metric, fontweight="bold")
    ax.legend()

    fig.savefig(
        os.path.join(_plot_dir(params), f"{metric}.pdf"),
        bbox_inches="tight"
    )
    plt.close(fig)


# =====================================================
# TOTAL TIME
# =====================================================

def plot_total_time(df, params):

    grouped = _group(df)

    fig, ax = plt.subplots()

    labels_list = []
    data = []

    for agent, trials in grouped.items():
        labels_list.append(labels[agent])
        data.append([t["time"].sum() for t in trials])

    ax.boxplot(data)
    ax.set_xticklabels(labels_list)
    ax.set_ylabel("Total time (s)")

    fig.savefig(
        os.path.join(_plot_dir(params), "total_time.pdf"),
        bbox_inches="tight"
    )
    plt.close(fig)


# =====================================================
# NPZ MAPS
# =====================================================

def plot_array_maps(arrays, params, env_shape):

    if not arrays:
        return

    for name, arr in arrays.items():

        fig, ax = plt.subplots()
        im = ax.imshow(arr.reshape(env_shape))
        fig.colorbar(im)

        fig.savefig(
            os.path.join(_plot_dir(params), f"{name}.pdf"),
            bbox_inches="tight"
        )
        plt.close(fig)


# =====================================================
# MASTER
# =====================================================

def get_all_plot(df, params, arrays=None, env_shape=None):

    if "reward" in df.columns:
        plot_metric(df, params, "reward", "Reward")

    if "time" in df.columns:
        plot_metric(df, params, "time", "Decision time (ms)", multiply=1000)
        plot_total_time(df, params)

    if env_shape is not None:
        plot_array_maps(arrays, params, env_shape)
