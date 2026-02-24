import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from consts import all_colors, colors, labels, markers


# =========================================================
# PATH
# =========================================================

def plot_dir(stamp):
    path = os.path.join("results", stamp, "plots")
    os.makedirs(path, exist_ok=True)
    return path


# =========================================================
# CORE DRAWER
# =========================================================

def draw_ci(ax, x, data, agent):
    data = np.asarray(data)

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    ci = 1.96 * std / np.sqrt(len(data))

    ax.plot(x, mean,
            color=all_colors[colors[agent]],
            marker=markers[agent],
            label=labels[agent],
            markersize=4)

    ax.fill_between(x, mean-ci, mean+ci,
                    color=all_colors[colors[agent]],
                    alpha=0.15)


# =========================================================
# DATA LOADING
# =========================================================

def load_csv(path):
    df = pd.read_csv(path)

    grouped = {}

    for (agent, trial), sub in df.groupby(["agent","trial"]):
        grouped.setdefault(agent, []).append(sub)

    return grouped


def load_npz(path):
    if path is None or not os.path.exists(path):
        return {}
    return dict(np.load(path, allow_pickle=True))


# =========================================================
# CURVE PLOT
# =========================================================

def plot_curve(csv_data, value_col, stamp,
               xlabel="Step", ylabel="", multiply=1):

    fig, ax = plt.subplots()

    for agent, trials in csv_data.items():
        arr = [t[value_col].to_numpy()*multiply for t in trials]
        length = min(map(len, arr))
        arr = np.array([a[:length] for a in arr])

        x = np.arange(length)
        draw_ci(ax, x, arr, agent)

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel or value_col, fontweight="bold")
    ax.legend()

    fig.savefig(os.path.join(plot_dir(stamp),
                             f"{value_col}.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# =========================================================
# TOTAL TIME BOXPLOT
# =========================================================

def plot_total_time(csv_data, stamp):

    fig, ax = plt.subplots()

    agents = []
    totals = []

    for agent, trials in csv_data.items():
        agents.append(labels[agent])
        totals.append([t["time"].sum() for t in trials])

    ax.boxplot(totals)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Total time (s)")

    fig.savefig(os.path.join(plot_dir(stamp),
                             "total_time.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# =========================================================
# NPZ HEATMAP (models per state etc.)
# =========================================================

def plot_npz_map(npz_data, key, shape, stamp):

    if key not in npz_data:
        return

    arr = npz_data[key]
    arr = arr.reshape(shape)

    fig, ax = plt.subplots()
    im = ax.imshow(arr)
    fig.colorbar(im, ax=ax)

    fig.savefig(os.path.join(plot_dir(stamp),
                             f"{key}.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MASTER ENTRY
# =========================================================

def get_all_plot(csv_path,
                   stamp,
                   npz_path=None,
                   env_shape=None):

    csv_data = load_csv(csv_path)
    npz_data = load_npz(npz_path)

    # ---- reward curve
    if "reward" in next(iter(next(iter(csv_data.values())))).columns:
        plot_curve(csv_data,
                   "reward",
                   stamp,
                   ylabel="Reward")

    # ---- decision time curve
    if "time" in next(iter(next(iter(csv_data.values())))).columns:
        plot_curve(csv_data,
                   "time",
                   stamp,
                   ylabel="Decision time (ms)",
                   multiply=1000)

        plot_total_time(csv_data, stamp)

    # ---- optional npz maps
    if env_shape is not None:
        for k in npz_data.keys():
            plot_npz_map(npz_data, k, env_shape, stamp)
