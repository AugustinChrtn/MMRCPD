import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from consts import all_colors, colors, labels, markers
from consts import one_step_environments, multi_model_agents, mM_and_RLCD
from const_maze import pattern


# =========================================================
# PATH HANDLING
# =========================================================

def _get_plot_path(run_dir, filename):
    if run_dir is None:
        os.makedirs("results", exist_ok=True)
        return os.path.join("results", filename)

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return os.path.join(plot_dir, filename)


# =========================================================
# CORE DRAWING UTILITIES
# =========================================================

def _plot_ci(ax, x, data, label, color, marker, multiply=1.0):
    data = np.array(data) * multiply

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    ci = 1.96 * std / np.sqrt(len(data))

    ax.plot(x, mean, label=label, color=color, marker=marker, markersize=4)
    ax.fill_between(x, mean-ci, mean+ci, color=color, alpha=0.15)


def _add_change_lines(ax, change_rate, total_steps):
    if change_rate is None:
        return
    nb = int(total_steps // change_rate)
    for i in range(1, nb):
        ax.axvline(change_rate*i, linestyle="--", color="black", alpha=0.15)


def _save(fig, run_dir, name):
    fig.savefig(_get_plot_path(run_dir, name), bbox_inches="tight")
    plt.close(fig)


# =========================================================
# GENERIC CURVE PLOT
# =========================================================

def plot_curves(results,
                change_rate,
                nb_iters,
                steps,
                ylabel,
                xlabel,
                title,
                legend=True,
                multiply=False,
                run_dir=None):

    fig, ax = plt.subplots(figsize=(10,5))

    mult = (1e3/steps) if multiply else 1.0

    for i, (agent, vals) in enumerate(results.items()):
        vals = np.array(vals)
        x = np.arange(vals.shape[1])

        if i == 0:
            _add_change_lines(ax, change_rate, len(x))

        _plot_ci(ax,
                 x,
                 vals,
                 labels[agent],
                 all_colors[colors[agent]],
                 markers[agent],
                 mult)

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    if legend:
        ax.legend()

    title = str(time.time())

    _save(fig, run_dir, f"{title}.pdf")


# =========================================================
# BOX PLOT TIMES
# =========================================================

# def plot_time(times, title, run_dir=None):

#     fig, ax = plt.subplots()

#     names, values = zip(*times.items())
#     ax.boxplot(values)

#     ax.set_xticklabels([labels[n] for n in names])
#     ax.set_ylabel("Time (s)")
#     ax.set_title("Average time per agent")

#     _save(fig, run_dir, f"times_{title}.pdf")


# =========================================================
# AVERAGE COMPUTATION (unified)
# =========================================================

def _compute_means(values, nb_steps, nb_trials, change_rate, mode, multiply):
    values = np.array(values)

    if change_rate % nb_steps != 0:
        raise ValueError("change_rate must be multiple of nb_steps")

    trials_each_change = change_rate // nb_steps
    nb_changes = nb_trials // trials_each_change

    if mode == "per_change":
        out = np.zeros((values.shape[0], nb_changes))

        for i in range(nb_changes):
            seg = values[:, i*trials_each_change:(i+1)*trials_each_change]
            out[:, i] = seg.mean(axis=1)

        x = np.arange(nb_changes)

    elif mode == "after_change":
        out = np.zeros((values.shape[0], trials_each_change))

        for j in range(trials_each_change):
            idx = np.arange(j, nb_trials, trials_each_change)
            seg = values[:, idx]
            out[:, j] = seg.mean(axis=1)

        x = np.arange(1, trials_each_change+1)

    else:
        raise ValueError("mode must be per_change or after_change")

    mult = (1e3/nb_steps) if multiply else 1.0
    return x, out*mult


# =========================================================
# SUMMARY PLOTS
# =========================================================

def plot_metric(results,
                nb_steps,
                nb_trials,
                change_rate,
                title,
                ylabel,
                mode,
                multiply=False,
                legend=True,
                run_dir=None):

    fig, ax = plt.subplots(figsize=(8,5))

    for agent, vals in results.items():

        x, arr = _compute_means(vals,
                                nb_steps,
                                nb_trials,
                                change_rate,
                                mode,
                                multiply)

        _plot_ci(ax,
                 x,
                 arr,
                 labels[agent],
                 all_colors[colors[agent]],
                 markers[agent])

    ax.set_xlabel("Change index" if mode=="per_change"
                  else "Steps after change", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    if legend:
        ax.legend()

    _save(fig, run_dir, f"{title}_{mode}.pdf")


# =========================================================
# PERFORMANCE SUMMARY FILE
# =========================================================

def save_summary(all_rewards, title, run_dir):

    path = _get_plot_path(run_dir, f"summary_{title}.txt")

    with open(path,"w") as f:
        for agent, vals in all_rewards.items():
            vals = np.array(vals)
            f.write(
                f"{agent} mean={vals.mean()} "
                f"std={vals.mean(axis=1).std()}\n"
            )


# =========================================================
# MAIN ENTRY
# =========================================================

def get_all_plot(results, parameters,
                 legend=True,
                 suptitle="Experiment"):

    run_dir = parameters["run_dir"]

    agents = parameters["agents"]
    env_name = parameters["env_name"]
    nb_iters = parameters["nb_iters"] * len(parameters["env_param"])
    trials = parameters["trials"]
    steps = parameters["max_step"]
    #title = parameters["time"]
    change_rate = parameters["env_param"][0]["step_change"]

    if change_rate is None:
        change_rate = steps*trials

    env_is_one_step = env_name in one_step_environments

    # -----------------------------------------------------
    # COLLECT
    # -----------------------------------------------------

    rewards = {a:[] for a in agents}
    times = {a:[] for a in agents}
    totals = {a:[] for a in agents}
    best = {a:[] for a in agents}

    for info, vals in results.items():
        agent = info[1]
        rewards[agent].append(vals["reward"])
        times[agent].append(vals["times"])
        totals[agent].append(vals["total_time"])
        if env_is_one_step:
            best[agent].append(vals["best_action"])

    # -----------------------------------------------------
    # CHOOSE TARGET
    # -----------------------------------------------------

    metric = best if env_is_one_step else rewards
    ylabel = ("Best action probability"
              if env_is_one_step else "Reward")

    # -----------------------------------------------------
    # CURVES
    # -----------------------------------------------------

    plot_curves(metric,
                change_rate,
                nb_iters,
                steps,
                ylabel,
                "Steps",
                legend,
                run_dir=run_dir)

    plot_curves(times,
                change_rate,
                nb_iters,
                steps,
                "Decision time (ms)",
                "Steps",
                legend,
                multiply=True,
                run_dir=run_dir)

    # plot_time(totals, title, run_dir)

    # -----------------------------------------------------
    # SUMMARY METRICS
    # -----------------------------------------------------

    plot_metric(metric,
                steps,
                trials,
                change_rate,
                ylabel,
                mode="per_change",
                run_dir=run_dir)

    plot_metric(metric,
                steps,
                trials,
                change_rate,
                ylabel,
                mode="after_change",
                run_dir=run_dir)

    save_summary(rewards, run_dir)


# =========================================================
# MAZE PLOT
# =========================================================

def plot_maze(world, 
              path,
              labels_arr=None,
              arrows=None):

    size = world.shape
    img = np.ones((*size,3))

    walls = world==-1
    start = world==-2
    goal = world>0

    img[walls] = [0,0,0]
    img[start] = [0.2,0.5,1]
    img[goal] = [1,0,0]

    fig, ax = plt.subplots()
    ax.imshow(img)

    if arrows is not None:
        for i in range(size[0]):
            for j in range(size[1]):
                if walls[i,j]:
                    continue
                a = arrows[i,j]
                if a==0: ax.arrow(j,i+0.2,0,-0.2)
                if a==1: ax.arrow(j,i-0.2,0,0.2)
                if a==2: ax.arrow(j,i,-0.2,0)
                if a==3: ax.arrow(j,i,0.2,0)

    if labels_arr is not None:
        for i in range(size[0]):
            for j in range(size[1]):
                ax.text(j,i,str(labels_arr[i,j]),
                        ha="center",va="center")

    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(path,bbox_inches="tight")
    plt.close(fig)
