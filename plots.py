import inspect
from consts import all_colors, colors, labels, markers
from consts import one_step_environments, multi_model_agents
from consts import mM_and_RLCD
import time
import numpy as np
import matplotlib.pyplot as plt
from const_maze import pattern
import matplotlib
import os
import json
import pandas as pd
matplotlib.use("Agg")

# ---------------------------------------------------------------------------- #
# Retrieve the data
# ---------------------------------------------------------------------------- #


def get_plot_from_saved(dir_path, suptitle="Uncertain variation", legend=False):

    # Path to files
    results_path = f"{dir_path}/episode_results.csv.gz"
    parameters_path = f"{dir_path}/parameters.json"
    arrays_path = f"{dir_path}/final_arrays.npz"

    # Loading the files
    results = pd.read_csv(results_path)
    final_arrays = np.load(arrays_path)

    with open(parameters_path, 'r') as f:
        # Parsing the json file into a Python dictionary
        parameters = json.load(f)

    get_all_plot(results, parameters, final_arrays, dir_path,
                 suptitle=suptitle, legend=legend)


# ---------------------------------------------------------------------------- #
# Useful functions - Data structure
# ---------------------------------------------------------------------------- #

# To use a dictionary which includes parameters func does not need
def call_with_valid_args(func, **kwargs):
    valid = inspect.signature(func).parameters
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return func(**filtered)


def get_agent_array(df, agent, metric):

    arr = df[df.agent == agent].pivot(
        index="trial_id", columns="episode", values=metric).to_numpy()

    return arr

# def build_raw_dict(df, metric):
#     out = {}

#     for agent, sub in df.groupby("agent"):
#         out[agent] = (
#             sub
#             .pivot(index="seed", columns="trial", values=metric)
#             .sort_index()
#             .to_numpy()
#         )

#     return out


# ---------------------------------------------------------------------------- #
# Useful functions - Plotting
# ---------------------------------------------------------------------------- #


def plot_change(total_steps,
                change_rate,
                steps):

    nb_changes = int(total_steps // change_rate)
    for i in range(1, nb_changes):
        plt.axvline(x=change_rate//steps*i,
                    linestyle='--',
                    color='black',
                    alpha=0.1)


# ---------------------------------------------------------------------------- #
# Main plotting function
# ---------------------------------------------------------------------------- #


def get_all_plot(results,
                 parameters,
                 arrays,
                 dir_path,
                 legend,
                 suptitle='Uncertain variation'):

    # # Creating a plots directory if it does not exist
    # save_path = os.path.join(dir_path, "plots")
    # os.makedirs(d, exist_ok=True)
    save_path = dir_path
    # General parameters used in all plots
    params_plot = {'change_rate': parameters['env_param'][0]['step_change'],
                   'steps': parameters['max_step'],
                   'trials': parameters['trials'],
                   'save_path': save_path,
                   'legend': legend,
                   'suptitle': suptitle,
                   'nb_iters': parameters['nb_iters']}

    # If there is no change, use an arbitrary high number for the change rate
    if params_plot['change_rate'] is None:
        params_plot['change_rate'] = params_plot['steps'] * \
            params_plot['trials']

    # Gives what to plot and the associated ylabel
    metric_specs = [
        ("reward", "Rewards"),
        ("best_action", "Probability of selecting the best action"),
        ("time (ms)", "Time per decision (ms)"),
        ("distance", "Euclidean distance")
    ]

    # if the env is one step, x-axis is step and we plot best_action instead
    # of reward
    env_is_one_step = results['environment'][0] in one_step_environments
    if env_is_one_step:
        event = 'steps'
        metric_specs.pop(0)
    else:
        event = 'trials'
        metric_specs.pop(1)

    # Uses metric_specs to specify the plotting parameters
    metrics_to_plot = {
        name: {
            **params_plot,
            "ylabel": label,
            "xlabel": f"Number of {event}",
            "save_name": name
        }
        for name, label in metric_specs
    }

    # Plots

    # The call with valid args makes sure that the function only uses the
    # necessary parameters
    call_with_valid_args(plot_four_models,
                         results=results,
                         xlabel=f"Number of task changes",
                         **params_plot)

    perf_metric = metric_specs[0][0]
    metric_2 = "time (ms)"
    call_with_valid_args(plot_four,
                         results=results,
                         metric_1=perf_metric,
                         metric_2=metric_2,
                         ylabel_metric_1=metrics_to_plot[perf_metric]["ylabel"],
                         ylabel_metric_2=metrics_to_plot[metric_2]["ylabel"],
                         xlabel=f"Number of {event} after the task change",
                         **params_plot)

    # .txt file with mean perf
    general_performance(results, perf_metric, save_path)
    # boxplot with total times per seed
    plot_boxplot_time(results=results, 
                      save_path=save_path,
                      metric_name=metric_2)

    for metric_name, metric_params in metrics_to_plot.items():
        call_with_valid_args(plot_curves,
                             results=results,
                             metric_name=metric_name,
                             **metric_params)

        # call_with_valid_args(plot_metric_from_df,
        #                      results=results,
        #                      metric_name=metric_name,
        #                      mode='after_change',
        #                      **metric_params
        #                      )

        call_with_valid_args(plot_two,
                             results=results,
                             metric_name=metric_name,
                             **metric_params
                             )
    

# ---------------------------------------------------------------------------- #
# Plot curves, for legend and uncertain variations
# ---------------------------------------------------------------------------- #

def plot_curves(results,
                metric_name,
                change_rate,
                steps,
                ylabel='Reward',
                xlabel='Steps',
                #legend=True,
                suptitle='Uncertain variation',
                save_name='',
                save_path=''):
    """Two-subplot version:
       - left: only the legend (bold)
       - right: the curves and change lines, no legend
    """
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax_left.plot([0, 1], [0, 1], alpha=0)     # invisible dummy plot
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)

    ax_left.tick_params(axis='x', colors='white')   # x-tick labels invisible
    ax_left.tick_params(axis='y', colors='white')   # y-tick labels invisible
    ax_left.spines['bottom'].set_color('white')
    ax_left.spines['left'].set_color('white')
    ax_left.spines['right'].set_color('white')
    ax_left.spines['top'].set_color('white')

    # Getting the stats for the metric
    stats = results.groupby(['agent', 'episode'], sort=False)[metric_name].agg(
        ['mean', 'std', 'count']).reset_index()

    agents = results['agent'].unique()

    for agent in agents:

        agent_stats = stats[stats['agent'] == agent]
        y = agent_stats['mean']
        std = agent_stats['std']
        n = agent_stats['count']
        x = np.arange(len(y))

        ci = 1.96 * std / np.sqrt(n)

        ax_right.plot(x,
                      y,
                      label=labels[agent],
                      color=all_colors[colors[agent]],
                      marker=markers[agent],
                      markersize=4)
        ax_right.fill_between(x, y-ci, y+ci, alpha=0.2, 
                              color=all_colors[colors[agent]])

    # Plot change lines
    total_steps = steps*len(x)
    plot_change(total_steps, change_rate, steps)

    # Configure right subplot (the actual plot)
    ax_right.set_xlabel(xlabel, fontweight='bold')
    ax_right.set_ylabel(ylabel, fontweight='bold')

    # Collect handles/labels from the right axis
    handles, labs = ax_right.get_legend_handles_labels()

    # Remove any legend from the right subplot (it is on the left)
    right_leg = ax_right.get_legend()
    if right_leg is not None:
        right_leg.remove()

    # Left subplot: only show the legend (centered)
    if len(handles) > 0:
        # Put legend into the left axis centered. Make labels bold via prop.
        legend_props = {'weight': 'bold', 'size': 20}
        ax_left.legend(handles, labs, loc='center', prop=legend_props)
    else:
        # If no legend requested, keep left axis empty
        ax_left.set_visible(False)

    # Turn off axes ticks/lines for left subplot so only the legend shows
    ax_left.set_frame_on(True)         # keeps a frame box
    ax_left.patch.set_alpha(0)         # (invisible)

    ax_left.set_xlabel(" x-axis ", alpha=0, fontweight='bold')
    ax_left.set_ylabel(" y-axis ", alpha=0, fontweight='bold')

    # Right subplot title in bold
    ax_left.set_title(suptitle, fontweight='bold',
                      size=22, pad=25, color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{save_name}.pdf'), bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------- #
# Plot average after change or over time
# ---------------------------------------------------------------------------- #

def aggregate_metric(values,
                     nb_trials,
                     nb_steps,
                     change_rate,
                     mode):

    trials_each_change = change_rate // nb_steps

    if mode == "after_change":

        not_used = nb_trials % trials_each_change
        out = np.zeros((len(values), trials_each_change))

        for j in range(trials_each_change):
            idx = np.arange(j, nb_trials-not_used, trials_each_change)
            r = values[:, idx]
            out[:, j] = r.mean(axis=1)

        x = np.arange(1, trials_each_change+1)

    elif mode == "over_time":

        nb_changes = nb_trials // trials_each_change
        out = np.zeros((len(values), nb_changes))

        for i in range(nb_changes):
            start = i*trials_each_change
            r = values[:, start:start+trials_each_change]
            out[:, i] = r.mean(axis=1)

        x = np.arange(nb_changes)

    else:
        raise ValueError("mode must be 'after_change' or 'over_time'")

    return out, x


def plot_metric_from_df(results,
                        metric_name,
                        mode,
                        trials,
                        steps,
                        change_rate,
                        nb_iters,
                        ax=None,
                        legend=True,
                        xlabel=None,
                        ylabel=None,
                        save=True,
                        save_path='',
                        save_name='',
                        grid=False):

    if ax is None:
        fig, ax = plt.subplots()

    agents = results['agent'].unique()

    for agent in agents:

        values = get_agent_array(results, agent, metric_name)
        agg, x = aggregate_metric(
            values,
            trials,
            steps,
            change_rate,
            mode
        )

        mean = agg.mean(axis=0)
        std = agg.std(axis=0)
        ci = 1.96 * std / np.sqrt(len(values))

        if not np.isnan(mean).any():
            ax.plot(x,
                    mean,
                    label=labels[agent],
                    color=all_colors[colors[agent]],
                    marker=markers[agent],
                    markersize=4)

            ax.fill_between(x,
                            mean-ci,
                            mean+ci,
                            color=all_colors[colors[agent]],
                            alpha=0.15)

    ax.xaxis.get_major_locator().set_params(integer=True)
    if grid:
        ax.grid(alpha=0.2)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")
    if legend:
        ax.legend()
    if save:
        plt.savefig(os.path.join(save_path, f'{save_name}_{mode}.pdf'))
        plt.close()

# ---------------------------------------------------------------------------- #
# Sum-up plots
# ---------------------------------------------------------------------------- #


def plot_two(results,
             metric_name,
             steps,
             trials,
             change_rate,
             nb_iters,
             xlabel,
             ylabel,
             save_path,
             legend=False,
             suptitle='Uncertain-Volatile variation'):

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    plot_metric_from_df(results,
                        metric_name,
                        "over_time",
                        trials,
                        steps,
                        change_rate,
                        nb_iters,
                        ax=axs[0],
                        legend=False,
                        xlabel="Number of task changes",
                        ylabel=ylabel,
                        save=False)

    plot_metric_from_df(results,
                        metric_name,
                        "after_change",
                        trials,
                        steps,
                        change_rate,
                        nb_iters,
                        ax=axs[1],
                        legend=False,
                        xlabel=f'{xlabel} after the task change',
                        ylabel=ylabel,
                        save=False)

    if legend:
        handles, labels = axs[0].get_legend_handles_labels()
        fig_legend = plt.figure(figsize=(8, 2))
        fig_legend.legend(handles, labels, loc='center', ncol=2,
                          prop={"weight": "bold", "size": 14})
        fig_legend.tight_layout()
        fig_legend.savefig(os.path.join(save_path, 'Legend.pdf'),
                           bbox_inches='tight')
        plt.close(fig_legend)

    axs[0].set_title(suptitle, fontweight='bold', size=22, pad=25)
    plt.savefig(os.path.join(save_path,
                             f'{metric_name}_over_time_and_after_change.pdf'),
                             bbox_inches='tight')
    plt.close()


def plot_four_models(results,
                     steps,
                     trials,
                     change_rate,
                     nb_iters,
                     xlabel,
                     save_path,
                     legend=False):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    plot_metric_from_df(results,
                        metric_name='nb_model',
                        mode="over_time",
                        trials=trials,
                        steps=steps,
                        change_rate=change_rate,
                        nb_iters=nb_iters,
                        ax=axs[0, 0],
                        legend=False,
                        xlabel=xlabel,
                        save=False,
                        grid=True)

    plot_metric_from_df(results,
                        metric_name='nb_creation',
                        mode="over_time",
                        trials=trials,
                        steps=steps,
                        change_rate=change_rate,
                        nb_iters=nb_iters,
                        ax=axs[0, 1],
                        legend=False,
                        xlabel=xlabel,
                        ylabel='Number of models created',
                        save=False,
                        grid=True)

    plot_metric_from_df(results,
                        metric_name='nb_merging',
                        mode="over_time",
                        trials=trials,
                        steps=steps,
                        change_rate=change_rate,
                        nb_iters=nb_iters,
                        ax=axs[1, 0],
                        legend=False,
                        xlabel=xlabel,
                        ylabel='Number of models merged',
                        save=False,
                        grid=True)

    plot_metric_from_df(results,
                        metric_name='nb_merging',
                        mode="over_time",
                        trials=trials,
                        steps=steps,
                        change_rate=change_rate,
                        nb_iters=nb_iters,
                        ax=axs[1, 1],
                        legend=False,
                        xlabel=xlabel,
                        ylabel='Number of models forgotten',
                        save=False,
                        grid=True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # Create a single legend below all subplots
    odd = len(labels) % 2
    if legend:
        fig.legend(handles,
                   labels,
                   loc="lower center",
                   ncol=2+odd,
                   fontsize=12)

    # Adjust layout to make space for the legend

    bottom_adjust = 0.1+0.03*(len(labels)//2)
    # Adjust layout to make space for the legend
    plt.subplots_adjust(left=0.1, right=0.9,
                        top=0.9, bottom=bottom_adjust,
                        wspace=0.3, hspace=0.3)
    # fig.tight_layout(pad=5.0)
    plt.savefig(os.path.join(save_path, 'multi-model-metrics.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_four(results,
              metric_1,
              metric_2,
              steps,
              trials,
              change_rate,
              nb_iters,
              xlabel,
              ylabel_metric_1,
              ylabel_metric_2,
              save_path,
              legend=True):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plot_metric_from_df(results,
                        metric_name=metric_1,
                        mode="over_time",
                        trials=trials,
                        steps=steps,
                        change_rate=change_rate,
                        nb_iters=nb_iters,
                        ax=axs[0, 0],
                        legend=False,
                        xlabel="Number of task changes",
                        ylabel=ylabel_metric_1,
                        save=False)

    plot_metric_from_df(results,
                        metric_name=metric_1,
                        mode="after_change",
                        trials=trials,
                        steps=steps,
                        change_rate=change_rate,
                        nb_iters=nb_iters,
                        ax=axs[0, 1],
                        legend=False,
                        xlabel=xlabel,
                        ylabel=ylabel_metric_1,
                        save=False)

    plot_metric_from_df(results,
                        metric_name=metric_2,
                        mode="over_time",
                        trials=trials,
                        steps=steps,
                        change_rate=change_rate,
                        nb_iters=nb_iters,
                        ax=axs[1, 0],
                        legend=False,
                        xlabel='Number of task changes',
                        ylabel=ylabel_metric_2,
                        save=False)

    plot_metric_from_df(results,
                        metric_name=metric_2,
                        mode="after_change",
                        trials=trials,
                        steps=steps,
                        change_rate=change_rate,
                        nb_iters=nb_iters,
                        ax=axs[1, 1],
                        legend=False,
                        xlabel=xlabel,
                        ylabel=ylabel_metric_2,
                        save=False)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # Create a single legend below all subplots
    odd = len(labels) % 2
    if legend:
        fig.legend(handles,
                   labels,
                   loc="lower center",
                   ncol=2+odd,
                   fontsize=12)

    # Adjust layout to make space for the legend

    bottom_adjust = 0.1+0.03*(len(labels)//2)
    # Adjust layout to make space for the legend
    plt.subplots_adjust(left=0.1, right=0.9,
                        top=0.9, bottom=bottom_adjust,
                        wspace=0.3, hspace=0.3)
    # fig.tight_layout(pad=5.0)
    plt.savefig(os.path.join(save_path, f'Metrics_{metric_1}_{metric_2}.pdf'),
                bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------- #
# Plot general metrics (average reward, average times)
# ---------------------------------------------------------------------------- #


def general_performance(results, metric_name, save_path):
    path = os.path.join(save_path, 'average_performance.txt')
    f = open(path, "w")
    agents = results["agent"].unique()
    for agent in agents:
        values = get_agent_array(results, agent, metric_name)
        avg = np.mean(values)
        std = np.std(np.mean(values, axis=1))
        f.write(agent + ', mean: ' + str(avg) + "; std: "+str(std))
        f.write("\n")
    f.close()



def plot_boxplot_time(results,
                      save_path,
                      metric_name='time (ms)'):
    metric_per_seed = (
    results
    .groupby(["agent", "seed"], as_index=False, sort=False)[metric_name]
    .sum()
    )
    groups = [g[metric_name].values for _, g in metric_per_seed.groupby("agent")]
    names = results["agent"].unique()
    plt.figure(figsize=(max(8, len(names)*1.8), 6))
    clean_names = [labels[name] for name in names]
    plt.boxplot(groups, labels=clean_names, showfliers=False)
    plt.xticks(range(1, len(names) + 1), clean_names, fontsize=16-len(names))
    plt.ylabel('Time (ms)')
    plt.savefig(os.path.join(save_path, 'Total_times.pdf'))
    plt.close()

# def plot_avg_over_time(dic_of_values,
#                        nb_steps,
#                        nb_trials,
#                        change_rate,
#                        nb_iters,
#                        title='',
#                        xlabel='Number of task changes',
#                        ylabel='Rewards',
#                        legend=True,
#                        ax=None,
#                        save=True,
#                        multiply=False,
#                        grid=False):
#     if ax is None:
#         fig, ax = plt.subplots()
#     all_means = {}
#     if change_rate % nb_steps != 0:
#         print("Cannot plot, change_rate % nb_steps != 0.""")
#         return None
#     trials_each_change = int(change_rate//nb_steps)
#     nb_values = nb_iters*trials_each_change
#     nb_changes = nb_trials//trials_each_change
#     nb_values = nb_iters
#     for agent in dic_of_values.keys():
#         all_means[agent] = np.zeros((nb_iters, nb_changes))
#         values = np.array(dic_of_values[agent])
#         for i in range(nb_changes):
#             starting_i = i*trials_each_change
#             r_change = values[:, starting_i:starting_i+trials_each_change]
#             if multiply:
#                 r_change *= 1e3/nb_steps
#             mean = np.mean(r_change, axis=1)
#             all_means[agent][:, i] = mean

#         array_mean = all_means[agent]
#         array_std = np.std(array_mean, axis=0)
#         array_mean = np.mean(array_mean, axis=0)
#         array_CI = 1.96*array_std / np.sqrt(nb_values)
#         x_axis = np.arange(0, nb_changes)
#         ax.plot(x_axis,
#                 array_mean,
#                 label=labels[agent],
#                 color=all_colors[colors[agent]],
#                 marker=markers[agent],
#                 markersize=4)
#         ax.fill_between(x_axis,
#                         array_mean-array_CI,
#                         array_mean+array_CI,
#                         color=all_colors[colors[agent]],
#                         alpha=0.15)

#     ax.set_ylabel(ylabel, fontweight='bold')
#     ax.set_xlabel(xlabel, fontweight='bold')
#     # ax.yaxis.get_major_locator().set_params(integer=True)
#     ax.xaxis.get_major_locator().set_params(integer=True)
#     if legend:
#         ax.legend(loc='lower center')
#     if title == '':
#         title = str(time.time())
#     if grid:
#         ax.grid(alpha=0.2)
#     if save:
#         plt.savefig('results/'+title+'.pdf')
#         plt.close()


# def plot_avg_after_change(dic_of_values,
#                           nb_steps,
#                           nb_trials,
#                           change_rate,
#                           nb_iters,
#                           title='',
#                           xlabel="Number of steps after the task change",
#                           ylabel='Rewards',
#                           legend=True,
#                           ax=None,
#                           save=True,
#                           multiply=False):
#     if ax is None:
#         fig, ax = plt.subplots()
#     all_means = {}
#     trials_each_change = int(change_rate//nb_steps)
#     nb_values = nb_iters
#     for agent in dic_of_values.keys():
#         all_means[agent] = np.zeros((nb_iters, trials_each_change))
#         values = np.array(dic_of_values[agent])
#         not_used = nb_trials % trials_each_change
#         for j in range(trials_each_change):
#             filter_indices = np.array([i+j for i in range(0,
#                                                           nb_trials-not_used,
#                                                           trials_each_change)])

#             r_change = np.take(values, filter_indices, 1)
#             if multiply:
#                 r_change *= 1e3/nb_steps
#             mean = np.mean(r_change, axis=1)
#             all_means[agent][:, j] = mean

#         array_mean = all_means[agent]
#         array_std = np.std(array_mean, axis=0)
#         array_mean = np.mean(array_mean, axis=0)

#         array_CI = 1.96*array_std / np.sqrt(nb_values)
#         x_axis = np.arange(1, trials_each_change+1)
#         ax.plot(x_axis,
#                 array_mean,
#                 label=labels[agent],
#                 color=all_colors[colors[agent]],
#                 marker=markers[agent],
#                 markersize=4)
#         ax.fill_between(x_axis,
#                         array_mean-array_CI,
#                         array_mean+array_CI,
#                         color=all_colors[colors[agent]],
#                         alpha=0.15)

#     ax.set_ylabel(ylabel, fontweight='bold')
#     ax.set_xlabel(xlabel, fontweight='bold')
#     # ax.yaxis.get_major_locator().set_params(integer=True)
#     ax.xaxis.get_major_locator().set_params(integer=True)
#     if legend:
#         ax.legend(loc='lower center')
#     if title == '':
#         title = str(time.time())
#     if save:
#         plt.savefig('results/'+title+'.pdf')
#         plt.close()
