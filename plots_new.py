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

def get_plot_from_saved(dir_path, *args):

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

    get_all_plot(results, parameters, final_arrays, dir_path, *args)

# ---------------------------------------------------------------------------- #
# Main plotting function
# ---------------------------------------------------------------------------- #


def get_all_plot(results,
                 parameters,
                 arrays,
                 dir_path,
                 legend=True,
                 suptitle='Uncertain variation'):

    # Creating a plots directory if it does not exist
    d = os.path.join(dir_path, "plots")
    os.makedirs(d, exist_ok=True)
    
    print(results.dtypes)
    numeric_cols = results.select_dtypes(include='number').columns
    stats = results.groupby(['agent', 'episode'])[numeric_cols].agg(
        ['mean', 'std', 'count']).reset_index()

    # stats = results.groupby(['agent', 'episode'])['reward'].agg(
    #     ['mean', 'std', 'count']).reset_index()
    # stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])

    # #print(results)
    # grouped_df = results.groupby(by=["agent", "episode"], as_index=False).agg(['mean', 'std', 'count'])
    # # print(grouped_df)
    # grouped_df = results
    # rewards = grouped_df["reward"]
    # times = grouped_df["time (ms)"]
    # distances = grouped_df["distance"]
    # multi_model = grouped_df[["nb_model", "nb_creation",
    #                          "nb_forgetting", "nb_merging"]]
    # best_action = grouped_df["best_action"]

    # mean = rewards.mean()
    # std = rewards.std()
    # names =  rewards['agent'].unique()

    # change_rate = parameters['env_param'][0]['step_change']
    # nb_iters = parameters['nb_iters']
    # steps = parameters['max_step']

    env_is_one_step = results['environment'][0] in one_step_environments
    if env_is_one_step:
        event = 'steps'
        actions_or_rewards = results['best_action']
        ylabel = 'Probability of selecting the best action'
    else:
        event = 'trials'
        actions_or_rewards = results['reward']
        ylabel = 'Rewards'

    metric_name = 'reward'
    plot_curves(stats,
                metric_name,
                change_rate=parameters['env_param'][0]['step_change'],
                nb_iters=parameters['nb_iters'],
                steps=parameters['max_step'],
                ylabel=ylabel,
                xlabel='Number of ' + event,
                title="Rewards",
                legend=legend,
                suptitle=suptitle,
                save_path=d)

    # print(results)
    # print(results)
    # print(parameters)
    # print(dict(arrays))


def plot_curves(stats,
                metric_name,
                change_rate,
                steps,
                nb_iters,
                ylabel='Reward',
                xlabel='Steps',
                title='',
                legend=True,
                suptitle='Uncertain variation',
                save_path=''):

    fig, ax = plt.subplots(figsize=(10, 5))

    print(stats)
    agents = stats['agent'].unique()
    for agent in agents:
        agent_stats = stats[stats['agent'] == agent]

        x = agent_stats.index
        y = agent_stats[(metric_name,'mean')]
        std = agent_stats[(metric_name,'std')]
        n = agent_stats[(metric_name,'count')]

        ci = 1.96 * std / np.sqrt(n)

        ax.plot(x, y, label=agent)
        ax.fill_between(x, y-ci, y+ci, alpha=0.2)

    # Plot change lines
    nb_changes = int(steps // change_rate)
    for i in range(1, nb_changes):
        plt.axvline(x=change_rate*i, linestyle='--', color='black', alpha=0.1)

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(suptitle, fontweight='bold', size=16)

    if legend:
        ax.legend()

    plt.tight_layout()
    save_name = ylabel+title+'.pdf'
    plt.savefig(os.path.join(save_path, save_name))
    plt.close()

# def plot_curves(results,
#                 change_rate,
#                 steps,
#                 nb_iters,
#                 ylabel='Reward',
#                 xlabel='Steps',
#                 title='',
#                 legend=True,
#                 multiply=False,
#                 suptitle='Uncertain variation'):
#     """Two-subplot version:
#        - left: only the legend (bold)
#        - right: the curves and change lines, titled 'rewards' (bold), no legend
#     """
#     fig = plt.figure(figsize=(14, 5))
#     gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
#     ax_left = fig.add_subplot(gs[0, 0])
#     ax_right = fig.add_subplot(gs[0, 1])

#     ax_left.plot([0, 1], [0, 1], alpha=0)     # invisible dummy plot
#     ax_left.set_xlim(0, 1)
#     ax_left.set_ylim(0, 1)
#     # ax_left.set_xticks([])
#     # ax_left.set_yticks([])

#     ax_left.tick_params(axis='x', colors='white')   # x-tick labels invisible
#     ax_left.tick_params(axis='y', colors='white')   # y-tick labels invisible
#     ax_left.spines['bottom'].set_color('white')
#     ax_left.spines['left'].set_color('white')
#     ax_left.spines['right'].set_color('white')
#     ax_left.spines['top'].set_color('white')

#     counter = 0

#     names =  results['agent'].unique()
#     # Plot everything on the right subplot
#     for name in names:
#         res_to_plot = results[['agent'] == name]
#         if multiply:
#             res_to_plot *= 1e3 / steps

#         counter += 1
#         if counter == 1:
#             total_steps = len(res_to_plot[0])
#             # ensure the change plot goes to the right axis
#             plt.sca(ax_right)
#             plot_change(change_rate, total_steps)

#         # plot_with_CI likely uses the current axes (plt); force it to the right axes
#         plt.sca(ax_right)
#         plot_with_CI(res_to_plot,
#                      steps,
#                      nb_iters,
#                      save=False,
#                      color=all_colors[colors[name]],
#                      label=labels[name],
#                      marker=markers[name],
#                      markersize=4)

#     # Configure right subplot (the actual plot)
#     ax_right.set_xlabel(xlabel, fontweight='bold')
#     ax_right.set_ylabel(ylabel, fontweight='bold')

#     # Collect handles/labels from the right axis (these come from plot_with_CI)
#     handles, labs = ax_right.get_legend_handles_labels()

#     # Remove any legend from the right subplot (we'll put it on the left)
#     right_leg = ax_right.get_legend()
#     if right_leg is not None:
#         right_leg.remove()

#     # Left subplot: only show the legend (centered)
#     if legend and len(handles) > 0:
#         # Put legend into the left axis centered. Make labels bold via prop.
#         legend_props = {'weight': 'bold', 'size': 20}
#         ax_left.legend(handles, labs, loc='center', prop=legend_props)
#     else:
#         # If no legend requested, keep left axis empty
#         ax_left.set_visible(False)

#     # Turn off axes ticks/lines for left subplot so only the legend shows
#     # ax_left.axis('off')
#     ax_left.set_frame_on(True)         # keeps a frame box (invisible)
#     ax_left.patch.set_alpha(0)

#     ax_left.set_xlabel(" x-axis ", alpha=0, fontweight='bold')
#     ax_left.set_ylabel(" y-axis ", alpha=0, fontweight='bold')
#     # ax_left.set_title(" Legend ", pad=25, size=14, fontweight='bold',alpha=0)

#     # Right subplot title in bold
#     ax_left.set_title(suptitle, fontweight='bold',
#                       size=22, pad=25, color='black')
#     # Save and close
#     if title == '':
#         title = str(time.time())
#     # fig.suptitle(suptitle, fontweight='bold', size=14, x=0.75)
#     # plt.tight_layout()

#     plt.savefig('results/' + ylabel + title + '.pdf', bbox_inches='tight')
#     plt.close()


def plot_change(change_rate,
                steps):
    nb_changes = int(steps//change_rate)
    for i in range(1, nb_changes):
        plt.axvline(x=change_rate*i,
                    linestyle='--',
                    color='black',
                    alpha=0.1)


def plot_with_CI(rewards,
                 steps_per_episode,
                 nb_iters,
                 save=True,
                 color='tab:blue',
                 label='Reward',
                 marker='.',
                 markersize=4):
    mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0)
    # index_plot = np.arange(len(mean))*steps_per_episode
    index_plot = np.arange(len(mean))
    n_values = nb_iters
    conf_I = 1.96*std/np.sqrt(n_values)
    yerr0 = mean-conf_I
    yerr1 = mean + conf_I

    plt.plot(index_plot,
             mean,
             color=color,
             linewidth=1.5,
             label=label,
             marker=marker,
             markersize=markersize)
    plt.fill_between(index_plot,
                     yerr0,
                     yerr1,
                     color=color,
                     alpha=0.15)
    if save:
        plt.savefig('plots/results'+str(time.time())+'.png')


# # ---------------------------------------------------------------------------- #
# # Basic 1D plot
# # ---------------------------------------------------------------------------- #

# def _plot_ci(ax, x, data, label, color, marker, markersize):
#     data = np.array(data)

#     mean = data.mean(axis=0)
#     std = data.std(axis=0)
#     ci = 1.96 * std / np.sqrt(len(data))

#     ax.plot(x,
#             mean,
#             label=label,
#             color=color,
#             marker=marker,
#             markersize=markersize)

#     ax.fill_between(x, mean-ci, mean+ci, color=color, alpha=0.15)


# def _add_change_lines(ax, change_rate, total_steps):
#     if change_rate is None:
#         return
#     nb = int(total_steps // change_rate)
#     for i in range(1, nb):
#         ax.axvline(change_rate*i, linestyle="--", color="black", alpha=0.15)

# # =========================================================
# # GENERIC CURVE PLOT
# # =========================================================

# def plot_curves(results,
#                 change_rate,
#                 nb_iters,
#                 steps,
#                 ylabel,
#                 xlabel,
#                 title,
#                 legend=True,
#                 run_dir=None):

#     fig, ax = plt.subplots(figsize=(10,5))


#     for i, (agent, vals) in enumerate(results.items()):
#         vals = np.array(vals)
#         x = np.arange(vals.shape[1])

#         if i == 0:
#             _add_change_lines(ax, change_rate, len(x))

#         _plot_ci(ax,
#                  x,
#                  vals,
#                  labels[agent],
#                  all_colors[colors[agent]],
#                  markers[agent])

#     ax.set_xlabel(xlabel, fontweight="bold")
#     ax.set_ylabel(ylabel, fontweight="bold")

#     if legend:
#         ax.legend()

#     title = str(time.time())

#     save_path = f"{run_dir}_times_{title}.pdf"
#     plt.savefig(save_path)


# # =========================================================
# # BOX PLOT TIMES
# # =========================================================

# def plot_time(times, title, run_dir=None):

#     fig, ax = plt.subplots()

#     names, values = zip(*times.items())
#     ax.boxplot(values)

#     ax.set_xticklabels([labels[n] for n in names])
#     ax.set_ylabel("Time (s)")
#     ax.set_title("Average time per agent")

#     save_path = f"{run_dir}_times_{title}.pdf"
#     plt.savefig(save_path)
