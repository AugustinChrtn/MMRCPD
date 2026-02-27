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

# ---------------------------------------------------------------------------- #
# Maze Plots
# ---------------------------------------------------------------------------- #


def plot_maze(world,
              path,
              labels=np.empty(0),
              arrows=np.empty(0),
              uncertain=np.empty(0),
              blue_circle=False):

    reward = world[world > 0][0]

    size = np.shape(world)
    black = [0, 0, 0]
    blue = [0.14, 0.48, 1]
    red = [1, 0, 0]
    yellow = [218/255,165/255,32/255]
    size_colors = size+tuple([3])
    array_of_colors = np.ones(size_colors)
    init_state = world == -2

    walls = world == -1

    pattern_array = np.zeros(np.shape(init_state), dtype='bool')
    for key, value in pattern.items():
        if value == 0:
            pattern_array[key] = True

    if len(uncertain) > 0:
        pattern_array = arrows != uncertain
        pattern_array[walls] = False

    array_of_colors[pattern_array] = yellow
    array_of_colors[init_state] = blue
    array_of_colors[walls] = black
    reward_matrix = world == reward
    array_of_colors[reward_matrix] = red

    _, ax = plt.subplots(1, 1, dpi=100)
    # adding colors
    ax.imshow(array_of_colors, aspect='equal')
    if len(labels) == 0 and len(arrows) == 0:
        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                if reward_matrix[j, i]:
                    ax.text(i, j, "R", va='center', ha='center')
                if init_state[j, i]:
                    ax.text(i, j, "I", va='center', ha='center')

    if len(arrows) != 0:
        for i in range(arrows.shape[0]):
            for j in range(arrows.shape[1]):
                if walls[i, j] ==0:
                    action = arrows[i, j]
                    if len(uncertain) > 0 and pattern_array[i, j]:
                        action = uncertain[i, j]
                    if action == 4:  # Draw a circle
                        circle = plt.Circle((j, i), 0.1,
                                            color='black', fill=True)
                        ax.add_patch(circle)
                    elif action == 0:  # Up arrow (↑)
                        ax.arrow(j, i+0.1, 0, -0.1, head_width=0.1,
                                 head_length=0.1, fc='black', ec='black')
                    elif action == 1:  # Down arrow (↓)
                        ax.arrow(j, i-0.1, 0, 0.1, head_width=0.1,
                                 head_length=0.1, fc='black', ec='black')
                    elif action == 2:  # Left arrow (←)
                        ax.arrow(j, i, -0.1, 0, head_width=0.1,
                                 head_length=0.1, fc='black', ec='black')
                    elif action == 3:  # Right arrow (→)
                        ax.arrow(j, i, 0.1, 0, head_width=0.1,
                                 head_length=0.1, fc='black', ec='black')
                    
                    if blue_circle :
                        circle = plt.Circle((j-0.4, i-0.4), 0.05,
                                            color='blue', fill=True)
                        ax.add_patch(circle)

    # adding labels
    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            c = labels[j, i]
            ax.text(i, j, str(c), va='center', ha='center')
    major_ticks = np.arange(-0.5, size[0] + 0.5)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid(True, alpha=1, color='black', linewidth=1)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_one_transition(world_number,
                        col,
                        row,
                        action,
                        cond=''):
    str_world = str(world_number)
    str_cell = str(col)+str(row)
    str_action = str(action)
    world = np.load('Env/Tables/World_'+str_world+'.npy')
    transitions = np.load('Env/Transitions/Transitions_'+str_world+cond+'.npy')
    transi_action = transitions[row][col][action]
    tmp_path = '_action_'+str_action+'_cond'+cond+'.pdf'
    path_save = 'Env/world_'+str_world+'_cell_'+str_cell+tmp_path

    walls = world == -1
    numbers = [0, 1, -1]
    pairs = [(x, y) for x in numbers for y in numbers]
    nine_walls = np.zeros((3, 3))
    nine_probas = np.zeros((3, 3))
    max_col = len(transitions)
    max_row = len(transitions[0])
    for (x, y) in pairs:
        new_row = row+x
        new_col = col+y
        cond_x = new_row >= max_row or new_row < 0
        cond_y = new_col >= max_col or new_col < 0
        if cond_x or cond_y:
            nine_walls[x+1, y+1] = 1
        else:
            nine_walls[x+1, y+1] = walls[new_row, new_col]
        if nine_walls[x+1, y+1] != 1:
            nine_probas[x+1, y+1] = transi_action[new_row, new_col]

    fig, ax = plt.subplots(figsize=(5, 5))

    # Loop through all cells and fill them accordingly
    for i in range(nine_probas.shape[0]):
        for j in range(nine_probas.shape[1]):
            color = 'black' if nine_walls[i, j] == 1 else 'white'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            # Add text only in white cells
            if nine_walls[i, j] == 0:
                percent = round(nine_probas[i, j] * 100, 1)
                if percent > 0:  # Only display if greater than 0
                    ax.text(j + 0.5, i + 0.5, f"{percent}%",
                            ha='center', va='center', color='black')

    if action == 4:  # Draw a circle
        circle = plt.Circle((1.5, 1.7), 0.1, color='blue', fill=True)
        ax.add_patch(circle)
    elif action == 0:  # Up arrow (↑)
        ax.arrow(1.5, 1.3, 0, -0.1, head_width=0.1,
                 head_length=0.1, fc='blue', ec='blue')
    elif action == 1:  # Down arrow (↓)
        ax.arrow(1.5, 1.7, 0, 0.1, head_width=0.1,
                 head_length=0.1, fc='blue', ec='blue')
    elif action == 2:  # Left arrow (←)
        ax.arrow(1.6, 1.3, -0.1, 0, head_width=0.1,
                 head_length=0.1, fc='blue', ec='blue')
    elif action == 3:  # Right arrow (→)
        ax.arrow(1.4, 1.3, 0.1, 0, head_width=0.1,
                 head_length=0.1, fc='blue', ec='blue')

    # Set limits and aspect
    ax.set_xlim(0, nine_probas.shape[1])
    ax.set_ylim(nine_probas.shape[0], 0)
    ax.set_xticks(np.arange(nine_probas.shape[1] + 1))
    ax.set_yticks(np.arange(nine_probas.shape[0] + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="black", linewidth=1.5)
    ax.tick_params(which="both", bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()


def plot_number_models_cross_env(cross_env_number,
                                 models,
                                 title='',
                                 title_fig=''):
    if title == '':
        title = str(time.time())
    world_array = np.load('Env/Tables/World_'+str(cross_env_number)+'.npy')
    shape_env = np.shape(world_array)
    models = np.reshape(models, shape_env)
    path = 'results/'+title_fig+title+'.pdf'
    plot_maze(world_array, path, models)