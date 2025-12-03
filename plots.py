from consts import all_colors, colors, labels, markers
from consts import one_step_environments, multi_model_agents
from consts import mM_and_RLCD
from matplotlib.ticker import MaxNLocator
import time
import numpy as np
import matplotlib.pyplot as plt
from const_maze import pattern
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------- #
# 1D Plots
# ---------------------------------------------------------------------------- #


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
    #index_plot = np.arange(len(mean))*steps_per_episode
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


def plot_change(change_rate,
                steps):
    nb_changes = int(steps//change_rate)
    for i in range(1, nb_changes):
        plt.axvline(x=change_rate*i,
                    linestyle='--',
                    color='black',
                    alpha=0.1)


def plot_legend(results):
    names = list(results.keys())

    fig = plt.figure(figsize=(2, 1.25))
    patches = []
    agent_names = []
    for name in names:
        color = all_colors[colors[name]]
        label = labels[name]
        agent_names.append(label)
        patches.append(matplotlib.patches.Patch(color=color, label=label))
    fig.legend(patches, labels, loc='center', frameon=False)
    plt.savefig('results/Legend'+str(time.time())+'.pdf', bbox_inches='tight')
    plt.close()

def plot_curves(results,
                change_rate,
                steps,
                nb_iters,
                ylabel='Reward',
                xlabel='Steps',
                title='',
                legend=True,
                multiply=False,
                suptitle='Uncertain variation'):
    """Two-subplot version:
       - left: only the legend (bold)
       - right: the curves and change lines, titled 'rewards' (bold), no legend
    """
    fig= plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax_left.plot([0, 1], [0, 1], alpha=0)     # invisible dummy plot
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    # ax_left.set_xticks([])
    # ax_left.set_yticks([])

    ax_left.tick_params(axis='x', colors='white')   # x-tick labels invisible
    ax_left.tick_params(axis='y', colors='white')   # y-tick labels invisible
    ax_left.spines['bottom'].set_color('white')   
    ax_left.spines['left'].set_color('white')       
    ax_left.spines['right'].set_color('white')
    ax_left.spines['top'].set_color('white')

    names = list(results.keys())
    counter = 0

    # Plot everything on the right subplot
    for name in names:
        res_to_plot = np.array(results[name])
        if multiply:
            res_to_plot *= 1e3 / steps

        counter += 1
        if counter == 1:
            total_steps = len(res_to_plot[0]) * steps
            # ensure the change plot goes to the right axis
            plt.sca(ax_right)
            plot_change(change_rate, total_steps)

        # plot_with_CI likely uses the current axes (plt); force it to the right axes
        plt.sca(ax_right)
        plot_with_CI(res_to_plot,
                     steps,
                     nb_iters,
                     save=False,
                     color=all_colors[colors[name]],
                     label=labels[name],
                     marker=markers[name],
                     markersize=4)

    # Configure right subplot (the actual plot)
    ax_right.set_xlabel(xlabel, fontweight='bold')
    ax_right.set_ylabel(ylabel, fontweight='bold')

    # Collect handles/labels from the right axis (these come from plot_with_CI)
    handles, labs = ax_right.get_legend_handles_labels()

    # Remove any legend from the right subplot (we'll put it on the left)
    right_leg = ax_right.get_legend()
    if right_leg is not None:
        right_leg.remove()

    # Left subplot: only show the legend (centered)
    if legend and len(handles) > 0:
        # Put legend into the left axis centered. Make labels bold via prop.
        legend_props = {'weight': 'bold', 'size': 20}
        ax_left.legend(handles, labs, loc='center', prop=legend_props)
    else:
        # If no legend requested, keep left axis empty
        ax_left.set_visible(False)

    # Turn off axes ticks/lines for left subplot so only the legend shows
    # ax_left.axis('off')
    ax_left.set_frame_on(True)         # keeps a frame box (invisible)
    ax_left.patch.set_alpha(0)

    ax_left.set_xlabel(" x-axis ", alpha=0, fontweight='bold')
    ax_left.set_ylabel(" y-axis ", alpha=0, fontweight='bold')
    #ax_left.set_title(" Legend ", pad=25, size=14, fontweight='bold',alpha=0)

    # Right subplot title in bold
    ax_left.set_title(suptitle, fontweight='bold', size=22, pad=25, color='black')
    # Save and close
    if title == '':
        title = str(time.time())
    #fig.suptitle(suptitle, fontweight='bold', size=14, x=0.75)
    #plt.tight_layout()
    
    plt.savefig('results/' + ylabel + title + '.pdf', bbox_inches='tight')
    plt.close()



def plot_time(times,
              title_fig='Average time in seconds for each agent',
              title=''):
    value_label, data = [*zip(*times.items())]
    plt.boxplot(data)
    names = []
    for key in value_label:
        name = labels[key]
        names.append(name)
    plt.xticks(range(1, len(names) + 1), names, fontsize=16-2*len(names))
    plt.ylabel('Time (s)')
    plt.title(title_fig)
    if title == '':
        title = str(time.time())
    plt.savefig('results/total_times'+title+'.pdf')
    plt.close()


# ---------------------------------------------------------------------------- #
# Plots average after change
# ---------------------------------------------------------------------------- #

def plot_two(dic_of_rewards,
             nb_steps,
             nb_trials,
             change_rate,
             nb_iters,
             title,
             xlabel,
             ylabel,
             legend=False,
             multiply=False,
             suptitle='Uncertain-Volatile variation'):

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    plot_avg_after_change(dic_of_rewards,
                          nb_steps,
                          nb_trials,
                          change_rate,
                          nb_iters,
                          title=title,
                          legend=False,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          ax=axs[1],
                          save=False,
                          multiply=multiply)

    plot_avg_over_time(dic_of_rewards,
                       nb_steps,
                       nb_trials,
                       change_rate,
                       nb_iters,
                       title=title,
                       ylabel=ylabel,
                       legend=False,
                       ax=axs[0],
                       save=False,
                       multiply=multiply)

    if legend : 
        handles, labels = axs[0].get_legend_handles_labels()
        fig_legend = plt.figure(figsize=(8, 2))
        fig_legend.legend(handles, labels, loc='center', ncol=2,
                        prop={"weight": "bold", "size": 14})
        fig_legend.tight_layout()
        fig_legend.savefig(
            "results/legend_"+title+".pdf", bbox_inches='tight')
        plt.close(fig_legend)

    axs[0].set_title(suptitle, fontweight='bold', size=22, pad=25)
    plt.savefig('results/'+title+'.pdf', bbox_inches='tight')
    plt.close()


def plot_four_models(models,
                     models_created,
                     models_merged,
                     models_forgotten,
                     nb_steps,
                     nb_trials,
                     change_rate,
                     nb_iters,
                     title):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    plot_avg_over_time(models,
                       nb_steps,
                       nb_trials,
                       change_rate,
                       nb_iters,
                       title=title,
                       legend=False,
                       ax=axs[0, 0],
                       save=False,
                       ylabel='Total number of models',
                       grid=True
                       )

    plot_avg_over_time(models_created,
                       nb_steps,
                       nb_trials,
                       change_rate,
                       nb_iters,
                       title=title,
                       ylabel='Number of models created',
                       legend=False,
                       ax=axs[0, 1],
                       save=False, grid=True)

    plot_avg_over_time(models_merged,
                       nb_steps,
                       nb_trials,
                       change_rate,
                       nb_iters,
                       title=title,
                       ylabel='Number of models merged',
                       legend=False,
                       ax=axs[1, 0],
                       save=False, grid=True)

    plot_avg_over_time(models_forgotten,
                       nb_steps,
                       nb_trials,
                       change_rate,
                       nb_iters,
                       title=title,
                       ylabel='Number of models forgotten',
                       legend=False,
                       ax=axs[1, 1],
                       save=False, grid=True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # Create a single legend below all subplots
    odd = len(labels) % 2
    fig.legend(handles, labels, loc="lower center", ncol=2+odd, fontsize=12)

    # Adjust layout to make space for the legend

    bottom_adjust = 0.1+0.03*(len(labels)//2)
    # Adjust layout to make space for the legend
    plt.subplots_adjust(left=0.1, right=0.9,
                        top=0.9, bottom=bottom_adjust,
                        wspace=0.3, hspace=0.3)
    # fig.tight_layout(pad=5.0)
    plt.savefig('results/sum_up_models'+title+'.pdf', bbox_inches='tight')
    plt.close()


def plot_four(dic_of_rewards,
              dic_of_times,
              nb_steps,
              nb_trials,
              change_rate,
              nb_iters,
              title,
              xlabel,
              ylabel_res):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plot_avg_after_change(dic_of_rewards,
                          nb_steps,
                          nb_trials,
                          change_rate,
                          nb_iters,
                          title=title,
                          legend=False,
                          xlabel=xlabel,
                          ylabel=ylabel_res,
                          ax=axs[0, 1],
                          save=False)

    plot_avg_over_time(dic_of_rewards,
                       nb_steps,
                       nb_trials,
                       change_rate,
                       nb_iters,
                       title='Reward_over_time'+title,
                       legend=False,
                       ax=axs[0, 0],
                       ylabel=ylabel_res,
                       save=False)

    plot_avg_over_time(dic_of_times,
                       nb_steps,
                       nb_trials,
                       change_rate,
                       nb_iters,
                       title="Computational_time_over_time"+title,
                       ylabel='Time per decision (ms)',
                       legend=False,
                       ax=axs[1, 0],
                       save=False,
                       multiply=True)

    plot_avg_after_change(dic_of_times,
                          nb_steps,
                          nb_trials,
                          change_rate,
                          nb_iters,
                          title=title,
                          xlabel=xlabel,
                          ylabel='Time per decision (ms)',
                          legend=False,
                          ax=axs[1, 1],
                          save=False,
                          multiply=True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # Create a single legend below all subplots
    odd = len(labels) % 2
    fig.legend(handles, labels, loc="lower center", ncol=2+odd, fontsize=12)

    # Adjust layout to make space for the legend

    bottom_adjust = 0.1+0.03*(len(labels)//2)
    # Adjust layout to make space for the legend
    plt.subplots_adjust(left=0.1, right=0.9,
                        top=0.9, bottom=bottom_adjust,
                        wspace=0.3, hspace=0.3)
    # fig.tight_layout(pad=5.0)
    plt.savefig('results/perf_sum_up'+title+'.pdf', bbox_inches='tight')
    plt.close()


def plot_avg_after_change(dic_of_values,
                          nb_steps,
                          nb_trials,
                          change_rate,
                          nb_iters,
                          title='',
                          xlabel="Number of steps after the task change",
                          ylabel='Rewards',
                          legend=True,
                          ax=None,
                          save=True,
                          multiply=False):
    if ax is None:
        fig, ax = plt.subplots()
    all_means = {}
    trials_each_change = int(change_rate//nb_steps)
    nb_values = nb_iters
    for agent in dic_of_values.keys():
        all_means[agent] = np.zeros((nb_iters, trials_each_change))
        values = np.array(dic_of_values[agent])
        not_used = nb_trials % trials_each_change
        for j in range(trials_each_change):
            filter_indices = np.array([i+j for i in range(0,
                                                          nb_trials-not_used,
                                                          trials_each_change)])

            r_change = np.take(values, filter_indices, 1)
            if multiply:
                r_change *= 1e3/nb_steps
            mean = np.mean(r_change, axis=1)
            all_means[agent][:, j] = mean

        array_mean = all_means[agent]
        array_std = np.std(array_mean, axis=0)
        array_mean = np.mean(array_mean, axis=0)

        array_CI = 1.96*array_std / np.sqrt(nb_values)
        x_axis = np.arange(1, trials_each_change+1)
        ax.plot(x_axis,
                array_mean,
                label=labels[agent],
                color=all_colors[colors[agent]],
                marker=markers[agent],
                markersize=4)
        ax.fill_between(x_axis,
                        array_mean-array_CI,
                        array_mean+array_CI,
                        color=all_colors[colors[agent]],
                        alpha=0.15)

    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold')
    # ax.yaxis.get_major_locator().set_params(integer=True)
    ax.xaxis.get_major_locator().set_params(integer=True)
    if legend:
        ax.legend(loc='lower center')
    if title == '':
        title = str(time.time())
    if save:
        plt.savefig('results/'+title+'.pdf')
        plt.close()


def plot_avg_over_time(dic_of_values,
                       nb_steps,
                       nb_trials,
                       change_rate,
                       nb_iters,
                       title='',
                       xlabel='Number of task changes',
                       ylabel='Rewards',
                       legend=True,
                       ax=None,
                       save=True,
                       multiply=False,
                       grid=False):
    if ax is None:
        fig, ax = plt.subplots()
    all_means = {}
    if change_rate % nb_steps != 0:
        print("Cannot plot, change_rate % nb_steps != 0.""")
        return None
    trials_each_change = int(change_rate//nb_steps)
    nb_values = nb_iters*trials_each_change
    nb_changes = nb_trials//trials_each_change
    nb_values = nb_iters
    for agent in dic_of_values.keys():
        all_means[agent] = np.zeros((nb_iters, nb_changes))
        values = np.array(dic_of_values[agent])
        for i in range(nb_changes):
            starting_i = i*trials_each_change
            r_change = values[:, starting_i:starting_i+trials_each_change]
            if multiply:
                r_change *= 1e3/nb_steps
            mean = np.mean(r_change, axis=1)
            all_means[agent][:, i] = mean

        array_mean = all_means[agent]
        array_std = np.std(array_mean, axis=0)
        array_mean = np.mean(array_mean, axis=0)
        array_CI = 1.96*array_std / np.sqrt(nb_values)
        x_axis = np.arange(0, nb_changes)
        ax.plot(x_axis,
                array_mean,
                label=labels[agent],
                color=all_colors[colors[agent]],
                marker=markers[agent],
                markersize=4)
        ax.fill_between(x_axis,
                        array_mean-array_CI,
                        array_mean+array_CI,
                        color=all_colors[colors[agent]],
                        alpha=0.15)

    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold')
    # ax.yaxis.get_major_locator().set_params(integer=True)
    ax.xaxis.get_major_locator().set_params(integer=True)
    if legend:
        ax.legend(loc='lower center')
    if title == '':
        title = str(time.time())
    if grid:
        ax.grid(alpha=0.2)
    if save:
        plt.savefig('results/'+title+'.pdf')
        plt.close()


def general_performance(all_rewards, title):
    path = 'results/general_info '+title+'.txt'
    f = open(path, "w")
    for agent in all_rewards.keys():
        avg = np.mean(all_rewards[agent])
        std = np.std(np.mean(all_rewards[agent], axis=1))
        f.write(agent + ', mean: ' + str(avg) + "; std: "+str(std))
        f.write("\n")
    f.close()


def get_all_plot(results, parameters, legend=True, suptitle='Uncertain variation'):
    # Get all the information
    agents_tested = parameters['agents']
    env_tested = parameters['env_name']
    nb_iters = parameters['nb_iters']*len(parameters['env_param'])
    trials = parameters['trials']
    steps = parameters['max_step']
    title = parameters['time']
    change_rate = parameters['env_param'][0]['step_change']
    if change_rate is None:
        change_rate = steps*trials

    env_is_one_step = env_tested in one_step_environments

    all_rewards = {}
    all_times = {}
    all_total_times = {}
    all_models = {}
    models_created_per_cell = {}
    models_per_cell = {}
    all_current_distance = {}
    all_best_actions = {}

    all_changes = {}

    all_current_models = {}
    all_models_created = {}
    all_models_merged = {}
    all_models_forgotten = {}

    for agent in agents_tested:
        if agent in multi_model_agents:
            all_models[agent] = {'nb_model': [],
                                 'nb_creation': [],
                                 'nb_forgetting': [],
                                 'nb_merging': []}
            if env_tested:
                models_created_per_cell[agent] = []
                models_per_cell[agent] = []
        if agent in mM_and_RLCD:
            all_changes[agent] = []

    for agent in agents_tested:
        all_rewards[agent] = []
        all_times[agent] = []
        all_total_times[agent] = []

        # all_distance[agent] = []
        all_current_distance[agent] = []

        if agent in multi_model_agents:
            all_current_models[agent] = []
            all_models_created[agent] = []
            all_models_forgotten[agent] = []
            all_models_merged[agent] = []

        if env_is_one_step:
            all_best_actions[agent] = []

    for info_exp, all_values in results.items():
        agent_name = info_exp[1]
        all_rewards[agent_name].append(all_values["reward"])
        all_times[agent_name].append(all_values["times"])
        all_total_times[agent_name].append(all_values["total_time"])

        # all_distance[agent_name].append(all_values["distance_model"])
        all_current_distance[agent_name].append(
            all_values["distance_current_model"])

        if agent_name in multi_model_agents:
            all_current_models[agent_name].append(all_values['nb_model'])
            all_models_created[agent_name].append(all_values['nb_creation'])
            all_models_forgotten[agent_name].append(
                all_values['nb_forgetting'])
            all_models_merged[agent_name].append(all_values['nb_merging'])

            for key_model in all_models[agent_name].keys():
                all_models[agent_name][key_model].append(all_values[key_model])

            if env_tested in ["ChangingCrossEnvironment",
                              "PartiallyChangingCrossEnvironment"]:
                models_created_per_cell[agent_name].append(
                    all_values['creation_per_state'])
                models_per_cell[agent_name].append(
                    all_values['model_per_state'])
            
        if agent_name in mM_and_RLCD:
            all_changes[agent_name].append(all_values['all_changes'])

        if env_is_one_step:
            all_best_actions[agent_name].append(all_values["best_action"])

    # plot_curves(all_rewards,
    #             change_rate=change_rate,
    #             nb_iters=nb_iters,
    #             steps=steps,
    #             title=title)

    # plot_legend(all_rewards)

    if env_is_one_step:
        event = 'steps'
        actions_or_rewards = all_best_actions
        spec_ylabel = 'Probability of selecting the best action'
    else:
        event = 'trials'
        actions_or_rewards = all_rewards
        spec_ylabel = 'Rewards'

    for no_legend in [True]:
        plot_curves(actions_or_rewards,
                    change_rate=change_rate,
                    nb_iters=nb_iters,
                    steps=steps,
                    title="Rewards"+title+str(no_legend),
                    legend=no_legend,
                    ylabel=spec_ylabel,
                    xlabel = 'Number of '+event,
                    suptitle=suptitle)

        plot_curves(all_times,
                    change_rate=change_rate,
                    nb_iters=nb_iters,
                    steps=steps,
                    title="Times"+title+str(no_legend),
                    legend=no_legend,
                    ylabel="Time per decision (ms)",
                    xlabel = 'Number of '+event,
                    multiply=True,
                    suptitle=suptitle)

    plot_time(all_total_times, title=title)


    for agent in agents_tested:
        if agent in multi_model_agents:
            # print(all_models[agent].keys())
            plot_curves(all_models[agent],
                        nb_iters=nb_iters,
                        change_rate=change_rate,
                        steps=steps,
                        ylabel='Number of models',
                        title=agent+title,
                        legend=legend,
                        suptitle=suptitle)
            # plot_curves(all_changes[agent],
            #             nb_iters=nb_iters,
            #             change_rate=change_rate,
            #             steps=steps,
            #             ylabel='Time of detected change',
            #             title=agent+title,
            #             legend=legend,
            #             suptitle=suptitle)
            one_env = len(parameters['env_param']) == 1
            if env_tested in ["ChangingCrossEnvironment",
                              "PartiallyChangingCrossEnvironment"] and one_env:
                number_env = parameters['env_param'][0]['number']
                mean_creation = np.mean(models_created_per_cell[agent], axis=0)
                mean_mod = np.mean(models_per_cell[agent], axis=0)
                round_creation = np.round(mean_creation, 1)
                round_mod = np.round(mean_mod, 1)
                plot_number_models_cross_env(number_env,
                                             round_creation,
                                             title=agent+title,
                                             title_fig='models_created')
                plot_number_models_cross_env(number_env,
                                             round_mod,
                                             title=agent+title,
                                             title_fig='models')
    if len(all_current_models.keys()) > 0:
        plot_avg_over_time(all_current_models,
                           steps,
                           trials,
                           change_rate,
                           nb_iters,
                           ylabel='Number of models',
                           title="Number of models"+title,
                           legend=legend)

    xlabel = 'Number of ' + event + ' after the task change'

    general_performance(all_rewards, title)

    plot_four(actions_or_rewards,
              all_times,
              steps,
              trials,
              change_rate,
              nb_iters,
              title,
              xlabel=xlabel,
              ylabel_res=spec_ylabel)

    if len(all_current_models.keys()) > 0:
        plot_four_models(all_current_models,
                         all_models_created,
                         all_models_merged,
                         all_models_forgotten,
                         steps,
                         trials,
                         change_rate,
                         nb_iters,
                         title)
        
    plot_two(actions_or_rewards,
                steps,
                trials,
                change_rate,
                nb_iters,
                title='reward_sum_up'+title+str(no_legend),
                xlabel=xlabel,
                ylabel=spec_ylabel,
                legend=True,
                suptitle=suptitle)

    plot_two(all_current_distance,
                steps,
                trials,
                change_rate,
                nb_iters,
                ylabel='Euclidean distance',
                title='distance_sum_up'+title+str(no_legend),
                xlabel=xlabel,
                legend=False,
                suptitle=suptitle)

    plot_two(all_times,
                steps,
                trials,
                change_rate,
                nb_iters,
                ylabel='Time per decision (ms)',
                title='time_sum_up'+title+str(no_legend),
                xlabel=xlabel,
                legend=False,
                multiply=True,
                suptitle=suptitle)
        
    if agent in mM_and_RLCD :
        
        plot_two(all_changes,
                steps,
                trials,
                change_rate,
                nb_iters,
                ylabel='Number of changes',
                title='changes'+title+str(no_legend),
                xlabel=xlabel,
                legend=no_legend,
                multiply=False,
                suptitle=suptitle)


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
