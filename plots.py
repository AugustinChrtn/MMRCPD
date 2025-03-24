from const_grid import pattern
from check_significance import get_stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import os
import glob
from consts import all_colors, colors, labels, smoothing_factors, multi_model_agents

# ---------------------------------------------------------------------------- #
# 1D Plots
# ---------------------------------------------------------------------------- #


def get_moving_avg(array, avg_size):
    if avg_size > len(array):
        return list(np.arange(len(array))), array
    else:
        index_avg = [i * len(array) // avg_size for i in range(avg_size)]
        index_plot = [(i+1/2) * len(array) / avg_size for i in range(avg_size)]
        moving_avg = [np.mean(array[index_avg[index]:index_avg[index + 1]])
                      for index in range(len(index_avg)-1)]
        moving_avg.append(np.mean(array[index_avg[len(index_avg)-1]:]))
        return index_plot, moving_avg


def basic_plot(rewards,
               steps_per_episode,
               save=True,
               avg_size=100,
               color='tab:blue',
               label='Reward'):

    index_plot, moving_avg = get_moving_avg(rewards, avg_size=avg_size)
    # plt.grid()
    index_plot = np.array(index_plot)*steps_per_episode
    plt.plot(index_plot,
             moving_avg,
             color=color,
             linewidth=3,
             label=label)
    if save:
        plt.savefig('plots/results'+str(time.time())+'.png')


def plot_with_CI(rewards,
                 steps_per_episode,
                 nb_iters,
                 change_rate,
                 save=True,
                 avg_size=100,
                 color='tab:blue',
                 label='Reward'):
    mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0)
    index_plot, moving_avg = get_moving_avg(mean, avg_size=avg_size)
    index_plot = np.array(index_plot)*steps_per_episode
    _, moving_std = get_moving_avg(std, avg_size=avg_size)

    moving_avg = np.array(moving_avg)
    moving_std = np.array(moving_std)
    n_values = nb_iters
    moving_CI = 1.96*moving_std/np.sqrt(n_values)
    yerr0 = moving_avg - moving_CI
    yerr1 = moving_avg + moving_CI

    plt.plot(index_plot,
             moving_avg,
             color=color,
             linewidth=1.5,
             label=label)
    plt.fill_between(index_plot,
                     yerr0,
                     yerr1,
                     color=color,
                     alpha=0.15)
    if save:
        plt.savefig('plots/results'+str(time.time())+'.png')


def get_task_change_ind(array,
                        change_rate,
                        steps_per_episode):
    nb_changes = int(len(array)*steps_per_episode/change_rate)
    shift = (len(array)*steps_per_episode % change_rate)//steps_per_episode
    return nb_changes, shift


def plot_change(change_rate,
                steps):
    nb_changes = int(steps//change_rate)
    for i in range(1, nb_changes):
        plt.axvline(x=change_rate*i,
                    linestyle='--',
                    color='black',
                    alpha=0.1)


def plot_all(results,
             to_plot,
             change_rate,
             steps,
             ylabel='Reward',
             xlabel='Steps',
             path='',
             legend=True):

    counter = 0
    for info in to_plot:

        res_to_plot = results[info]
        counter += 1
        if counter == 1:
            total_steps = len(res_to_plot)*steps
            plot_change(change_rate, total_steps)

        basic_plot(res_to_plot,
                   steps,
                   save=False,
                   avg_size=smoothing_factors[info],
                   color=all_colors[colors[info]],
                   label=labels[info])

    if legend:
        plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if path == '':
        plt.savefig('plots/results'+str(time.time())+'.png')
    else:
        plt.savefig(path)
    plt.close()


def plot_stats_multi(results,
                     change_rate,
                     steps):

    name_agent = results['name_agent']
    nb = str(results['number'])
    path = 'data/'+name_agent+'/plots/models'+nb+'.pdf'
    to_plot = ['nb_merging', 'nb_creation', 'nb_model', 'nb_forgetting']

    plot_all(results,
             to_plot,
             change_rate,
             steps,
             ylabel='Number of models',
             xlabel='Steps',
             path=path)


def plot_rewards(results,
                 change_rate,
                 steps,
                 to_plot=['reward']):
    name_agent = results['name_agent']
    nb = str(results['number'])
    path = 'data/'+name_agent+'/plots/rewards'+nb+'.pdf'

    to_plot = ['reward']

    plot_all(results,
             to_plot,
             change_rate,
             steps,
             ylabel='Rewards',
             xlabel='Steps',
             path=path)


def plot_curves(results,
                change_rate,
                steps,
                nb_iters,
                ylabel='Reward',
                xlabel='Steps',
                title='',
                legend=True):
    """Plot the results"""

    plt.figure(dpi=300)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.grid(linestyle='--')

    names = list(results.keys())
    counter = 0
    for name in names:

        res_to_plot = np.array(results[name])
        counter += 1
        if counter == 1:
            total_steps = len(res_to_plot[0])*steps
            plot_change(change_rate, total_steps)

        plot_with_CI(res_to_plot,
                     steps,
                     nb_iters,
                     change_rate,
                     save=False,
                     avg_size=smoothing_factors[name],
                     color=all_colors[colors[name]],
                     label=labels[name])
        # plot_change(change_rate, total_steps)

    if legend:
        plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if title == '':
        title = str(time.time())
    plt.savefig('results/'+ylabel+title+'.pdf')
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

# def plot_non_stat(rewards,
#                   change_rate,
#                   steps,
#                   nb_models=[],
#                   avg_size=100,
#                   ylabel='Reward',
#                   xlabel='Trial'):
#     basic_plot(rewards,
#                save=False,
#                avg_size=avg_size,
#                color='tab:blue')
#     if nb_models != []:
#         basic_plot(nb_models,
#                    save=False,
#                    avg_size=avg_size,
#                    legend='nb_models',
#                    ylabel=ylabel,
#                    xlabel=xlabel,
#                    color='tab:red')
#     nb_changes = int(len(rewards)*steps/change_rate)
#     left = (len(rewards)*steps % change_rate)/steps
#     for i in range(nb_changes-1):
#         plt.axvline(x=(len(rewards)-int(left))*(i+1)/nb_changes,
#                     linestyle='--',
#                     color='black',
#                     alpha=0.5)
#     plt.legend()
#     plt.savefig('plots/results'+str(time.time())+'.png')
#     plt.close()


# def double_plot(rewards1, rewards2):
#     index_avg = [i * len(rewards1) // 200 for i in range(200)]
#     moving_average_rewards = [np.mean(rewards1[index_avg[index]:index_avg[index + 1]])
#                               for index in range(len(index_avg) - 1)]
#     moving_average_rewards2 = [np.mean(rewards2[index_avg[index]:index_avg[index + 1]])
#                                for index in range(len(index_avg) - 1)]
#     plt.grid()
#     plt.plot(index_avg[: -1], moving_average_rewards,
#              color='tab:blue', linewidth=3)
#     plt.plot(index_avg[: -1], moving_average_rewards2,
#              color='tab:red', linewidth=3)
#     plt.ylabel('Reward', fontsize=12)
#     plt.xlabel('Trial', fontsize=12)
#     plt.savefig('plots/double_results.png')


# ---------------------------------------------------------------------------- #
# 2D Plots
# ---------------------------------------------------------------------------- #


def get_max_Q_values_and_policy(table):
    best_values = np.max(table, axis=1)
    random_noise = 1e-5 * np.random.random(table.shape)
    best_actions = np.argmax(table + random_noise, axis=1)
    return best_values, best_actions


def plot_2D(table, shape):
    table = np.reshape(table, shape)
    sns.heatmap(table,
                cmap='Blues',
                cbar=False,
                annot=table,
                fmt='.1f',
                annot_kws={"size": 35 / (np.sqrt(len(table)) + 2.5)})


def plot_arrow(i, j, direction):
    rotation = {0: (0.5, 0.9, 0, -0.05),
                1: (0.5, 0.65, 0, 0.05),
                2: (0.65, 0.8, -0.05, 0),
                3: (0.4, 0.8, +0.05, 0)}
    rotation_to_make = rotation[direction]

    plt.arrow(j+rotation_to_make[0],
              i+rotation_to_make[1],
              rotation_to_make[2],
              rotation_to_make[3],
              head_width=0.12,
              head_length=0.12,
              fc='black',
              ec='black',
              linewidth=0.5,
              length_includes_head=False,
              shape='full',
              overhang=0,
              head_starts_at_zero=True)


def plot_V(table, policy_table, shape, path=''):
    """Plot the heatmap of maximal Q-values."""
    plt.figure(figsize=(8, 8), dpi=200)
    table = np.reshape(table, shape)
    policy_table = np.reshape(policy_table, shape)
    plot_2D(table, shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if policy_table[i, j] == 4:
                circle = plt.Circle(
                    (j+0.5, i+0.8), radius=0.07, color='black', fill=True)
                plt.gca().add_patch(circle)
                pass
            else:
                direction = policy_table[i, j]
                plot_arrow(i, j, direction)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal")
    if path != '':
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------------- #
# Plot distributions over states
# ---------------------------------------------------------------------------- #


def plot_all_distrib(env, agent):
    path = 'distrib/*'

    files = glob.glob(path)
    for f in files:
        os.remove(f)
    for state in env.states:
        for action in env.actions:
            existing_mod = agent.find_existing_models(state, action)
            for mod in existing_mod:
                count_passages = agent.all_nSAS[mod, state, action]
                reward = agent.all_Rsum[mod, state, action]
                count = agent.all_nSA[mod, state, action]
                transi = count_passages / count
                r = reward / count

                plot_distrib(transi, count, mod, state, action, r)


def plot_all_distrib_several_models(env, agent, nb_min_distrib=2):
    path = 'distrib/*'

    files = glob.glob(path)
    for f in files:
        os.remove(f)
    for state in env.states:
        for action in env.actions:
            existing_mod = agent.find_existing_models(state, action)
            if len(existing_mod) >= nb_min_distrib:
                for mod in existing_mod:
                    count_passages = agent.all_nSAS[mod, state, action]
                    reward = agent.all_Rsum[mod, state, action]
                    count = agent.all_nSA[mod, state, action]
                    transi = count_passages / count
                    r = reward / count

                    plot_distrib(transi, count, mod, state, action, r)


def plot_distrib_state_action(agent, state, action):
    existing_mod = agent.find_existing_models(state, action)

    for mod in existing_mod:
        print(existing_mod)
        count_passages = agent.all_nSAS[mod, state, action]
        reward = agent.all_Rsum[mod, state, action]
        count = agent.all_nSA[mod, state, action]
        transi = count_passages / count
        r = reward / count
        plot_distrib(transi, count, mod, state, action, r)


def plot_distrib(probabilities, nb_experiences, mod, state, action, reward):

    states = np.arange(len(probabilities))
    states = (probabilities > 0).nonzero()[0]
    probabilities = probabilities[probabilities > 0]
    ind_states = np.arange(len(states))
    # Plotting the histogram
    plt.figure(figsize=(8, 6))
    bars = plt.bar(ind_states, probabilities, tick_label=states)
    plt.title(str(int(nb_experiences))+' steps with action ' +
              str(action) + ' in state '+str(state)+' and model ' +
              str(mod)+'. Expected reward: ' + str(round(reward, 2)))
    for bar, prob in zip(bars, probabilities):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval +
                 0.01, f'{prob:.2f}', ha='center', va='bottom')
    plt.xlabel('Arrival state')
    plt.ylabel('Probability of reaching the state')
    plt.grid(axis='y', linestyle='--')
    ylim = np.max(probabilities)+0.1
    plt.ylim(0, ylim)
    plt.savefig('distrib/probas'+str(state)+str(action)+str(mod)+'.png')
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
             legend=True,
             multiply=False):

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
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

    if legend:
        handles, labels = axs[0].get_legend_handles_labels()
        # Create a single legend below all subplots
        odd = len(labels) % 2
        fig.legend(handles, labels, loc="lower center",
                   ncol=2+odd, fontsize=12)
        bottom_adjust = 0.2+0.03*(len(labels)//2)
        # Adjust layout to make space for the legend
        plt.subplots_adjust(bottom=bottom_adjust)
    else:
        handles, labels = axs[0].get_legend_handles_labels()
        bottom_adjust = 0.2+0.03*(len(labels)//2)
        plt.subplots_adjust(bottom=bottom_adjust)

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
              xlabel):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plot_avg_after_change(dic_of_rewards,
                          nb_steps,
                          nb_trials,
                          change_rate,
                          nb_iters,
                          title=title,
                          legend=False,
                          xlabel=xlabel,
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
                marker='.')
        ax.fill_between(x_axis,
                        array_mean-array_CI,
                        array_mean+array_CI,
                        color=all_colors[colors[agent]],
                        alpha=0.15)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
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
                marker='.')
        ax.fill_between(x_axis,
                        array_mean-array_CI,
                        array_mean+array_CI,
                        color=all_colors[colors[agent]],
                        alpha=0.15)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if legend:
        ax.legend(loc='lower center')
    if title == '':
        title = str(time.time())
    if grid:
        ax.grid(alpha=0.2)
    if save:
        plt.savefig('results/'+title+'.pdf')
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


def general_performance(all_rewards, title):
    path = 'results/general_info '+title+'.txt'
    f = open(path, "w")
    for agent in all_rewards.keys():
        avg = np.mean(all_rewards[agent])
        std = np.std(np.mean(all_rewards[agent], axis=1))
        f.write(agent + ', mean: ' + str(avg) + "; std: "+str(std))
        f.write("\n")
    f.close()


def get_all_plot(results, parameters, legend=True):
    # Get all the information
    agents_tested = parameters['agents']
    env_tested = parameters['env_name']
    nb_iters = parameters['nb_iters']*len(parameters['env_param'])
    trials = parameters['trials']
    steps = parameters['max_step']
    title = parameters['time']
    change_rate = parameters['env_param'][0]['step_change']

    all_rewards = {}
    all_times = {}
    all_total_times = {}
    all_models = {}
    models_created_per_cell = {}
    models_per_cell = {}
    all_distance = {}
    all_current_distance = {}

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

    plot_curves(all_rewards,
                change_rate=change_rate,
                nb_iters=nb_iters,
                steps=steps,
                title=title)
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
                        legend=legend)
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
    # plot_curves(all_distance,
    #             nb_iters=nb_iters,
    #             change_rate=change_rate,
    #             steps=steps,
    #             ylabel='Best Euclidean distance to the true transitions',
    #             title=agent+title,
    #             legend=legend)
    if len(all_current_models.keys()) > 0 :
        plot_avg_over_time(all_current_models,
                        steps,
                        trials,
                        change_rate,
                        nb_iters,
                        ylabel='Number of models',
                        title="Number of models"+title,
                        legend=legend)
    if env_tested in ["ThreeStates", "FourStates", "MAB"] :
        event = 'steps'
    else:
        event = 'trials'
    xlabel = 'Number of ' + event + ' after the task change'

    plot_avg_over_time(all_current_distance,
                       steps,
                       trials,
                       change_rate,
                       nb_iters,
                       ylabel='Euclidean distance',
                       title="Current_distance_over_time"+title,
                       legend=legend)

    plot_avg_over_time(all_current_distance,
                       steps,
                       trials,
                       change_rate,
                       nb_iters,
                       ylabel='Euclidean distance',
                       title="Current_distance_over_time"+title,
                       legend=legend)

    plot_avg_over_time(all_rewards,
                       steps,
                       trials,
                       change_rate,
                       nb_iters,
                       title="Reward_over_time"+title,
                       legend=legend)

    plot_avg_over_time(all_times,
                       steps,
                       trials,
                       change_rate,
                       nb_iters,
                       title="Computational_time_over_time"+title,
                       legend=legend,
                       ylabel="Time per decision (ms)",
                       multiply=True)

    plot_avg_after_change(all_rewards,
                          steps,
                          trials,
                          change_rate,
                          nb_iters,
                          title='Reward_after_change'+title,
                          xlabel=xlabel,
                          legend=legend)

    plot_avg_after_change(all_times,
                          steps,
                          trials,
                          change_rate,
                          nb_iters,
                          title='Computational_time_after_change'+title,
                          xlabel=xlabel,
                          ylabel='Time per decision (ms)',
                          legend=legend,
                          multiply=True)

    general_performance(all_rewards, title)

    plot_four(all_rewards,
              all_times,
              steps,
              trials,
              change_rate,
              nb_iters,
              title,
              xlabel=xlabel)

    if len(all_current_models.keys()) > 0 :
        plot_four_models(all_current_models,
                        all_models_created,
                        all_models_merged,
                        all_models_forgotten,
                        steps,
                        trials,
                        change_rate,
                        nb_iters,
                        title)

    for no_legend in [True, False]:
        plot_two(all_rewards,
                 steps,
                 trials,
                 change_rate,
                 nb_iters,
                 title='reward_sum_up'+title+str(no_legend),
                 xlabel=xlabel,
                 ylabel='Reward',
                 legend=no_legend)

        plot_two(all_current_distance,
                 steps,
                 trials,
                 change_rate,
                 nb_iters,
                 ylabel='Euclidean distance',
                 title='distance_sum_up'+title+str(no_legend),
                 xlabel=xlabel,
                 legend=no_legend)

        plot_two(all_times,
                 steps,
                 trials,
                 change_rate,
                 nb_iters,
                 ylabel='Time per decision (ms)',
                 title='time_sum_up'+title+str(no_legend),
                 xlabel=xlabel,
                 legend=no_legend,
                 multiply=True)


# ---------------------------------------------------------------------------- #
# Maze Plots
# ---------------------------------------------------------------------------- #


def plot_maze(world,
              path,
              labels=np.empty(0),
              arrows=np.empty(0),
              uncertain = np.empty(0)):

    reward = world[world > 0][0]

    size = np.shape(world)
    black = [0, 0, 0]
    blue = [0.14, 0.48, 1]
    red = [1, 0, 0]
    yellow = [1, 1, 0.6]
    size_colors = size+tuple([3])
    array_of_colors = np.ones(size_colors)
    init_state = world == -2

    walls = world == -1

    pattern_array = np.zeros(np.shape(init_state), dtype='bool')
    for key, value in pattern.items():
        if value == 0:
            pattern_array[key] = True

    if len(uncertain) > 0 :
        pattern_array = np.random.choice(a=[False, True], 
                                        size=np.shape(init_state),
                                        p=[0.8,0.2])
        pattern_array[walls]=False

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
                if walls[i, j] != -1:
                    action = arrows[i, j]
                    if len(uncertain) > 0 and pattern_array[i,j]:
                        action = uncertain[i,j]
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
    # adding labels
    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            c = labels[j, i]
            ax.text(i, j, str(c), va='center', ha='center')
    major_ticks = np.arange(-0.5, size[0] +0.5)
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
