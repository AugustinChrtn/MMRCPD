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
    n_values = len(mean)*nb_iters*steps_per_episode/change_rate
    moving_CI = 3.291*moving_std/np.sqrt(n_values)
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
                    alpha=0.2)


def plot_all(results,
             to_plot,
             change_rate,
             steps,
             ylabel='Reward',
             xlabel='Steps',
             path=''):

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
                title=''):
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
                cmap='crest',
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


def plot_reward_after_change(dic_of_rewards,
                             nb_steps,
                             nb_trials,
                             change_rate,
                             nb_iters,
                             title=''):
    plt.figure(dpi=100)
    all_means = {}
    all_stds = {}
    trials_each_change = int(change_rate//nb_steps)
    nb_values = nb_trials*nb_iters//trials_each_change
    for agent in dic_of_rewards.keys():
        all_means[agent] = []
        all_stds[agent] = []
        rewards = dic_of_rewards[agent]
        not_used = nb_trials % trials_each_change
        for j in range(trials_each_change):
            filter_indices = np.array([i+j for i in range(0,
                                                          nb_trials-not_used,
                                                          trials_each_change)])
            r_change = np.take(rewards, filter_indices, 1).flatten()

            mean = np.mean(r_change)
            std = np.std(r_change)
            all_means[agent].append(mean)
            all_stds[agent].append(std)
        array_mean = np.array(all_means[agent])
        array_std = np.array(all_stds[agent])
        array_CI = 3.291*array_std / np.sqrt(nb_values)
        x_axis = np.arange(1, trials_each_change+1)
        plt.plot(x_axis,
                 array_mean,
                 label=labels[agent],
                 color=all_colors[colors[agent]],
                 marker='.')
        plt.fill_between(x_axis,
                         array_mean-array_CI,
                         array_mean+array_CI,
                         color=all_colors[colors[agent]],
                         alpha=0.15)

    # significance, bigger = np.array(get_stats(all_means, n=nb_values))
    # very_significant = np.where(significance < 0.001)[0]
    # for i in very_significant:
    #     if bigger[i]:
    #         color = all_colors[colors['SoftmaxFiniteHorizon']]
    #     else:
    #         color = all_colors[colors['SoftmaxMultiModel']]
    #     plt.text(x_axis[i], plt.gca().get_ylim()[0] - 0.1, '*', fontsize=12, color=color,
    #              ha='center', va='top')
    # plt.ylim(plt.gca().get_ylim()[0] - 0.5, plt.gca().get_ylim()[1])

    plt.ylabel("Reward")
    plt.xlabel("Number of steps after the task change")
    plt.legend()
    if title == '':
        title = str(time.time())
    plt.savefig('results/reward_after_change'+title+'.pdf')
    plt.close()


def plot_trial_reward_after_change(dic_of_rewards,
                                   nb_steps,
                                   nb_trials,
                                   change_rate,
                                   nb_iters,
                                   title=''):
    plt.figure(dpi=100)
    all_means = {}
    all_stds = {}
    trials_each_change = int(change_rate//nb_steps)
    nb_values = nb_trials*nb_iters//trials_each_change
    for agent in dic_of_rewards.keys():
        all_means[agent] = []
        all_stds[agent] = []
        rewards = dic_of_rewards[agent]
        not_used = nb_trials % trials_each_change
        for j in range(trials_each_change):
            filter_indices = np.array([i+j for i in range(0,
                                                          nb_trials-not_used,
                                                          trials_each_change)])
            r_change = np.take(rewards, filter_indices, 1).flatten()

            mean = np.mean(r_change)
            std = np.std(r_change)
            all_means[agent].append(mean)
            all_stds[agent].append(std)

        array_mean = np.array(all_means[agent])
        array_std = np.array(all_stds[agent])
        array_CI = 3.291*array_std / np.sqrt(nb_values)
        x_axis = np.arange(1, trials_each_change+1)
        plt.plot(x_axis,
                 array_mean,
                 label=labels[agent],
                 color=all_colors[colors[agent]],
                 marker='.')
        plt.fill_between(x_axis,
                         array_mean-array_CI,
                         array_mean+array_CI,
                         color=all_colors[colors[agent]],
                         alpha=0.15)

    plt.ylabel("Rewards")
    plt.xlabel("Number of trials after the task change")
    plt.legend(loc='lower center')
    if title == '':
        title = str(time.time())
    plt.savefig('results/reward_trial_after_change'+title+'.pdf')
    plt.close()


def plot_reward_per_change(dic_of_rewards,
                           nb_steps,
                           nb_trials,
                           change_rate,
                           nb_iters,
                           title=''):
    plt.figure(dpi=100)
    all_means = {}
    all_stds = {}
    if change_rate % nb_steps != 0:
        print("Cannot plot reward per change, change_rate % nb_steps != 0.""")
        return None
    trials_each_change = int(change_rate//nb_steps)
    nb_values = nb_iters*trials_each_change
    nb_changes = nb_trials//trials_each_change
    for agent in dic_of_rewards.keys():
        all_means[agent] = []
        all_stds[agent] = []
        rewards = np.array(dic_of_rewards[agent])
        for i in range(nb_changes):
            starting_i = i*trials_each_change
            r_change = rewards[:,starting_i:starting_i+trials_each_change]
            mean = np.mean(r_change)
            std = np.std(r_change)
            all_means[agent].append(mean)
            all_stds[agent].append(std)

        array_mean = np.array(all_means[agent])
        array_std = np.array(all_stds[agent])
        array_CI = 3.291*array_std / np.sqrt(nb_values)
        x_axis = np.arange(0, nb_changes)
        plt.plot(x_axis,
                 array_mean,
                 label=labels[agent],
                 color=all_colors[colors[agent]],
                 marker='.')
        plt.fill_between(x_axis,
                         array_mean-array_CI,
                         array_mean+array_CI,
                         color=all_colors[colors[agent]],
                         alpha=0.15)

    plt.ylabel("Average reward")
    plt.xlabel("Task change number")
    plt.legend(loc='lower center')
    if title == '':
        title = str(time.time())
    plt.savefig('results/reward_per_task_change'+title+'.pdf')
    plt.close()

def plot_time_per_change(dic_of_times,
                           nb_steps,
                           nb_trials,
                           change_rate,
                           nb_iters,
                           title=''):
    plt.figure(dpi=100)
    all_means = {}
    all_stds = {}
    if change_rate % nb_steps != 0:
        print("Cannot plot reward per change, change_rate % nb_steps != 0.""")
        return None
    trials_each_change = int(change_rate//nb_steps)
    nb_values = nb_iters*trials_each_change
    nb_changes = nb_trials//trials_each_change
    for agent in dic_of_times.keys():
        all_means[agent] = []
        all_stds[agent] = []
        rewards = np.array(dic_of_times[agent])
        for i in range(nb_changes):
            starting_i = i*trials_each_change
            r_change = rewards[:,starting_i:starting_i+trials_each_change]*1e3  # ms
            mean = np.mean(r_change)
            std = np.std(r_change)
            all_means[agent].append(mean)
            all_stds[agent].append(std)

        array_mean = np.array(all_means[agent])
        array_std = np.array(all_stds[agent])
        array_CI = 3.291*array_std / np.sqrt(nb_values)
        x_axis = np.arange(0, nb_changes)
        plt.plot(x_axis,
                 array_mean,
                 label=labels[agent],
                 color=all_colors[colors[agent]],
                 marker='.')
        plt.fill_between(x_axis,
                         array_mean-array_CI,
                         array_mean+array_CI,
                         color=all_colors[colors[agent]],
                         alpha=0.15)

    plt.ylabel("Average time per decision (ms)")
    plt.xlabel("Task change number")
    plt.legend(loc='lower center')
    if title == '':
        title = str(time.time())
    plt.savefig('results/time_per_task_change'+title+'.pdf')
    plt.close()

def plot_time_after_change(dic_of_times,
                           nb_steps,
                           nb_trials,
                           change_rate,
                           nb_iters,
                           title='',
                           event='steps'):
    plt.figure(dpi=100)
    all_medians = {}
    all_q_25 = {}
    all_q_75 = {}
    all_means = {}
    all_stds = {}
    length_change = int(change_rate//nb_steps)
    not_used = nb_trials % length_change
    nb_values = nb_trials*nb_iters//length_change
    for agent in dic_of_times.keys():
        all_medians[agent] = []
        all_q_25[agent] = []
        all_q_75[agent] = []
        all_means[agent] = []
        all_stds[agent] = []
        times = dic_of_times[agent]
        for j in range(length_change):
            filter_indices = np.array([i+j for i in range(0,
                                                          nb_trials-not_used,
                                                          length_change)])

            t_change = np.take(times, filter_indices, 1).flatten()*1e3  # ms
            # all_values[agent].append(t_change)
            median = np.median(t_change)
            q_25 = np.quantile(t_change, 0.25)
            q_75 = np.quantile(t_change, 0.75)

            all_medians[agent].append(median)
            all_q_25[agent].append(q_25)
            all_q_75[agent].append(q_75)

            mean = np.mean(t_change)
            std = np.std(t_change)
            all_means[agent].append(mean)
            all_stds[agent].append(std)

    # Show the plot
        array_median = np.array(all_medians[agent])
        array_q_25 = np.array(all_q_25[agent])
        array_q_75 = np.array(all_q_75[agent])
        array_mean = np.array(all_means[agent])
        array_std = np.array(all_stds[agent])
        array_CI = 3.291*array_std / np.sqrt(nb_values)

        x_axis = np.arange(1, length_change+1)
        plt.plot(x_axis,
                 array_mean,
                 label=labels[agent],
                 color=all_colors[colors[agent]],
                 marker='.')
        plt.fill_between(x_axis,
                         array_mean-array_CI,
                         array_mean+array_CI,
                         color=all_colors[colors[agent]],
                         alpha=0.15)
        # print(array_std)
        # print(array_mean)

        # nb_values = nb_trials*nb_steps*nb_iters//change_rate
        # significance, bigger = np.array(get_stats(all_means,n=nb_values))
        # very_significant = np.where(significance < 0.001)[0]
        # for i in very_significant:
        #     if bigger[i] : color = all_colors[colors['SoftmaxFiniteHorizon']]
        #     else : color = all_colors[colors['SoftmaxMultiModel']]
        #     plt.text(x_axis[i], plt.gca().get_ylim()[0] - 0.1, '*', fontsize=12, color=color,
        #             ha='center', va='top')
        # plt.ylim(plt.gca().get_ylim()[0] - 5, plt.gca().get_ylim()[1])

    plt.ylabel("Time for each decision (ms)")
    plt.xlabel("Number of " + event + " after the task change")
    plt.legend(loc='upper right')

    if title == '':
        title = str(time.time())
    plt.savefig('results/times_after_change'+title+'.pdf')
    plt.close()

    # all_values = {k:np.array(v) for (k,v) in all_values.items()}
    # x_ticks = np.arange(change_rate)

    # # Width of each boxplot group
    # box_width = 0.2

    # # Define colors for the categories
    # colors = ['lightblue', 'lightgreen', 'lightcoral']  # Different colors for the triplets

    # # Loop over the categories (3 in this case) and plot them
    # for i, (category, values) in enumerate(all_values.items()):
    #     # Offset each category slightly on the x-axis
    #     positions = x_ticks + i * box_width
    #     # Plot the boxplot for the current category with patch_artist=True
    #     box = ax.boxplot(values.T, positions=positions, widths=box_width, patch_artist=True,showfliers=False)

    #     # Set the color for each boxplot in the triplet
    #     for patch in box['boxes']:
    #         patch.set_facecolor(colors[i])

    # # Customizing the plot
    # ax.set_xticks(x_ticks + box_width)  # Shift x-ticks to the center of the group
    # ax.set_xticklabels([f'{i+1}' for i in range(change_rate)])  # Custom x-labels

    # # Label axes
    # ax.set_xlabel('Group')
    # ax.set_ylabel('Values')
    # ax.set_title('50 Triplets of Boxplots with Different Colors')


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


def get_all_plot(results, parameters):
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

    for info_exp, all_values in results.items():
        agent_name = info_exp[1]
        all_rewards[agent_name].append(all_values["reward"])
        all_times[agent_name].append(all_values["times"])
        all_total_times[agent_name].append(all_values["total_time"])

        if agent_name in multi_model_agents:
            for key_model in all_models[agent_name].keys():
                all_models[agent_name][key_model].append(all_values[key_model])
                # print(all_models[agent])
                # print(agent)
            if "ChangingCrossEnvironment" == env_tested:
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
                        title=agent+title)
            one_env = len(parameters['env_param']) == 1
            if env_tested == "ChangingCrossEnvironment" and one_env:
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
        if "ThreeStates" == env_tested or "MAB"==env_tested:
            event = 'steps'
            plot_reward_after_change(all_rewards,
                                     steps,
                                     trials,
                                     change_rate,
                                     nb_iters,
                                     title)
        else:
            event = 'trials'
            plot_trial_reward_after_change(all_rewards,
                                           steps,
                                           trials,
                                           change_rate,
                                           nb_iters,
                                           title)
        plot_reward_per_change(all_rewards,
                               steps,
                               trials,
                               change_rate,
                               nb_iters,
                               title)
        
        plot_time_per_change(all_times,
                               steps,
                               trials,
                               change_rate,
                               nb_iters,
                               title)
        plot_time_after_change(all_times,
                               steps,
                               trials,
                               change_rate,
                               nb_iters,
                               title,
                               event)
        general_performance(all_rewards, title)


# ---------------------------------------------------------------------------- #
# Maze Plots
# ---------------------------------------------------------------------------- #


def plot_maze(world,
              path,
              labels=np.empty(0)):

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

    array_of_colors[init_state] = blue
    array_of_colors[walls] = black
    array_of_colors[pattern_array] = yellow

    reward_matrix = world == reward
    array_of_colors[reward_matrix] = red
    _, ax = plt.subplots(1, 1, dpi=100)
    # adding colors
    ax.imshow(array_of_colors, aspect='equal')
    if len(labels) == 0:
        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                if reward_matrix[j, i]:
                    ax.text(i, j, "R", va='center', ha='center')
                if init_state[j, i]:
                    ax.text(i, j, "I", va='center', ha='center')
    # adding labels
    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            c = labels[j, i]
            ax.text(i, j, str(c), va='center', ha='center')
    major_ticks = np.arange(-0.5, 7.5)
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
    plt.savefig(path)
    plt.close()


def plot_one_transition(world,
                        transitions,
                        path_save,
                        cell_number,
                        action_direction):

    action_to_arrow = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

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

    array_of_colors[init_state] = blue
    array_of_colors[walls] = black
    array_of_colors[pattern_array] = yellow

    array_of_colors = np.pad(array_of_colors, pad_width=1, mode='constant',
                             constant_values=black)

    print(np.shape(array_of_colors))

    reward_matrix = world == reward
    array_of_colors[reward_matrix] = red
    _, ax = plt.subplots(1, 1, dpi=100)
    # adding colors
    # adding labels
    ax.imshow(array_of_colors, aspect='equal')
    for i in range(0, transitions.shape[0]):
        for j in range(0, transitions.shape[1]):
            percent = transitions[j, i]
            ax.text(i, j, str(percent), va='center', ha='center')
    dx, dy = 50*action_to_arrow[action_direction]
    ax.arrow(30, 30, dx, dy)

    major_ticks = np.arange(-0.5, 7.5)
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
    plt.savefig(path_save)
    plt.close()
