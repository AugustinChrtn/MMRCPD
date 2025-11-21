import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import glob
import os
import numpy as np
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
# 2D Plots
# ---------------------------------------------------------------------------- #


def get_max_Q_values_and_policy(table):
    best_values = np.max(table, axis=1)
    random_noise = 1e-5 * np.random.random(table.shape)
    best_actions = np.argmax(table + random_noise, axis=1)
    return best_values, best_actions


def plot_2D(table, shape):
    import seaborn as sns
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
