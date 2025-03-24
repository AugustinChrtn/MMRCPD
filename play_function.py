from variables import envs, agents
from multiprocessing import Pool
from consts import multi_model_agents
import numpy as np
import os
import time

# Function to save a file as the first non-indexed number


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "(" + str(counter) + ")" + extension
        counter += 1

    return path, counter


def play(environment,
         agent,
         trials=100,
         max_step=30):

    name_agent = agent.__class__.__name__
    multi_model = name_agent in multi_model_agents

    log = {}

    if multi_model:
        log = {**log,
               'nb_model': [],
               'nb_creation': [],
               'nb_forgetting': [],
               'nb_merging': []}

    env_name = environment.__class__.__name__

    # log['distance_model'] = []
    log['distance_current_model'] = []

    start_time = time.time()
    reward_per_episode = []
    time_per_episode = []

    for trial in range(trials):
        cumulative_reward, step, game_over, time_trial = 0, 0, False, 0

        mod_agent = agent.tSAS
        mod_env = environment.transitions
        distance = compute_current_distance_transition(mod_env, mod_agent)
        log["distance_current_model"].append(distance)
        
        time_init_trial = time.time()
        while not game_over:
            old_state = environment.agent_state
            action = agent.choose_action(old_state)
            reward, new_state = environment.make_step(action)
            agent.learn(old_state, reward, new_state, action)
            cumulative_reward += reward
            time_trial = time.time()-time_init_trial

            step += 1
            if step == max_step:
                game_over = True
                environment.new_episode()
        time_trial = time.time()-time_init_trial
        time_per_episode.append(time_trial)
        reward_per_episode.append(cumulative_reward)
        if multi_model:
            log['nb_model'].append(agent.total_nb_models)
            log['nb_creation'].append(agent.total_creation)
            log['nb_forgetting'].append(agent.total_forgetting)
            log['nb_merging'].append(agent.total_merging)

        # mod_agent = agent.get_all_transitions()
        # mod_env = environment.all_transitions
        # distance = compute_distance_transitions(mod_env, mod_agent)
        # log["distance_model"].append(distance)

    end_time = time.time()
    log['reward'] = reward_per_episode
    log['total_time'] = end_time-start_time
    log['times'] = time_per_episode
    if multi_model:
        log['creation_per_state'] = agent.creation_per_state
        log['model_per_state'] = agent.model_per_state
        # from plots import plot_all_distrib_several_models, plot_V
        # # plot_all_distrib_several_models(environment, agent, nb_min_distrib=2)
        # policy_table = np.argmax(agent.Q, axis=1)
        # table = np.max(agent.Q,axis=1)
        # shape = (7,7)
        # plot_V(table, policy_table, shape, path='distrib/table.png')
    # log['policy_value_error']= policy_value_error
    # np.save(unique_path, log)
    # from plots import plot_all_distrib_several_models, plot_V
    # policy_table = np.argmax(agent.Q, axis=1)
    # table = np.max(agent.Q,axis=1)
    # shape = (7,7)
    # plot_V(table, policy_table, shape, path='distrib/'+name_agent+'.png')
    # print(log['distance_model'])
    # import matplotlib.pyplot as plt
    # plt.plot(log['distance_model'])
    # plt.show()
    return log


def compute_current_distance_transition(transi_env, transi_agent):
    return np.sqrt(np.sum((transi_env - transi_agent) ** 2))

# def compute_distance_transitions(transi_env, transi_agent):
#     distance = []
#     for (state, action) in transi_env.keys():
#         mod_env = transi_env[state, action]
#         mod_agent = transi_agent[state, action]
#         if len(mod_agent) == 0:
#             mod_agent = [[0.]*len(mod_env[0])]
#         mod_env = np.array(mod_env)
#         mod_agent = np.array(mod_agent)
#         # distances = cdist(mod_env, mod_agent, metric='euclidean')
#         distances = np.sqrt(np.sum((mod_env[:, np.newaxis] - mod_agent) ** 2, 
#                                    axis=2))
#         min_distances = np.min(distances, axis=1)

#         avg_min_distance = np.mean(min_distances)
#         # print(avg_min_distance)
#         distance.append(avg_min_distance)

#     return np.mean(distance)


# def play_with_logs(environment, agent, trials=100, max_step=30):
#     reward_per_episode = []
#     nb_models = []
#     for _ in range(trials):
#         cumulative_reward, step, game_over = 0, 0, False
#         while not game_over:
#             old_state = environment.agent_state
#             action = agent.choose_action(old_state)
#             reward, new_state = environment.make_step(action)
#             agent.learn(old_state, reward, new_state, action)
#             cumulative_reward += reward
#             step += 1
#             if step == max_step:
#                 game_over = True
#                 environment.new_episode()
#         reward_per_episode.append(cumulative_reward)
#         nb_models.append(np.sum(agent.nb_models))
#     return reward_per_episode, nb_models


def get_simulation_to_do(agent_to_test,
                         env_name,
                         nb_tests,
                         play_parameters,
                         starting_seed,
                         env_parameters,
                         agent_parameters):
    simulation_to_do = []
    seed = starting_seed
    for agent_name in agent_to_test:
        count = 0
        for _ in range(nb_tests):
            for param_env in env_parameters:
                trial_name = (env_name, agent_name, count)
                simulation_to_do.append({'trial_name': trial_name,
                                        'env_name': env_name,
                                         'agent_name': agent_name,
                                         'seed': seed,
                                         'play_parameters': play_parameters,
                                         'env_param': param_env,
                                         'agent_param': agent_parameters[agent_name]})
                count += 1
                seed += 1
    return simulation_to_do


def one_parameter_play_function(all_params_one_trial):

    seed = all_params_one_trial['seed']
    play_parameters = all_params_one_trial['play_parameters']
    env_name = all_params_one_trial['env_name']
    agent_name = all_params_one_trial['agent_name']
    trial_name = all_params_one_trial['trial_name']
    env_parameter = all_params_one_trial['env_param']
    agent_parameter = all_params_one_trial['agent_param']
    np.random.seed(seed)

    environment = envs[env_name](**env_parameter)
    agent = agents[agent_name](environment, **agent_parameter)
    return trial_name, play(environment, agent, **play_parameters)


def sum_up_all_parameters(agents,
                          env_name,
                          nb_iters,
                          play_params,
                          starting_seed,
                          env_param,
                          agent_param):

    parameters = {**play_params,
                  'env_name': env_name,
                  'agents': agents,
                  'nb_iters': nb_iters,
                  'starting_seed': starting_seed,
                  'env_param': env_param,
                  'agent_param': agent_param
                  }
    return parameters


def main_function(agent_to_test,
                  env_to_test,
                  nb_tests,
                  play_parameters,
                  starting_seed,
                  env_parameters,
                  agent_parameters,
                  nb_processes=5):

    time_before = time.time()
    every_simulation = get_simulation_to_do(agent_to_test,
                                            env_to_test,
                                            nb_tests,
                                            play_parameters,
                                            starting_seed,
                                            env_parameters,
                                            agent_parameters)
    pool = Pool(processes=nb_processes)
    results = pool.map(one_parameter_play_function, every_simulation)
    pool.close()
    pool.join()
    logs = {}
    for result in results:
        logs[result[0]] = result[1]
    time_after = time.time()
    print('Computation time: '+str(time_after - time_before))
    title = str(time_before)
    np.save('results/logs'+title+' .npy', logs)

    parameters = {**play_parameters,
                  'env_name': env_to_test,
                  'agents': agent_to_test,
                  'nb_iters': nb_tests,
                  'starting_seed': starting_seed,
                  'env_param': env_parameters,
                  'agent_param': agent_parameters,
                  'time': title
                  }

    np.save('results/parameters'+title+'.npy', parameters)
    return logs, parameters
