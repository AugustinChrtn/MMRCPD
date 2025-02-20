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
    # print(name_agent)
    # print(multi_model_agents)
    # print(name_agent)
    # print(multi_model)
    # path = 'data/'+name_agent+'/log.npy'
    # unique_path, number = uniquify(path)

    log = {}

    if multi_model:
        log = {**log,
               'nb_model': [],
               'nb_creation': [],
               'nb_forgetting': [],
               'nb_merging': []}
    start_time = time.time()
    reward_per_episode = []
    time_per_episode = []
    # policy_value_error = []
    # threshold = 1e-3

    for trial in range(trials):
        cumulative_reward, step, game_over, time_trial = 0, 0, False, 0
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

    end_time = time.time()
    log['reward'] = reward_per_episode
    log['total_time'] = end_time-start_time
    log['times'] = time_per_episode
    if multi_model:
        log['creation_per_state'] = agent.creation_per_state
        log['model_per_state'] = agent.model_per_state
    # log['policy_value_error']= policy_value_error
    # np.save(unique_path, log)
    return log


def play_with_logs(environment, agent, trials=100, max_step=30):

    # environment.__init__()
    reward_per_episode = []
    nb_models = []
    for _ in range(trials):
        cumulative_reward, step, game_over = 0, 0, False
        while not game_over:
            old_state = environment.agent_state
            action = agent.choose_action(old_state)
            reward, new_state = environment.make_step(action)
            agent.learn(old_state, reward, new_state, action)
            cumulative_reward += reward
            step += 1
            if step == max_step:
                game_over = True
                environment.new_episode()
        reward_per_episode.append(cumulative_reward)
        nb_models.append(np.sum(agent.nb_models))
    return reward_per_episode, nb_models

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
        for iteration in range(nb_tests):
            for param_env in env_parameters :
                trial_name = (env_name, agent_name, iteration)
                simulation_to_do.append({'trial_name': trial_name,
                                        'env_name': env_name,
                                        'agent_name': agent_name,
                                        'seed': seed,
                                        'play_parameters': play_parameters,
                                        'env_param': param_env,
                                        'agent_param': agent_parameters[agent_name]})

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
                  'time':title
                  }
    
    np.save('results/parameters'+title+'.npy', parameters)
    return logs, parameters


# def play_likelihood(agent_parameter_list,
#                     subject_choice,
#                     env_parameter,
#                     env_name,
#                     agent_name,):

#     environment = env_name_to_class[env_name](**env_parameter)
#     agent = agent_name_to_class[agent_name](environment, *agent_parameter_list)

#     return -play(environment,
#                  agent, forced_choice=subject_choice)["total_likelihood"]


# def find_best_param_minimize(x0,
#                              bounds,
#                              env_parameter,
#                              env_name,
#                              agent_name,
#                              subject_choice):

#     result = minimize(play_likelihood, x0=x0, bounds=bounds,
#                       args=(subject_choice, env_parameter,
#                             env_name, agent_name),
#                       )

#     return result.x, -result.fun


# def find_best_fit(agents,
#                   participant_choices,
#                   env_name,
#                   env_parameter):
#     '''Finding the best fit with the minimize function. To improve this code, 
#     we should use multiprocessing.'''
#     print("Parameter fitting started")
#     start_time = time.time()
#     best_params = {}
#     best_likelihood = {}
#     for agent_name in agents:
#         best_params[agent_name] = []
#         best_likelihood[agent_name] = []
#         bounds = agent_bounds[agent_name]
#         x0 = agent_initial_guess[agent_name]
#         for subject_choice in participant_choices:
#             values, likelihood = find_best_param_minimize(x0,
#                                                           bounds,
#                                                           env_parameter,
#                                                           env_name,
#                                                           agent_name,
#                                                           subject_choice)
#             best_params[agent_name].append(list(values))
#             best_likelihood[agent_name].append(likelihood)
#     current_time = str(time.time())
#     path_lik = 'data/all_likelihoods'+current_time+'.npy'
#     path_params = 'data/all_params'+current_time+'.npy'
#     np.save('data/all_likelihoods'+current_time+'.npy', best_likelihood)
#     np.save('data/all_params'+current_time+'.npy', best_params)
#     print("Computational Time for parameter fitting: "
#           + str(time.time()-start_time) + "seconds")
#     print("The best parameters and the corresponding likelihood were saved " +
#           "in "+path_lik + " and " + path_params)
#     return best_params,  best_likelihood, path_lik, path_params, current_time

