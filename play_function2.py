from variables import envs, agents
from multiprocessing import Pool
from consts import multi_model_agents, one_step_environments
from consts import mM_and_RLCD
import numpy as np
import time
import pandas as pd
import json
import os

def get_run_dir(base_dir, env, cond=None):

    prefix = env if cond is None else f"{env}_{cond}"
    os.makedirs(base_dir, exist_ok=True)

    runs = []
    for name in os.listdir(base_dir):
        if name.startswith(prefix + "_exp"):
            try:
                idx = int(name.split("_exp")[-1])
                runs.append(idx)
            except ValueError:
                pass

    next_id = max(runs)+1 if runs else 1
    return os.path.join(base_dir, f"{prefix}_exp{next_id:02d}")



def play(environment,
         agent,
         trials=100,
         max_step=30):

    name_agent = agent.__class__.__name__
    name_env = environment.__class__.__name__

    is_multi_model = name_agent in multi_model_agents
    env_is_one_step = name_env in one_step_environments

    logs = []

    # track_changes = name_agent in mM_and_RLCD
    # if track_changes :
    #     previous_changes = agent.nb_changes
    # else :
    #     previous_changes = 0

    for episode in range(trials):
        cumulative_reward = 0

        distance = compute_current_distance_transition(environment.transitions,
                                                       agent.tSAS)

        start_time = time.time()
        best_action_count = 0

        for _ in range(max_step):

            state = environment.agent_state
            action = agent.choose_action(state)
            reward, new_state = environment.make_step(action)

            agent.learn(state, reward, new_state, action)

            cumulative_reward += reward

        if env_is_one_step and action == environment.best_action:
            best_action_count += 1

        episode_time = time.time() - start_time
        environment.new_episode()

        # Adding res common to all agents
        result = {
            "episode": episode,
            "reward": cumulative_reward,
            "steps": max_step,
            "time (ms)": episode_time*1e3,
            "distance": distance
        }

        # Adding res in one step environment
        if env_is_one_step:
            result["best_action"] = best_action_count / max_step
        else :
            result["best_action"] = None

        # Adding res for muti model agents
        if is_multi_model:
            result.update({
                "nb_model": agent.total_nb_models,
                "nb_creation": agent.total_creation,
                "nb_forgetting": agent.total_forgetting,
                "nb_merging": agent.total_merging
            })

        # Adding changes res for RLCD and Multi-model agents
        # if track_changes:
        #     result["changes"] = agent.nb_changes - previous_changes
        #     previous_changes = agent.nb_changes

        logs.append(result)

    all_results = {"logs": logs}

    # Adding spatial information for multi-model agents
    if is_multi_model:
        final_arrays = {"creation_per_state": agent.creation_per_state,
                        "model_per_state": agent.model_per_state}
    else:
        final_arrays = {}

    all_results["final_arrays"] = final_arrays

    return all_results


def compute_current_distance_transition(transi_env, transi_agent):
    return np.sqrt(np.sum((transi_env - transi_agent) ** 2))


def get_simulation_to_do(agent_to_test,
                         env_name,
                         nb_tests,
                         play_parameters,
                         starting_seed,
                         env_parameters,
                         agent_parameters):

    simulations = []
    seed = starting_seed

    for agent_name in agent_to_test:
        count = 0

        for _ in range(nb_tests):
            for param_env in env_parameters:
                simulations.append({
                    "trial_id": count,
                    "env_name": env_name,
                    "agent_name": agent_name,
                    "seed": seed,
                    "play_parameters": play_parameters,
                    "env_param": param_env,
                    "agent_param": agent_parameters[agent_name]
                })
                count += 1
                seed += 1

    return simulations


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


def one_parameter_play_function(params):

    np.random.seed(params["seed"])

    environment = envs[params["env_name"]](**params["env_param"])
    agent = agents[params["agent_name"]](environment, **params["agent_param"])

    result = play(environment, agent, **params["play_parameters"])

    rows = result["logs"]
    arrays = result["final_arrays"]
    trial_id = params["trial_id"]

    # add metadata to every row of the csv file
    for r in rows:
        r.update({
            "trial_id": trial_id,
            "agent": params["agent_name"],
            "environment": params["env_name"],
            "seed": params["seed"]
        })

        #check for the maze number for maze exp
        if 'number' in params['env_param']:
            r['maze_number'] = params['env_param']['number']
    

    return rows, arrays, trial_id


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
                  nb_processes=5,
                  save_dir="results"):

    start_time = time.time()

    simulations = get_simulation_to_do(
        agent_to_test,
        env_to_test,
        nb_tests,
        play_parameters,
        starting_seed,
        env_parameters,
        agent_parameters
    )

    if nb_processes > 1:
        with Pool(nb_processes) as pool:
            results = pool.map(one_parameter_play_function, simulations)
    else:
        results = [one_parameter_play_function(sim) for sim in simulations]
    runtime = time.time() - start_time
    print("Computation time:", runtime)

    
    all_rows = []
    all_arrays = {}

    for rows, arrays, trial_id in results:
        all_rows.extend(rows)

        if arrays:
            all_arrays[trial_id] = arrays

    df = pd.DataFrame(all_rows)
    
    # Round some values to lower storing cost
    df = df.round({
        "time (ms)": 5,
        "distance": 4,
    })

    parameters = {
        **play_parameters,
        "env_name": env_to_test,
        "agents": agent_to_test,
        "nb_iters": nb_tests,
        "starting_seed": starting_seed,
        "env_param": env_parameters,
        "agent_param": agent_parameters,
        "runtime": runtime
    }

    run_dir = get_run_dir(save_dir, env_to_test)
    os.makedirs(run_dir, exist_ok=True)


    # ---------- CSV ----------
    init_compression = time.time()
    df.to_csv(f"{run_dir}/episode_results.csv.gz", 
                index=False,
                compression="gzip")
    
    print("Compression time to .csv", time.time()-init_compression)

    # ---------- ARRAYS ----------
    flat_arrays = {}
    if all_arrays:
        for trial_id, arr_dict in all_arrays.items():
            for name, arr in arr_dict.items():
                flat_arrays[f"{trial_id}_{name}"] = arr

        np.savez_compressed(
            f"{run_dir}/final_arrays.npz",
            **flat_arrays
        )

    # ---------- PARAMETERS ----------
    with open(f"{run_dir}/parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)

    return run_dir
