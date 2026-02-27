import os
from play_function import main_function
from plots import get_plot_from_saved


def set_param_agent(params_shared={},
                    params_multi_model={},
                    params_rlcd={}):

    params_VI_softmax = {**params_shared}

    params_SoftmaxFiniteHorizon3 = {**params_shared,
                                    'horizon': 3}

    params_SoftmaxFiniteHorizon5 = {**params_shared,
                                    'horizon': 5}

    params_SoftmaxFiniteHorizon10 = {**params_shared,
                                     'horizon': 10}

    params_SoftmaxFiniteHorizon20 = {**params_shared,
                                     'horizon': 20}

    params_SoftmaxFiniteHorizon30 = {**params_shared,
                                     'horizon': 30}

    params_MMRCPD = {**params_shared,
                     **params_multi_model,
                     'reassign': True,
                     'semi_jensen': False}

    params_MMRCPD_no_reassign = {**params_shared,
                                 **params_multi_model,
                                 'reassign': False}

    params_MMRCPD_semi_jensen = {**params_shared,
                                 **params_multi_model,
                                 'semi_jensen': True}

    params_RLCD = {**params_shared,
                   **params_rlcd}

    low_KL = 0.3
    no_merging = 0.
    low_merging = 0.01
    high_merging = 0.3
    low_memory = 2
    low_horizon = 3
    high_horizon = 20
    high_KL = 1.5

    agent_parameters = {'VI_softmax': params_VI_softmax,
                        'SoftmaxFiniteHorizon3': params_SoftmaxFiniteHorizon3,
                        'SoftmaxFiniteHorizon5': params_SoftmaxFiniteHorizon5,
                        'SoftmaxFiniteHorizon10': params_SoftmaxFiniteHorizon10,
                        'SoftmaxFiniteHorizon20': params_SoftmaxFiniteHorizon20,
                        'SoftmaxFiniteHorizon30': params_SoftmaxFiniteHorizon30,
                        'MMRCPD': params_MMRCPD,
                        'MMRCPDNoReassign': params_MMRCPD_no_reassign,
                        'MMRCPDSemiJensen': params_MMRCPD_semi_jensen,
                        'RLCD': params_RLCD
                        }

    for agent in ['MMLowKL', 'MMHighKL', 'MMLowHorizon', 'MMHighHorizon',
                  'MMHighHorizonLowKL', 'MMLowMerging', 'MMHighMerging',
                  'MMForget', 'MMNoMerging', 'Baseline']:
        agent_parameters[agent] = params_MMRCPD.copy()

    agent_parameters['MMLowKL']['kl_threshold'] = low_KL
    agent_parameters['MMHighKL']['kl_threshold'] = high_KL
    agent_parameters['MMLowHorizon']['horizon'] = low_horizon
    agent_parameters['MMHighHorizon']['horizon'] = high_horizon
    agent_parameters['MMHighHorizonLowKL']['horizon'] = high_horizon
    agent_parameters['MMHighHorizonLowKL']['kl_threshold'] = low_KL

    for agent in ['MMLowMerging', 'MMHighMerging', 'MMForget', 'MMNoMerging']:
        agent_parameters[agent]['kl_threshold'] = low_KL
    agent_parameters['MMLowMerging']['merging_threshold'] = low_merging
    agent_parameters['MMHighMerging']['merging_threshold'] = high_merging

    agent_parameters['MMNoMerging']['merging_threshold'] = no_merging

    agent_parameters['MMForget']['nb_max_models'] = low_memory
    agent_parameters['MMForget']['merging_threshold'] = no_merging

    return agent_parameters


def generate_seed(number_of_the_experiment):
    return 1000*number_of_the_experiment+1

# ---------------------------------------------------------------------------- #
# Example to generate the plots
# ---------------------------------------------------------------------------- #

# all_names = []
# init_dir = 'results'
# for e in os.scandir(init_dir):
#     if e.is_dir():
#         all_names.append(init_dir+'/'+e.name)
# for save_dir in all_names:
#     if '01' in save_dir and 'Chain' in save_dir:
#         suptitle = "Local volatility"
#     elif '02' in save_dir and 'Chain' in save_dir:
#         suptitle = "Global volatility"
#     elif '03' in save_dir and 'Chain' in save_dir:
#         suptitle = "All-but-one volatility"
#     elif '01' in save_dir:
#         suptitle = "Uncertain variation"
#     elif '02' in save_dir:
#         suptitle = "Volatile variation"
#     elif '03' in save_dir:
#         suptitle = 'Uncertain and Volatile variation'
#     elif '04' in save_dir:
#         suptitle = 'Retrospective change detection'
#     else:
#         suptitle = ''
#     get_plot_from_saved(save_dir, suptitle=suptitle)


# ---------------------------------------------------------------------------- #
# Experiments
# ---------------------------------------------------------------------------- #


# Indicate all the experiments you would like to launch (check below)
experiments_to_launch = []

# Indicate the number of processors you would like to use
nb_proc = 10

# ---------------------------------------------------------------------------- #
# The advantage of using change detection methods
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Exp 1 - Three states - Volatile variation
# ---------------------------------------------------------------------------- #
nb_exp = 1
starting_seed = generate_seed(nb_exp)
nb_tests = 500

env_to_test = 'ThreeStates'
play_parameters = {'trials': 1000,
                   'max_step': 1}

env_parameters = [{"slip": 0.1,
                   "step_change": 50}]

agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon3',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxFiniteHorizon30',
                  'RLCD',
                  'MMRCPD']

params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 5,
                 "gamma": 0.95}

params_multi_model = {"horizon": 3,
                      "kl_threshold": 0.5,
                      "merging_threshold": 0.3,
                      "delay": 1,
                      "nb_max_models": 5}

params_rlcd = {"horizon": 50,
               "Emin": -0.05,
               "rho": 0.4}


agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle="Volatile variation")


# ---------------------------------------------------------------------------- #
# Exp 2 - Four States - Uncertain transition
# ---------------------------------------------------------------------------- #
nb_exp = 2
nb_tests = 2000
starting_seed = generate_seed(nb_exp)
env_to_test = 'FourStates'

play_parameters = {'trials': 200,
                   'max_step': 1}

env_parameters = [{"step_change": None,
                   'slip': 0.1}]

agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle="Uncertain variation")

# ---------------------------------------------------------------------------- #
# Exp 3 - Four States - Volatile Uncertain transition
# ---------------------------------------------------------------------------- #
nb_exp = 3
nb_tests = 500
starting_seed = generate_seed(nb_exp)

env_to_test = 'FourStates'
play_parameters = {'trials': 1000,
                   'max_step': 1}

env_parameters = [{"step_change": 50,
                   'slip': 0.1}]


agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle="Uncertain and Volatile variation")

# ---------------------------------------------------------------------------- #
# End of one-step environment
# Beginning of mazes
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Exp 4 - Mazes - Volatile variation
# ---------------------------------------------------------------------------- #
nb_exp = 4
starting_seed = generate_seed(nb_exp)
nb_tests = 20
nb_mazes = 10


agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon3',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxFiniteHorizon30',
                  'RLCD',
                  'MMRCPD']

# agents_to_test = ['MMRCPD']

# agents_to_test = [
#                   'MMRCPD',
#                   'SoftmaxFiniteHorizon20',
#                   'SoftmaxFiniteHorizon3']


play_parameters = {'trials': 1000,
                   'max_step': 100}

env_parameters = []
env_to_test = 'PartiallyChangingMaze'

for i in range(nb_mazes):
    env_parameters.append({'number': i,
                           'step_change': 2000,
                           'conds': ['', '_D'],
                           'value_change': 0.2,
                           'uncertain': False})

params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 3,
                 "gamma": 0.95}


params_multi_model = {"horizon": 10,
                      "kl_threshold": 1,
                      "merging_threshold": 0.1,
                      "delay": 1,
                      "nb_max_models": 5}

params_rlcd = {"horizon": 50,
               "Emin": -0.05,
               "rho": 0.2}

agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle="Volatile variation")


# ---------------------------------------------------------------------------- #
# Maze Environment Back to initial position
# Exp 5 - Mazes - uncertain variation
# ---------------------------------------------------------------------------- #
nb_exp = 5
nb_tests = 50
starting_seed = generate_seed(nb_exp)

play_parameters = {'trials': 200,
                   'max_step': 100}

env_parameters = []

env_to_test = 'PartiallyChangingMaze'
for i in range(nb_mazes):
    env_parameters.append({'number': i,
                           'step_change': None,
                           'conds': ['', '_D'],
                           'value_change': 0.2,
                           'uncertain': True})


agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle='Uncertain variation')

# ---------------------------------------------------------------------------- #
# Exp 6 - Mazes - uncertain-volatile variation
# ---------------------------------------------------------------------------- #
nb_exp = 6
nb_tests = 20
starting_seed = generate_seed(nb_exp)

play_parameters = {'trials': 1000,
                   'max_step': 100}

env_parameters = []

env_to_test = 'PartiallyChangingMaze'
for i in range(nb_mazes):
    env_parameters.append({'number': i,
                           'step_change': 2000,
                           'conds': ['', '_D'],
                           'value_change': 0.2,
                           'uncertain': True})

agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle='Uncertain and volatile variation')

# ---------------------------------------------------------------------------- #
# End of mazes
# Beginning of chain env
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Context vs local
# Exp 7 - Chain Task local change - RLCD
# ---------------------------------------------------------------------------- #
nb_exp = 7
starting_seed = generate_seed(nb_exp)
nb_tests = 100
env_to_test = 'ChainProblem'

agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon5',
                  'RLCD',
                  'MMRCPD']

play_parameters = {'trials': 1000,
                   'max_step': 20}

env_parameters = [{"slip": 0.1,
                  "size_chain": 5,
                   "step_change": 500,
                   "changes": ['S']}]

params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 3,
                 "gamma": 0.95}


params_multi_model = {"horizon": 5,
                      "kl_threshold": 1.,
                      "merging_threshold": 0.1,
                      "delay": 1,
                      "nb_max_models": 5}

params_rlcd = {"horizon": 20,
               "Emin": -0.05,
               "rho": 0.3}

agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle='Local volatility')

# ---------------------------------------------------------------------------- #
# Exp 8 - Chain Task global change
# ---------------------------------------------------------------------------- #
nb_exp = 8
starting_seed = generate_seed(nb_exp)

play_parameters = {'trials': 2000,
                   'max_step': 20}

env_parameters = [{"slip": 0.1,
                  "size_chain": 5,
                   "step_change": 1000,
                   "changes": ['T']}]

agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle='Global volatility')


# ---------------------------------------------------------------------------- #
# Exp 9 - Chain Task all-but-one
# ---------------------------------------------------------------------------- #
nb_exp = 9
starting_seed = generate_seed(nb_exp)
play_parameters = {'trials': 2000,
                   'max_step': 20}

env_parameters = [{"slip": 0.1,
                  "size_chain": 5,
                   "step_change": 1000,
                   "changes": ['T minus S']}]

agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle='All-but-one volatility')


# ---------------------------------------------------------------------------- #
# Robustness to parametrization
# Retrospective Change-point detection
# Exp 10 - Chain Retrospective
# ---------------------------------------------------------------------------- #
nb_exp = 10
starting_seed = generate_seed(nb_exp)
nb_tests = 200
env_to_test = 'ChainProblem'

agents_to_test = [
    'MMRCPD',
    'MMRCPDNoReassign']

play_parameters = {'trials': 1000,
                   'max_step': 20}

env_parameters = [{"slip": 0.1,
                  "size_chain": 5,
                   "step_change": 100,
                   "changes": ['S']}]

params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 3,
                 "gamma": 0.95}


params_multi_model = {"horizon": 5,
                      "kl_threshold": 1.,
                      "merging_threshold": 0.1,
                      "delay": 1,
                      "nb_max_models": 5}

params_rlcd = {"horizon": 20,
               "Emin": -0.05,
               "rho": 0.3}

agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle='Retrospective change detection')


# ---------------------------------------------------------------------------- #
# Chain Environment
# Exp 11 - Parameter Comparison horizon and creation
# ---------------------------------------------------------------------------- #
nb_exp = 11
starting_seed = generate_seed(nb_exp)
nb_tests = 100

agents_to_test = ['Baseline',
                  'MMLowKL',
                  'MMHighKL',
                  'MMLowHorizon',
                  'MMHighHorizon',
                  'MMHighHorizonLowKL']


env_to_test = 'ChainProblem'
play_parameters = {'trials': 500,
                   'max_step': 50}

env_parameters = [{"slip": 0.1,
                  "size_chain": 5,
                   "step_change": 500,
                   "changes": ['S']}]

params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 3,
                 "gamma": 0.95}


params_multi_model = {"horizon": 10,
                      "kl_threshold": 1.,
                      "merging_threshold": 0.1,
                      "delay": 1,
                      "nb_max_models": 5}

agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle=None)


# ---------------------------------------------------------------------------- #
# Chain Environment
# Exp 12 - Parameter Comparison merging and forgetting
# ---------------------------------------------------------------------------- #
nb_exp = 12
starting_seed = generate_seed(nb_exp)
nb_tests = 100
agents_to_test = ['Baseline',
                  'MMHighMerging',
                  'MMForget',
                  'MMNoMerging']

# agents_to_test = ['Baseline']


env_to_test = 'ChainProblem'
play_parameters = {'trials': 500,
                   'max_step': 50}


agent_parameters = set_param_agent(params_shared,
                                   params_multi_model,
                                   params_rlcd)

agent_parameters = {agent_name: agent_parameters.get(agent_name)
                    for agent_name in agents_to_test}

if nb_exp in experiments_to_launch and __name__ == '__main__':
    save_dir = main_function(agents_to_test,
                             env_to_test,
                             nb_tests,
                             play_parameters,
                             starting_seed,
                             env_parameters,
                             agent_parameters,
                             nb_processes=nb_proc)

    get_plot_from_saved(save_dir, suptitle='')


# # ---------------------------------------------------------------------------- #
# # Maze environment
# # Exp 15 - Partial change + RLCD
# # ---------------------------------------------------------------------------- #
# nb_exp = 15
# agents_to_test = ['VI_softmax',
#                   'SoftmaxFiniteHorizon10',
#                   'RLCD', 'MMRCPD']


# play_parameters = {'trials': 1000,
#                    'max_step': 100}

# env_parameters = []

# env_to_test = 'PartiallyChangingMaze'
# for i in range(10):
#     env_parameters.append({'number': i,
#                            'step_change': 2000,
#                            'conds': ['', '_D'],
#                            'value_change': 0.2,
#                            'uncertain': False})


# starting_seed = generate_seed(nb_exp)
# nb_tests = 1

# # Parameters agents
# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 3,
#                  "gamma": 0.95}


# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 1.,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}

# params_rlcd = {"horizon": 10,
#                "Emin": -0.15,
#                "rho": 0.3}

# agent_parameters = set_param_agent(params_shared,
#                                    params_multi_model,
#                                    params_rlcd)

# if nb_exp in experiments_to_launch and __name__ == '__main__':
#     save_dir = main_function(agents_to_test,
#                                env_to_test,
#                                nb_tests,
#                                play_parameters,
#                                starting_seed,
#                                env_parameters,
#                                agent_parameters,
#                                nb_processes=nb_proc)

#     get_all_plot(save_dir)

# # ---------------------------------------------------------------------------- #
# # Maze environment
# # Exp 16 - Total change RLCD
# # ---------------------------------------------------------------------------- #
# nb_exp = 16
# agents_to_test = ['VI_softmax', 'SoftmaxFiniteHorizon10',
#                   'RLCD', 'MMRCPD']


# play_parameters = {'trials': 1000,
#                    'max_step': 100}

# env_parameters = []
# env_to_test = 'PartiallyChangingMaze'
# for i in range(3):
#     env_parameters.append({'number': i,
#                            'step_change': 2000,
#                            'conds': ['', '_D'],
#                            'value_change': 1.,
#                            'uncertain': False})


# starting_seed = generate_seed(nb_exp)
# nb_tests = 1

# # Parameters agents
# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 3,
#                  "gamma": 0.95}


# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 1.,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}

# params_rlcd = {"horizon": 10,
#                "Emin": -0.15,
#                "rho": 0.3}

# agent_parameters = set_param_agent(params_shared,
#                                    params_multi_model,
#                                    params_rlcd)

# if nb_exp in experiments_to_launch and __name__ == '__main__':
#     save_dir = main_function(agents_to_test,
#                                env_to_test,
#                                nb_tests,
#                                play_parameters,
#                                starting_seed,
#                                env_parameters,
#                                agent_parameters,
#                                nb_processes=nb_proc)

#     get_all_plot(save_dir)

# # ---------------------------------------------------------------------------- #
# # Four states
# # Exp 17 - Total change Four states + RLCD
# # ---------------------------------------------------------------------------- #
# nb_exp = 17
# env_to_test = 'FourStates'
# play_parameters = {'trials': 2000,
#                    'max_step': 1}

# env_parameters = [{"step_change": 50,
#                    'slip': 0.1}]


# starting_seed = generate_seed(nb_exp)
# nb_tests = 1


# # Agents
# agents_to_test = ['VI_softmax',
#                   'SoftmaxFiniteHorizon5',
#                   'SoftmaxFiniteHorizon20',
#                   'RLCD',
#                   'MMRCPD']

# agents_to_test = ['RLCD',
#                   'MMRCPD']

# # Parameters agents
# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 10,
#                  "gamma": 0.95}


# params_multi_model = {"horizon": 5,
#                       "kl_threshold": 0.5,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}

# params_rlcd = {"horizon": 50,
#                "Emin": -0.1,
#                "rho": 0.3}


# agent_parameters = set_param_agent(params_shared,
#                                    params_multi_model,
#                                    params_rlcd)

# if nb_exp in experiments_to_launch and __name__ == '__main__':
#     save_dir = main_function(agents_to_test,
#                                env_to_test,
#                                nb_tests,
#                                play_parameters,
#                                starting_seed,
#                                env_parameters,
#                                agent_parameters,
#                                nb_processes=nb_proc)

#     get_all_plot(save_dir)
