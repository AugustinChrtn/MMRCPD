from play_function import main_function
from plots import get_all_plot


def set_param_agent(params_shared={},
                    params_finite_horizon={},
                    params_multi_model={}):

    params_VI_softmax = {**params_shared}

    params_SoftmaxFiniteHorizon = {**params_shared,
                                   **params_finite_horizon}

    params_SoftmaxFiniteHorizon5 = {**params_shared,
                                    'horizon': 5}

    params_SoftmaxFiniteHorizon10 = {**params_shared,
                                     'horizon': 10}

    params_SoftmaxFiniteHorizon20 = {**params_shared,
                                     'horizon': 20}

    params_SoftmaxFiniteHorizon3 = {**params_shared,
                                    'horizon': 3}
    
    params_SoftmaxFiniteHorizon30 = {**params_shared,
                                    'horizon': 30}

    params_SoftmaxMultiModel = {**params_shared,
                                **params_multi_model}

    low_KL = 0.3
    no_merging = 0.
    low_merging = 0.01
    high_merging = 0.5
    low_memory = 2
    low_horizon = 3
    high_horizon = 20
    high_KL = 1.5

    agent_parameters = {'VI_softmax': params_VI_softmax,
                        'SoftmaxFiniteHorizon': params_SoftmaxFiniteHorizon,
                        'SoftmaxFiniteHorizon3': params_SoftmaxFiniteHorizon3,
                        'SoftmaxFiniteHorizon5': params_SoftmaxFiniteHorizon5,
                        'SoftmaxFiniteHorizon10': params_SoftmaxFiniteHorizon10,
                        'SoftmaxFiniteHorizon20': params_SoftmaxFiniteHorizon20,
                        'SoftmaxFiniteHorizon30': params_SoftmaxFiniteHorizon30,
                        'SoftmaxMultiModel': params_SoftmaxMultiModel,
                        }

    for agent in ['MMLowKL', 'MMHighKL', 'MMLowHorizon', 'MMHighHorizon',
                  'MMHighHorizonLowKL', 'MMLowMerging', 'MMHighMerging',
                  'MMForget', 'MMNoMerging', 'Baseline']:
        agent_parameters[agent] = params_SoftmaxMultiModel.copy()

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
# Experiments
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Three States environment
# Exp 1 - Basic task
# ---------------------------------------------------------------------------- #

# Environment

env_to_test = 'ThreeStates'
play_parameters = {'trials': 2000,
                   'max_step': 1}
env_parameters = [{"slip": 0.1,
                   "step_change": 50}]


starting_seed = generate_seed(0)
nb_tests = 200
nb_proc = 10


# Agents
agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon3',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxMultiModel']

# Parameters agents
params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 10,
                 "gamma": 0.95}

params_finite_horizon = {'horizon': 3}

params_multi_model = {"horizon": 5,
                      "kl_threshold": 0.3,
                      "merging_threshold": 0.3,
                      "delay": 1,
                      "nb_max_models": 5}


agent_parameters = set_param_agent(params_shared,
                                   params_finite_horizon,
                                   params_multi_model)

# res, param = main_function(agents_to_test,
#                            env_to_test,
#                            nb_tests,
#                            play_parameters,
#                            starting_seed,
#                            env_parameters,
#                            agent_parameters,
#                            nb_processes=nb_proc)

# get_all_plot(res, param)


# ---------------------------------------------------------------------------- #
# Three States environment
# Exp 2 with uncertain reward
# ---------------------------------------------------------------------------- #


# Environment

starting_seed = generate_seed(1)
env_to_test = 'ThreeStates'

play_parameters = {'trials': 2000,
                   'max_step': 1}

env_parameters = [{"slip": 0.1,
                   "step_change": 50,
                   "uncertain": True}]

nb_tests = 500

# Agents
agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon3',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxMultiModel']

params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 10,
                 "gamma": 0.95}

params_finite_horizon = {'horizon': 5}

params_multi_model = {"horizon": 5,
                      "kl_threshold": 0.3,
                      "merging_threshold": 0.3,
                      "delay": 1,
                      "nb_max_models": 5}

agent_parameters = set_param_agent(params_shared,
                                   params_finite_horizon,
                                   params_multi_model)

# res, param = main_function(agents_to_test,
#                            env_to_test,
#                            nb_tests,
#                            play_parameters,
#                            starting_seed,
#                            env_parameters,
#                            agent_parameters,
#                            nb_processes=nb_proc)

# get_all_plot(res, param)

# ---------------------------------------------------------------------------- #
# Four States
# Exp 1 - Basic task
# ---------------------------------------------------------------------------- #
env_to_test = 'FourStates'
play_parameters = {'trials': 2000,
                   'max_step': 1}

env_parameters = [{"step_change": 50,
                   'slip': 0.1}]


starting_seed = generate_seed(0)
nb_tests = 500
nb_proc = 10


# Agents
agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon3',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxMultiModel']

# Parameters agents
params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 10,
                 "gamma": 0.95}

params_finite_horizon = {'horizon': 3}

params_multi_model = {"horizon": 5,
                      "kl_threshold": 0.3,
                      "merging_threshold": 0.3,
                      "delay": 1,
                      "nb_max_models": 5}


agent_parameters = set_param_agent(params_shared,
                                   params_finite_horizon,
                                   params_multi_model)

# res, param = main_function(agents_to_test,
#                            env_to_test,
#                            nb_tests,
#                            play_parameters,
#                            starting_seed,
#                            env_parameters,
#                            agent_parameters,
#                            nb_processes=nb_proc)

# get_all_plot(res, param)


# ---------------------------------------------------------------------------- #
# Three States environment
# Exp 3 with small beta
# ---------------------------------------------------------------------------- #

starting_seed = generate_seed(42)

env_to_test = 'ThreeStates'

play_parameters = {'trials': 2000,
                   'max_step': 1}

env_parameters = [{"slip": 0.1,
                   "step_change": 50,
                   "uncertain": False}]
nb_tests = 200

# Agents
agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon3',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxMultiModel']

params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 1,
                 "gamma": 0.95}

params_finite_horizon = {'horizon': 5}

params_multi_model = {"horizon": 5,
                      "kl_threshold": 0.3,
                      "merging_threshold": 0.3,
                      "delay": 1,
                      "nb_max_models": 5}

agent_parameters = set_param_agent(params_shared,
                                   params_finite_horizon,
                                   params_multi_model)

# res, param = main_function(agents_to_test,
#                            env_to_test,
#                            nb_tests,
#                            play_parameters,
#                            starting_seed,
#                            env_parameters,
#                            agent_parameters,
#                            nb_processes=nb_proc)

# get_all_plot(res, param)

# ---------------------------------------------------------------------------- #
# Three States environment
# Exp 4 with no slipping
# ---------------------------------------------------------------------------- #

starting_seed = generate_seed(43)


env_to_test = 'ThreeStates'
play_parameters = {'trials': 2000,
                   'max_step': 1}

env_parameters = [{"slip": 0.,
                   "step_change": 50,
                   "uncertain": False}]
nb_tests = 200

# Agents
agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon3',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxMultiModel']

params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 10,
                 "gamma": 0.95}

params_finite_horizon = {'horizon': 5}

params_multi_model = {"horizon": 5,
                      "kl_threshold": 0.3,
                      "merging_threshold": 0.3,
                      "delay": 1,
                      "nb_max_models": 5}

# res, param = main_function(agents_to_test,
#                            env_to_test,
#                            nb_tests,
#                            play_parameters,
#                            starting_seed,
#                            env_parameters,
#                            agent_parameters,
#                            nb_processes=nb_proc)

# get_all_plot(res, param)


# ---------------------------------------------------------------------------- #
# Chain Environment
# Comparison between agents
# ---------------------------------------------------------------------------- #

agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon3',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxMultiModel']


# agents_to_test = ['VI_softmax',
#                   'SoftmaxFiniteHorizon10',
#                   'SoftmaxMultiModel']

env_to_test = 'ChainProblem'
play_parameters = {'trials': 500,
                   'max_step': 50}

env_parameters = [{"slip": 0.1,
                  "size_chain": 5,
                   "step_change": 500,
                   "changes": ['S']}]


starting_seed = generate_seed(2)
nb_tests = 100


nb_proc = 10

# Parameters agents
params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 3,
                 "gamma": 0.95}

params_finite_horizon = {'horizon': 10}

params_multi_model = {"horizon": 10,
                      "kl_threshold": 1,
                      "merging_threshold": 0.1,
                      "delay": 1,
                      "nb_max_models": 5}


agent_parameters = set_param_agent(params_shared,
                                   params_finite_horizon,
                                   params_multi_model)

# res, param = main_function(agents_to_test,
#                            env_to_test,
#                            nb_tests,
#                            play_parameters,
#                            starting_seed,
#                            env_parameters,
#                            agent_parameters,
#                            nb_processes=nb_proc)

# get_all_plot(res, param)


# ---------------------------------------------------------------------------- #
# Chain Environment
# Parameter Comparison horizon and creation
# ---------------------------------------------------------------------------- #


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


starting_seed = generate_seed(1000)
nb_tests = 100


nb_proc = 10

# Parameters agents
params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 3,
                 "gamma": 0.95}
params_finite_horizon = {'horizon': 10}

params_multi_model = {"horizon": 10,
                      "kl_threshold": 1,
                      "merging_threshold": 0.1,
                      "delay": 1,
                      "nb_max_models": 5}


agent_parameters = set_param_agent(params_shared,
                                   params_finite_horizon,
                                   params_multi_model)

res, param = main_function(agents_to_test,
                           env_to_test,
                           nb_tests,
                           play_parameters,
                           starting_seed,
                           env_parameters,
                           agent_parameters,
                           nb_processes=nb_proc)

get_all_plot(res, param)


# ---------------------------------------------------------------------------- #
# Chain Environment
# Parameter Comparison merging and forgetting
# ---------------------------------------------------------------------------- #


agents_to_test = ['Baseline',
                  'MMLowMerging',
                  'MMHighMerging',
                  'MMForget',
                  'MMNoMerging']


env_to_test = 'ChainProblem'
play_parameters = {'trials': 500,
                   'max_step': 50}

env_parameters = [{"slip": 0.1,
                  "size_chain": 5,
                   "step_change": 500,
                   "changes": ['S']}]


starting_seed = generate_seed(1001)
nb_tests = 100


nb_proc = 10

# Parameters agents
params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 3,
                 "gamma": 0.95}
params_finite_horizon = {'horizon': 10}

params_multi_model = {"horizon": 10,
                      "kl_threshold": 1,
                      "merging_threshold": 0.1,
                      "delay": 1,
                      "nb_max_models": 5}


agent_parameters = set_param_agent(params_shared,
                                   params_finite_horizon,
                                   params_multi_model)

res, param = main_function(agents_to_test,
                           env_to_test,
                           nb_tests,
                           play_parameters,
                           starting_seed,
                           env_parameters,
                           agent_parameters,
                           nb_processes=nb_proc)

get_all_plot(res, param)


# ---------------------------------------------------------------------------- #
# Maze
# Exp 1
# ---------------------------------------------------------------------------- #


agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxFiniteHorizon30',
                  'SoftmaxMultiModel']


play_parameters = {'trials': 1000,
                   'max_step': 100}

env_parameters = []

# env_to_test = 'ChangingCrossEnvironment'
# for i in range(1):
#     env_parameters.append({'number': i,
#                            'step_change': 1000,
#                            'conds': ['', '_C']})

env_to_test = 'PartiallyChangingCrossEnvironment'
for i in range(10):
    env_parameters.append({'number': i,
                           'step_change': 2000,
                           'conds': ['', '_D'],
                           'value_change': 0.2,
                           'uncertain': False})


starting_seed = generate_seed(17)
nb_tests = 5
nb_proc = 10

# Parameters agents
params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 3,
                 "gamma": 0.95}

params_finite_horizon = {'horizon': 10}

params_multi_model = {"horizon": 10,
                      "kl_threshold": 1.,
                      "merging_threshold": 0.1,
                      "delay": 1,
                      "nb_max_models": 5}


agent_parameters = set_param_agent(params_shared,
                                   params_finite_horizon,
                                   params_multi_model)

# res, param = main_function(agents_to_test,
#                            env_to_test,
#                            nb_tests,
#                            play_parameters,
#                            starting_seed,
#                            env_parameters,
#                            agent_parameters,
#                            nb_processes=nb_proc)

# get_all_plot(res, param)


# ---------------------------------------------------------------------------- #
# Maze Environment Back to initial position
# Exp 2
# ---------------------------------------------------------------------------- #

agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon10',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxFiniteHorizon30',
                  'SoftmaxMultiModel']



play_parameters = {'trials': 1000,
                   'max_step': 100}

env_parameters = []

env_to_test = 'PartiallyChangingCrossEnvironment'
for i in range(10):
    env_parameters.append({'number': i,
                           'step_change': 2000,
                           'conds': ['', '_D'],
                           'value_change': 0.2,
                           'uncertain': True})


starting_seed = generate_seed(17)
nb_tests = 5
nb_proc = 10

# Parameters agents
params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 3,
                 "gamma": 0.95}

params_finite_horizon = {'horizon': 50}

params_multi_model = {"horizon": 10,
                      "kl_threshold": 1.,
                      "merging_threshold": 0.1,
                      "delay": 1,
                      "nb_max_models": 5}


agent_parameters = set_param_agent(params_shared,
                                   params_finite_horizon,
                                   params_multi_model)

# res, param = main_function(agents_to_test,
#                            env_to_test,
#                            nb_tests,
#                            play_parameters,
#                            starting_seed,
#                            env_parameters,
#                            agent_parameters,
#                            nb_processes=nb_proc)

# get_all_plot(res, param)
