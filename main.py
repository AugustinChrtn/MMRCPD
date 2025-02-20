from play_function import main_function
from plots import get_all_plot


def set_param_agent(params_shared={},
                    params_finite_horizon={},
                    params_multi_model={},
                    params_finite_horizon2={}):

    params_VI_softmax = {**params_shared}

    params_SoftmaxFiniteHorizon = {**params_shared,
                                   **params_finite_horizon}

    params_SoftmaxFiniteHorizon2 = {**params_shared,
                                    **params_finite_horizon2}

    params_SoftmaxMultiModel = {**params_shared,
                                **params_multi_model}

    agent_parameters = {'VI_softmax': params_VI_softmax,
                        'SoftmaxFiniteHorizon': params_SoftmaxFiniteHorizon,
                        'SoftmaxFiniteHorizon5': params_SoftmaxFiniteHorizon,
                        'SoftmaxFiniteHorizon20': params_SoftmaxFiniteHorizon2,
                        'SoftmaxMultiModel': params_SoftmaxMultiModel,
                        }

    return agent_parameters


def generate_seed(number_of_the_experiment):
    return 1000*number_of_the_experiment+1


# ---------------------------------------------------------------------------- #
# Experiments
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# MAB
# Exp1 - Test
# ---------------------------------------------------------------------------- #

# agents_to_test = ['ThompsonBernouilliFiniteHorizon',
#                   'SoftmaxFiniteHorizon',
#                   'ThompsonBernouilli']

# env_to_test = 'MAB'
# play_parameters = {'trials': 1000,
#                    'max_step': 1}
# env_parameters = [{'number_arms':4,
#                    'step_change':100}]


# starting_seed = generate_seed(0)
# nb_tests = 10


# nb_proc = 10

# # Parameters agents
# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 10,
#                  "gamma": 0.95}

# params_finite_horizon = {'horizon': 5}

# params_finite_horizon2 = {'horizon': 20}

# params_multi_model = {"horizon": 5,
#                       "kl_threshold": 0.2,
#                       "merging_threshold": 0.2,
#                       "delay": 1,
#                       "nb_max_models": 5}



# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_multi_model,
#                                    params_finite_horizon2)

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
# Exp 1 - Basic task
# ---------------------------------------------------------------------------- #
agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon5',
                  'SoftmaxFiniteHorizon20',
                  'SoftmaxMultiModel']

env_to_test = 'ThreeStates'
play_parameters = {'trials': 2000,
                   'max_step': 1}
env_parameters = [{"slip": 0.1,
                   "step_change": 50}]


starting_seed = generate_seed(0)
nb_tests = 100


nb_proc = 10

# Parameters agents
params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 10,
                 "gamma": 0.95}

params_finite_horizon = {'horizon': 5}

params_finite_horizon2 = {'horizon': 20}

params_multi_model = {"horizon": 5,
                      "kl_threshold": 0.3,
                      "merging_threshold": 0.2,
                      "delay": 1,
                      "nb_max_models": 5}


# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_multi_model,
#                                    params_finite_horizon2)

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
# Exp 2 with uncertain reward - Not in the article
# ---------------------------------------------------------------------------- #
# starting_seed = generate_seed(1)

# env_parameters = [{"slip": 0.1,
#                    "step_change": 50,
#                    "uncertain": True}]
# nb_tests = 500

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_multi_model,
#                                    params_finite_horizon2)

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

# starting_seed = generate_seed(42)

# env_parameters = [{"slip": 0.1,
#                    "step_change": 50,
#                    "uncertain": False}]
# nb_tests = 100

# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 1,
#                  "gamma": 0.95}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_multi_model,
#                                    params_finite_horizon2)

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

# starting_seed = generate_seed(43)

# env_parameters = [{"slip": 0.,
#                    "step_change": 50,
#                    "uncertain": False}]
# nb_tests = 100

# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 10,
#                  "gamma": 0.95}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_multi_model,
#                                    params_finite_horizon2)

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
# Exp 1 with small change
# ---------------------------------------------------------------------------- #

agents_to_test = ['VI_softmax',
                  'SoftmaxFiniteHorizon',
                  'SoftmaxMultiModel']
agents_to_test = [
                  'SoftmaxMultiModel']

env_to_test = 'ChainProblem'
# play_parameters = {'trials': 500,
#                    'max_step': 50}

# env_parameters = [{"slip": 0.1,
#                   "size_chain": 5,
#                    "step_change": 500,
#                    "changes": ['S']}]


starting_seed = generate_seed(2)
nb_tests = 20


# nb_proc = 10

# # Parameters agents
# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 5,
#                  "gamma": 0.95}
# params_finite_horizon = {'horizon': 10}
# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 1,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}


# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_multi_model)

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
# Exp 2 with total change
# ---------------------------------------------------------------------------- #

# env_to_test = 'ChainProblem'
# play_parameters = {'trials': 1000,
#                    'max_step': 50}

# env_parameters = [{"slip": 0.1,
#                   "size_chain": 5,
#                    "step_change": 2000,
#                    "changes": ['T']}]


# starting_seed = generate_seed(3)
# nb_proc = 10

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
# Exp 3 with h=3
# ---------------------------------------------------------------------------- #

play_parameters = {'trials': 500,
                   'max_step': 50}

env_parameters = [{"slip": 0.1,
                  "size_chain": 5,
                   "step_change": 500,
                   "changes": ['S']}]


starting_seed = generate_seed(4)
nb_tests = 20
nb_proc = 10

# Parameters agents
params_shared = {"threshold_VI": 1e-3,
                 "max_iterations": 1000,
                 "step_update": 1,
                 "beta": 5,
                 "gamma": 0.95}
params_finite_horizon = {'horizon': 3}
params_multi_model = {"horizon": 3,
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
# Exp 4 with h=20
# ---------------------------------------------------------------------------- #

# starting_seed = generate_seed(5)
# nb_tests = 20
# nb_proc = 10

# params_finite_horizon = {'horizon': 20}
# params_multi_model = {"horizon": 20,
#                       "kl_threshold": 1,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Exp 5 with h=20 and kl_threshold = 0.5
# ---------------------------------------------------------------------------- #


# starting_seed = generate_seed(6)
# nb_tests = 20
# nb_proc = 10

# params_finite_horizon = {'horizon': 20}
# params_multi_model = {"horizon": 20,
#                       "kl_threshold": 0.5,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Exp 6 with kl_treshold=0.3
# ---------------------------------------------------------------------------- #

# starting_seed = generate_seed(7)
# nb_tests = 20
# nb_proc = 10

# params_finite_horizon = {'horizon': 10}
# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 0.3,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Exp 7 with kl_treshold=1.5
# ---------------------------------------------------------------------------- #

# starting_seed = generate_seed(8)
# nb_tests = 20
# nb_proc = 10

# params_finite_horizon = {'horizon': 10}
# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 1.5,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Exp 8 with kl_treshold=0.3 merging threshold = 0.01
# ---------------------------------------------------------------------------- #

# starting_seed = generate_seed(9)
# nb_tests = 20
# nb_proc = 10

# params_finite_horizon = {'horizon': 10}
# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 0.3,
#                       "merging_threshold": 0.01,
#                       "delay": 1,
#                       "nb_max_models": 5}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Exp 9 with kl_treshold=0.3 merging threshold = 0.3
# ---------------------------------------------------------------------------- #


# starting_seed = generate_seed(10)
# nb_tests = 20
# nb_proc = 10

# params_finite_horizon = {'horizon': 10}
# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 0.3,
#                       "merging_threshold": 0.3,
#                       "delay": 1,
#                       "nb_max_models": 5}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Exp 10 with kl_treshold=0.3 merging_treshold = 0
# ---------------------------------------------------------------------------- #

# starting_seed = generate_seed(11)
# nb_tests = 20
# nb_proc = 10

# params_finite_horizon = {'horizon': 10}
# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 0.3,
#                       "merging_threshold": 0.,
#                       "delay": 1,
#                       "nb_max_models": 5}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Exp 11 with kl_treshold=0.3, merging threshold = 0.01, max_mod = 2
# ---------------------------------------------------------------------------- #

# starting_seed = generate_seed(12)
# nb_tests = 20
# nb_proc = 10

# params_finite_horizon = {'horizon': 10}
# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 0.3,
#                       "merging_threshold": 0.01,
#                       "delay": 1,
#                       "nb_max_models": 2}

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Chain environment
# Exp 12 ALL agents small change
# ---------------------------------------------------------------------------- #

# agents_to_test = ['VI_softmax',
#                   'SoftmaxFiniteHorizon',
#                   'SoftmaxMultiModel',
#                   'RmaxNovelty',
#                   'RmaxExploration',
#                   'RmaxContext']

# env_to_test = 'ChainProblem'
# play_parameters = {'trials': 500,
#                    'max_step': 50}

# env_parameters = [{"slip": 0.1,
#                   "size_chain": 5,
#                    "step_change": 500,
#                    "changes": ['S']}]


# starting_seed = generate_seed(13)
# nb_tests = 20


# nb_proc = 10

# # Parameters agents
# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 5,
#                  "gamma": 0.95}
# params_finite_horizon = {'horizon': 10}
# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 1,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}
# params_finite_horizon2 = {}
# params_Nov = {'Rmax': 1}
# params_Explo = {'exploration_threshold': 0.3}
# params_Context = {'alpha_cov': 0.2}


# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Chain environment
# Exp 13 ALL agents big change
# ---------------------------------------------------------------------------- #

# agents_to_test = ['VI_softmax',
#                   'SoftmaxFiniteHorizon',
#                   'SoftmaxMultiModel',
#                   'RmaxNovelty',
#                   'RmaxExploration',
#                   'RmaxContext']

# env_to_test = 'ChainProblem'
# nb_tests = 20
# nb_proc = 10

# # Parameters agents
# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 5,
#                  "gamma": 0.95}
# params_finite_horizon = {'horizon': 10}
# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 1,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}
# params_finite_horizon2 = {}
# params_Nov = {'Rmax': 1}
# params_Explo = {'exploration_threshold': 0.3}
# params_Context = {'alpha_cov': 0.2}


# starting_seed = generate_seed(14)
# play_parameters = {'trials': 1000,
#                    'max_step': 50}
# env_parameters = [{"slip": 0.1,
#                   "size_chain": 5,
#                    "step_change": 2000,
#                    "changes": ['T']}]

# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_finite_horizon2,
#                                    params_multi_model,
#                                    params_Nov,
#                                    params_Explo,
#                                    params_Context)

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
# Cross Environment
# Exp 1
# ---------------------------------------------------------------------------- #


# agents_to_test = ['VI_softmax',
#                   'SoftmaxFiniteHorizon',
#                   'SoftmaxMultiModel']

# # agents_to_test = ['SoftmaxMultiModel']
# # agents_to_test = ['SoftmaxFiniteHorizon']


# # env_to_test = 'ChangingCrossEnvironment'
# env_to_test = 'PartiallyChangingCrossEnvironment'
# play_parameters = {'trials': 200,
#                    'max_step': 100}

# env_parameters = []
# for i in range(5):
#     env_parameters.append({'number': i,
#                            'step_change': 2e3,
#                            'conds': ['', '_C']})


# starting_seed = generate_seed(15)
# nb_tests = 10
# nb_proc = 11

# # Parameters agents
# params_shared = {"threshold_VI": 1e-3,
#                  "max_iterations": 1000,
#                  "step_update": 1,
#                  "beta": 5,
#                  "gamma": 0.95}
# params_finite_horizon = {'horizon': 10}

# params_multi_model = {"horizon": 10,
#                       "kl_threshold": 1.,
#                       "merging_threshold": 0.1,
#                       "delay": 1,
#                       "nb_max_models": 5}


# agent_parameters = set_param_agent(params_shared,
#                                    params_finite_horizon,
#                                    params_multi_model)

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
# Other
# Exp X
# ---------------------------------------------------------------------------- #


# agents_to_test = ['VI_softmax',
#                   'SoftmaxFiniteHorizon',
#                   'SoftmaxMultiModel',
#                   'RmaxNovelty',
#                   'RmaxExploration',
#                   'RmaxContext']


# possible_envs = ["ThreeStates", 'SocialTask',
#                  'ChangingCrossEnvironment', 'ChainProblem']


# env_parameters = []
# for i in range(1):
#     env_parameters.append({'number': i,
#                            'step_change': 2e3,
#                            'conds': ['', '_C']})

# Parameters environment
# env_parameters = [{"slip": 0.1,
#                   "size_chain": 5,
#                    "step_change": 500,
#                    "changes": ['S']}]
