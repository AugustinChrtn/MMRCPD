from agents import SoftmaxFiniteHorizon, VI_softmax, SoftmaxFiniteHorizon2
from agents import ThompsonBernouilli, ThompsonBernouilliFiniteHorizon
from task_change_agents import SoftmaxMultiModel, RmaxExploration, RmaxNovelty
from context_change_agents import RmaxContext
from envs import ChainProblem, ChangingCrossEnvironment, ThreeStates, MAB
from envs import PartiallyChangingCrossEnvironment

# ---------------------------------------------------------------------------- #
# Agents definition
# ---------------------------------------------------------------------------- #


agents = {'VI_softmax': VI_softmax,
          'SoftmaxFiniteHorizon': SoftmaxFiniteHorizon,
          'SoftmaxFiniteHorizon5': SoftmaxFiniteHorizon,
          'SoftmaxFiniteHorizon20': SoftmaxFiniteHorizon,
          'SoftmaxMultiModel': SoftmaxMultiModel,
          'RmaxContext': RmaxContext,
          'RmaxExploration': RmaxExploration,
          'RmaxNovelty': RmaxNovelty,
          'ThompsonBernouilli': ThompsonBernouilli,
          'ThompsonBernouilliFiniteHorizon': ThompsonBernouilliFiniteHorizon
          }

agent_names = list(agents.keys())


# ---------------------------------------------------------------------------- #
# Environments definition
# ---------------------------------------------------------------------------- #


envs = {"ChainProblem": ChainProblem,
        "ChangingCrossEnvironment": ChangingCrossEnvironment,
        "ThreeStates": ThreeStates,
        "MAB": MAB,
        "PartiallyChangingCrossEnvironment":PartiallyChangingCrossEnvironment}


params_chain = {"slip": 0.1,
                "size_chain": 5,
                "step_change": 500,
                "changes": ['S']}

params_cross = {'number': 0,
                'step_change': 2e3,
                'conds': ['', '_C']}


params_three_states = {'slip': 0.1,
                       'step_change': 50}

params_partial_cross = {'number': 0,
                'step_change': 2e3,
                'conds': ['', '_C']}


env_to_param = {"ChainProblem": params_chain,
                "ChangingCrossEnvironment": params_cross,
                "PartiallyChangingCrossEnvironment":params_partial_cross,
                "ThreeStates": params_three_states}

env_names = list(envs.keys())
