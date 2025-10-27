import numpy as np
# ---------------------------------------------------------------------------- #
# Infinite horizon agents
# ---------------------------------------------------------------------------- #
class Basic_MB:

    def __init__(self, environment, gamma=0.95):
        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.gamma = gamma
        self.shape_SA = (self.size_environment, self.size_actions)
        self.shape_SAS = (self.size_environment,
                          self.size_actions, self.size_environment)
        self.R = np.zeros(self.shape_SA)
        self.Rsum = np.zeros(self.shape_SA)
        # Reward for value iteration
        self.R_VI = np.zeros(self.shape_SA)

        self.nSA = np.zeros(self.shape_SA)
        self.nSAS = np.zeros(self.shape_SAS)

        # self.tSAS = np.ones(self.shape_SAS) / self.size_environment
        self.tSAS = np.zeros(self.shape_SAS)
        for action in range(self.size_actions):
                self.tSAS[:,action,:]=np.eye(self.size_environment)

        self.Q = np.zeros(self.shape_SA)

    def choose_action(self, state):
        q_values = self.Q[state]
        return np.random.choice(np.flatnonzero(q_values == np.max(q_values)))

    def learn_the_model(self, old_state, reward, new_state, action):
        self.nSA[old_state][action] += 1
        self.nSAS[old_state][action][new_state] += 1
        self.Rsum[old_state][action] += reward
        self.R[old_state][action] = self.Rsum[old_state][action] / \
            self.nSA[old_state][action]

    def learn(self, old_state, reward, new_state, action):

        self.learn_the_model(old_state, reward, new_state, action)
        self.compute_reward_VI(old_state, reward, action)
        self.compute_transitions(old_state, new_state, action)
        self.value_iteration()

    def compute_transitions(self, old_state, new_state, action):
        self.tSAS[old_state][action] = self.nSAS[old_state][action] / \
            self.nSA[old_state][action]

    def compute_reward_VI(self, old_state, reward, action):
        self.R_VI[old_state][action] = self.R[old_state][action]

    def value_iteration(self):
        threshold = 1e-3
        converged = False
        while not converged:
            max_Q = np.max(self.Q, axis=1)
            new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)

            diff = np.abs(self.Q - new_Q)
            self.Q = new_Q
            if np.max(diff) < threshold:
                converged = True

    def get_all_transitions(self):
        all_transitions = {}
        for state in range(self.size_environment):
            for action in range(self.size_actions):
                all_transitions[state,action]=[]
                if self.nSA[state][action]>0:
                    index = (state, action)
                    tSAS = self.nSAS[index] / self.nSA[index]
                    all_transitions[state,action].append(list(tSAS))
        # print(all_transitions)
        return all_transitions


class VI_softmax(Basic_MB):
    def __init__(self, environment,
                 gamma=0.95,
                 beta=1,
                 max_iterations=10000,
                 step_update=1,
                 threshold_VI=1e-3):
        super().__init__(environment, gamma)
        self.beta = beta
        self.Q_probas = np.ones(self.shape_SA) / self.size_actions
        self.max_iterations = max_iterations
        self.counter = 0
        self.step_update = step_update
        self.threshold_VI = threshold_VI

    def choose_action(self, current_state):
        max_Q_values = np.max(self.Q[current_state])
        q_values_to_use = self.Q[current_state] - max_Q_values
        exp_Q = np.exp(q_values_to_use * self.beta)
        self.Q_probas[current_state, :] = exp_Q / np.sum(exp_Q)
        action = np.random.choice(np.arange(self.size_actions),
                                  p=self.Q_probas[current_state, :])
        return action

    def value_iteration(self):
        self.counter += 1
        if self.counter % self.step_update == 0:
            threshold = self.threshold_VI
            converged = False
            nb_iters = 0
            while (not converged and nb_iters < self.max_iterations):
                nb_iters += 1
                max_Q = np.max(self.Q, axis=1)
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)
                diff = np.abs(self.Q - new_Q)
                self.Q = new_Q
                if np.max(diff) < threshold:
                    converged = True


# ---------------------------------------------------------------------------- #
# Finite horizon agents
# ---------------------------------------------------------------------------- #


class FiniteHorizonMB:

    def __init__(self, environment,
                 gamma=0.95,
                 horizon=10,
                 max_iterations=10000,
                 step_update=1,
                 threshold_VI=1e-3):

        # environment information
        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.shape_SA = (self.size_environment, self.size_actions)
        self.shape_SAS = (self.size_environment,
                          self.size_actions,
                          self.size_environment)

        # hyperparameters
        self.gamma = gamma
        self.horizon = horizon

        # Table initial values
        self.R = np.zeros(self.shape_SA)
        self.Rsum = np.zeros(self.shape_SA)
        self.R_VI = np.zeros(self.shape_SA)  # reward for value iteration
        self.nSA = np.zeros(self.shape_SA)
        self.nSAS = np.zeros(self.shape_SAS, dtype='int')

        # self.tSAS = np.ones(self.shape_SAS) / self.size_environment
        
        self.tSAS = np.zeros(self.shape_SAS)
        for action in range(self.size_actions):
                self.tSAS[:,action,:]=np.eye(self.size_environment)


        self.Q = np.zeros(self.shape_SA)

        # horizon tables
        self.shape_SAH = (self.size_environment,
                          self.size_actions, self.horizon)
        self.R_horizon = np.zeros(self.shape_SAH)
        self.nSA_horizon = np.zeros(self.shape_SAH, dtype='int')
        self.counter = 0
        self.threshold_VI = threshold_VI
        self.max_iterations = max_iterations
        self.step_update = step_update

    def learn(self, old_state, reward, new_state, action):
        '''main learning function'''
        self.learn_the_model(old_state, reward, new_state, action)
        self.compute_reward_VI(old_state, reward, action)
        self.compute_transitions(old_state, new_state, action)
        self.value_iteration()

    def choose_action(self, state):
        '''chooses an action with an argmax decision-making'''
        q_values = self.Q[state]
        return np.random.choice(np.flatnonzero(q_values == np.max(q_values)))

    def learn_the_model(self, old_state, reward, new_state, action):
        '''Learns the model with respect to the horizon'''
        self.nSA[old_state][action] += 1
        self.nSAS[old_state][action][new_state] += 1
        self.Rsum[old_state][action] += reward

        ind_to_change = int(self.nSA[old_state][action] % self.horizon)

        if self.nSA[old_state][action] > self.horizon:
            state_to_forget = self.nSA_horizon[old_state][action][ind_to_change]
            reward_to_substract = self.R_horizon[old_state][action][ind_to_change]

            self.Rsum[old_state][action] -= reward_to_substract
            self.nSAS[old_state][action][state_to_forget] -= 1

        self.nSA_horizon[old_state][action][ind_to_change] = new_state
        self.R_horizon[old_state][action][ind_to_change] = reward

        self.normalization_factor = min(
            self.nSA[old_state][action], self.horizon)
        self.R[old_state][action] = self.Rsum[old_state][action] / \
            self.normalization_factor

    def compute_transitions(self, old_state, new_state, action):
        self.tSAS[old_state][action] = self.nSAS[old_state][action] / \
            self.normalization_factor

    def compute_reward_VI(self, old_state, reward, action):
        self.R_VI[old_state][action] = self.R[old_state][action]

    def value_iteration(self):
        self.counter += 1
        if self.counter % self.step_update == 0:
            threshold = self.threshold_VI
            converged = False
            nb_iters = 0
            while (not converged and nb_iters < self.max_iterations):
                nb_iters += 1
                max_Q = np.max(self.Q, axis=1)
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)
                diff = np.abs(self.Q - new_Q)
                self.Q = new_Q
                if np.max(diff) < threshold:
                    converged = True
                    
    def get_all_transitions(self):
        all_transitions = {}
        for state in range(self.size_environment):
            for action in range(self.size_actions):
                all_transitions[state,action]=[]
                if self.nSA[state][action]>0:
                    index = (state, action)
                    tSAS = self.nSAS[state][action] / \
            self.normalization_factor
                    all_transitions[state,action].append(list(tSAS))
        return all_transitions

class Epsilon_MB_horizon(FiniteHorizonMB):
    def __init__(self, environment, gamma, horizon, epsilon):
        super().__init__(environment, gamma, horizon)
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.choice(self.environment.actions)
        else:
            q_values = self.Q[state]
            action = np.random.choice(
                np.flatnonzero(q_values == q_values.max()))
        return action


class SoftmaxFiniteHorizon(FiniteHorizonMB):

    def __init__(self,
                 environment,
                 gamma,
                 horizon,
                 max_iterations,
                 step_update,
                 threshold_VI,
                 beta):
        super().__init__(environment,
                         gamma,
                         horizon,
                         max_iterations,
                         step_update,
                         threshold_VI)
        self.beta = beta

    def choose_action(self, current_state):
        max_Q_values = np.max(self.Q[current_state])
        q_values_to_use = self.Q[current_state] - max_Q_values
        exp_Q = np.exp(q_values_to_use * self.beta)
        probas = exp_Q / np.sum(exp_Q)
        action = np.random.choice(np.arange(self.size_actions), p=probas)
        return action