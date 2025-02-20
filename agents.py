import numpy as np
import time
from scipy.special import gammaln, psi
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

        self.tSAS = np.ones(self.shape_SAS) / self.size_environment
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

                a = time.time()
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)
                b = time.time()
                # print("cost_one_comput = "+str(b-a))

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
        self.tSAS = np.ones(self.shape_SAS) / self.size_environment
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

                a = time.time()
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)
                b = time.time()
                # print("cost_one_comput = "+str(b-a))

                diff = np.abs(self.Q - new_Q)
                self.Q = new_Q
                if np.max(diff) < threshold:
                    converged = True


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

    # def choose_action(self, state):
    #     q_values = self.Q[state]
    #     action = np.random.choice(
    #             np.flatnonzero(q_values == q_values.max()))
    #     return action


class SoftmaxFiniteHorizon2(SoftmaxFiniteHorizon):
    pass


# ---------------------------------------------------------------------------- #
# Bayesian Agents
# ---------------------------------------------------------------------------- #
class ThompsonBernouilli:
    def __init__(self, environment):
        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.alpha = np.ones(self.size_actions)  # Success counts
        self.beta = np.ones(self.size_actions)   # Failure counts

    def choose_action(self, current_state):
        sampled_means = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_means)

    def learn(self, old_state, reward, new_state, action):
        self.alpha[action] += reward
        self.beta[action] += 1 - reward


class ThompsonBernouilliFiniteHorizon:
    def __init__(self, environment, horizon):
        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.alpha = np.ones(self.size_actions)  # Success counts
        self.beta = np.ones(self.size_actions)   # Failure counts

        self.nSA = np.zeros(self.size_actions)
        self.horizon = horizon
        self.R_horizon = np.zeros((self.size_actions, self.horizon))

    def choose_action(self, current_state):
        sampled_means = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_means)

    def learn(self, old_state, reward, new_state, action):

        self.nSA[action] += 1

        ind_to_change = int(self.nSA[action] % self.horizon)

        if self.nSA[action] > self.horizon:
            minus_reward = self.R_horizon[action, ind_to_change]
            self.alpha[action] -= minus_reward
            self.beta[action] -= 1-minus_reward

        self.R_horizon[action, ind_to_change] = reward
        self.alpha[action] += reward
        self.beta[action] += 1 - reward


class ThompsonMultiModel:

    def __init__(self, environment,
                 horizon,
                 kl_threshold,
                 merging_threshold,
                 nb_max_models,
                 delay):

        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)

        self.horizon = horizon

        # Multi-model parameters
        self.kl_threshold = kl_threshold
        self.horizon = horizon
        self.merging_threshold = merging_threshold
        self.nb_max_model = nb_max_models
        self.delay = delay

        # Basic tables
        self.nSA = np.zeros(self.size_actions)
        self.Rsum = np.zeros(self.size_actions)
        # Multi-model tables
        shape_all_A = (self.nb_max_model, self.size_actions)
        self.all_nSA = np.zeros(shape_all_A, dtype=int)
        self.all_Rsum = np.zeros(shape_all_A, dtype=int)

        # Last counts
        shape_last_A = (self.size_actions, horizon)
        self.last_rewards = np.zeros(shape_last_A)

        # Contains what models were used for the last k iterations
        self.last_used_models = np.zeros(shape_last_A, dtype=int)

        # Contains what model is currently used (0 at start)
        self.current_model = np.zeros(self.shape_A, dtype=int)

        # Contains the number of total models for each transition
        self.nb_models = np.ones(self.shape_A, dtype=int)

        # Contains the last time a model was created
        self.last_creation = np.zeros(self.shape_A, dtype=int)

        # Contains the last time the model changed for each action
        self.last_change = np.zeros(self.shape_A, dtype=int)

        # Index to know what index is used for the last counts
        self.rel_ind = np.zeros(self.shape_A, dtype=int)

        # General counters
        self.counter = 0  # +1 every action
        self.total_nSA = np.zeros(self.shape_A, dtype=int)
        self.total_R = 0  # used for debug
        self.model_changed = False  # whether a model has just changed or not

        # Store the kl for all current models / not used right now
        self.all_current_kl = np.zeros(self.shape_A)

        # General info for plotting purposes
        self.total_nb_models = self.size_environment*self.size_actions
        self.total_creation = 0
        self.total_forgetting = 0
        self.total_merging = 0

    def choose_action(self, current_state):
        sampled_means = np.random.beta(self.Rsum, self.nSA-self.Rsum)
        return np.argmax(sampled_means)

    def swap_model(self, old_state, action):
        '''Check whether models should be swapped and if so, swap them
        Return whether models need to be swapped or not.'''

        number_of_models = self.nb_models[old_state][action]

        if number_of_models == 1:  # if one model, no swap
            need_to_swap = False
        else:
            need_to_swap = self.try_to_swap(old_state, action)
        return need_to_swap

    def try_to_swap(self, old_state, action):
        '''Find the KL divergence of each model depending on the last 
        observations and return whether models need to be swapped or not.'''
        current_model = self.current_model[old_state][action]
        all_kl = []
        existing_models = self.find_existing_models(old_state, action)
        for model_number in existing_models:
            kl = self.from_distrib_to_kl(old_state,
                                         action,
                                         model_number)
            all_kl.append(kl)

        min_kl = np.min(all_kl)
        index_model = existing_models.index(current_model)

        current_model_is_not_best = all_kl[index_model] > min_kl
        no_model_creation = min_kl < self.kl_threshold
        need_to_swap = current_model_is_not_best and no_model_creation
        return need_to_swap

    def change_model(self, new_nb_model, state, action):
        '''Update all current values to use after a swap.'''
        self.last_change[action] = 0
        self.current_model[action] = new_nb_model

        new_nSA = self.all_nSA[new_nb_model][action]
        new_Rsum = self.all_Rsum[new_nb_model][action]

        self.nSA[action] = new_nSA
        self.Rsum[action] = new_Rsum

    def learn_the_model(self, reward, action):

        # General
        self.counter += 1
        self.last_change[action] += 1
        self.last_creation[action] += 1

        # Update last arrival state, reward and model used at the right index
        ind_rel = self.rel_ind[action]
        model_number = self.current_model[action]

        self.last_rewards[action][ind_rel] = reward
        self.last_used_models[action][ind_rel] = model_number

        # Update all counts
        self.total_nSA[action] += 1

        self.all_nSA[model_number][action] += 1
        self.nSA[model_number][action] += 1

        self.all_Rsum[model_number][action] += reward
        self.Rsum[action] += reward

        # Without enough experience for the current model, no creation or swap
        enough_experience = self.check_model_experience(model_number,
                                                        action)
        if enough_experience:
            model_swap = self.check_model_change(action)
            if not model_swap:
                self.model_created = self.check_model_creation(action)
            self.model_changed = model_swap or self.model_created

        # Merging part active at all step.
        merging = True
        while merging:
            merging = self.try_to_merge(action)

    def check_model_experience(self, model_number, action):
        '''Take a model, a state and an action and return whether the model
        has enough experience or not (>= self.horizon).'''
        nb_exp_model = self.all_nSA[model_number][action]
        enough_experience = nb_exp_model >= self.horizon
        return enough_experience

    def check_model_change(self, action):
        '''Check whether models need to be swapped and do it.'''
        cur_model = self.current_model[action]
        need_to_swap = self.swap_model(action)

        # If one model seems better than our current model, we find
        # the best model, since when, and rearrange the counts
        if need_to_swap:

            existing_models = self.find_existing_models(
                action)
            existing_models.remove(cur_model)

            new_model_number = self.swap_with_best_model(
                action,
                cur_model,
                existing_models)

            self.change_model(new_model_number, action)

        return need_to_swap

    def check_model_creation(self, action):
        '''Check whether a model needs to be created and do it.'''
        cur_model = self.current_model[action]
        cur_kl = self.from_distrib_to_kl(action,
                                         cur_model)

        # We checked for swapping before so this condition makes sure that
        # no model has a good enough KL div on last observations.
        model_creation = cur_kl > self.kl_threshold

        if model_creation:
            created_model_number = self.create_new_model(action)
            existing_models = [created_model_number]
            new_model_number = self.swap_with_best_model(
                action,
                cur_model,
                existing_models)

            self.update_counter_new_model(action)

            self.change_model(new_model_number, action)

        return model_creation

    def get_distrib_cur_model(self, action):
        ''' Get the histogram corresponding to the last states of the current
        model.'''
        current_model = self.current_model[action]

        last_one_mod, _ = self.get_last_one_model(
            action,
            current_model)
        alpha2 = 1+np.sum(last_one_mod)
        beta2 = 1+len(last_one_mod)-np.sum(last_one_mod)
        return alpha2, beta2

    def swap_with_best_model(self,
                             action,
                             current_model,
                             all_mods_to_test):

        current_model = self.current_model[action]

        # For finding the change point, we use the current model horizon
        last_count_one_model, indexes = self.get_last_one_model(
            action,
            current_model)

        # The best model and the index of change are found using log likelihood
        new_model, index_change = self.min_log_likelihood(
            action,
            current_model,
            last_count_one_model,
            models_to_test=all_mods_to_test)

        # Reassigns the counts depending on the change point and the new model
        self.reassign_counts(
            action,
            current_model,
            new_model,
            last_count_one_model,
            indexes,
            index_change)

        return new_model

    def learn(self, old_state, reward, new_state, action):
        '''Main learning function. Called by the function play at every step.'''
        self.learn_the_model(reward, action)
        self.update_relative_index(action)

    def update_relative_index(self, action):
        '''Relative index used for filling the horizon tables. Similar to 
        using a queue.'''
        self.rel_ind[action] += 1
        self.rel_ind[action] %= self.horizon

    # MODEL CREATION

    def get_number_new_model(self, action):
        '''Since models can be merged, the number of the new model can be any 
        number. We take the value corresponding to the first empty model.'''
        for index_model in range(self.nb_max_model):
            count = self.all_nSA[index_model][action]
            if count == 0:
                return index_model

    def create_new_model(self, action):
        '''Create an empty model.'''

        # If the agent reached its number of models threshold, it tries to merge
        # models. If it cannot, it forgets the least used model.
        if self.nb_models[action] == self.nb_max_model:
            # model_merged is not used in practice as we merge models before
            model_merged = self.try_to_merge(action)
            if not model_merged:
                self.forget_least_used(action)

        nb_new_model = self.get_number_new_model(action)

        return nb_new_model  # returns the number of the current model

    def update_counter_new_model(self, action):
        self.nb_models[action] += 1
        self.last_creation[action] = 0

        # Update plot counters
        self.total_creation += 1
        self.total_nb_models += 1

    # COUNTS

    def reassign_counts(self,
                        action,
                        former_model,
                        new_model,
                        last_count_one_model,
                        indexes,
                        index_change):

        # Get all the states to reassign depending on the index
        to_reassign = last_count_one_model[:index_change]

        for ind, state in enumerate(to_reassign):

            # Update count of the old model
            self.all_nSA[former_model][action] -= 1

            # Update count of the new model
            self.all_nSA[new_model][action] += 1

            # Update rewards for both models
            minus_reward = self.last_rewards[action][indexes[ind]]
            self.all_Rsum[former_model][action] -= minus_reward
            self.all_Rsum[new_model][action] += minus_reward

            # Model used
            self.last_used_models[action][indexes[ind]] = new_model

        # If the model is empty, forget it
        empty_model = self.all_nSA[former_model][action] == 0
        if empty_model:
            self.forget_model(former_model, action)

    def get_last_one_model(self, action, model):
        '''Get the last arrival states with the current model and the 
        corresponding indexes.'''

        last_models = self.last_used_models[action]
        last_r = self.last_rewards[action]
        index_model = []
        last_one_model = []

        # Fill lists of the last states used and corresponding indexes.
        # Stop when another model than the current one was used.
        for i in range(self.horizon):
            ind = self.rel_ind[action]-i

            if last_models[ind] == model:
                last_one_model.append(last_r[ind])
                index_model.append(ind)
            else:
                break

        # Used for debug
        if len(last_one_model) == 0:
            text = "Current model is not the last used. Index problem."
            raise ValueError(text)

        return last_one_model, index_model

    def count_to_all_distrib(self,
                             rewards,
                             count,
                             last_count_one_model,
                             size,
                             order,
                             norm=True):
        '''Take a given count (number of passages of a model), the last 
        observations and return all the distribution to compare depending on
        all the possible changepoints. Order indicates in what order to cut the
        distribution (whether the count is the old count or the new count.)'''
        new_count = count.copy()
        new_reward = rewards.copy()

        all_distrib = []
        for index in range(len(last_count_one_model)):
            reward_to_change = last_count_one_model[index]
            if order:
                new_count -= 1
                new_reward -= reward_to_change
                last_new_index = last_count_one_model[index:]
                # Used for debug
                if new_count < 0:
                    raise ValueError("Negative count.")
            else:
                new_count += 1
                new_reward += reward_to_change
                last_new_index = last_count_one_model[:index+1]

            count_last = self.last_to_count(size, last_new_index)

            distrib_new_count = np.array(new_count)/np.sum(new_count)
            if norm:  # Normalize the distribution, not used with log likelihood.
                distrib_last = count_last/np.sum(count_last)
            else:
                distrib_last = count_last
            all_distrib.append((distrib_new_count,
                                distrib_last))
        return all_distrib

    def last_to_count(self, size, arr):
        '''Take the last observations and return an histogram of them.'''
        unique_indices, counts = np.unique(arr, return_counts=True)
        unique_indices = unique_indices.astype(int)
        transformed_arr = np.zeros(size, dtype=int)
        transformed_arr[unique_indices] = counts
        return transformed_arr

    # KL DIV

    def from_distrib_to_kl(self, action, model_number):
        '''Compute the kl between the last transitions of the current model
        and the count of the model indicated. '''

        nb_experiences = self.all_nSA[model_number][action]
        alpha1 = self.all_Rsum[model_number][action]
        beta1 = nb_experiences - alpha1
        alpha2, beta2 = self.get_distrib_cur_model(action)
        if nb_experiences < self.delay:
            return 0
        else:
            kl = self.kl_div(alpha1, beta1, alpha2, beta2)
            return kl

    def kl_divergence_beta(alpha1, beta1, alpha2, beta2):
        # Log of Beta function differences
        log_beta1 = gammaln(alpha1) + gammaln(beta1) - gammaln(alpha1 + beta1)
        log_beta2 = gammaln(alpha2) + gammaln(beta2) - gammaln(alpha2 + beta2)

        # KL divergence computation
        kl = (log_beta2 - log_beta1 +
              (alpha1 - alpha2) * psi(alpha1) +
              (beta1 - beta2) * psi(beta1) +
              (alpha2 - alpha1 + beta2 - beta1) * psi(alpha1 + beta1))

        return kl

    # LOG LIKELIHOOD

    def find_log_likelihood(self,
                            mod_number,
                            action,
                            last_count_one_model,
                            size,
                            order):
        '''Take a count and the last observations and returns the log likelihood
        for all change points.'''

        count = self.all_nSA[mod_number][action]
        reward = self.all_Rsum[mod_number][action]

        all_distrib = self.count_to_all_distrib(reward,
                                                count,
                                                last_count_one_model,
                                                size,
                                                order,
                                                norm=False)
        all_LL = []
        for distrib in all_distrib:
            p = distrib[0]
            events = distrib[1]
            LL_value = self.compute_likelihood(probas=p, events=events)
            all_LL.append(LL_value)
        return np.array(all_LL)

    def compute_likelihood(self, probas, events, epsilon=1e-5):
        probas = probas+epsilon
        probas /= np.sum(probas)
        log_array = -np.log(probas)
        log_array[log_array < 0] = 0
        log_likelihood = np.sum(log_array*events)
        return log_likelihood

    def min_log_likelihood(self,
                           old_state,
                           action,
                           model_number,
                           last_count_one_model,
                           models_to_test):
        '''Depending on all possible models and change points, returns the 
        best model and the best change point using the minimum negative
        log likelihood.'''
        size = self.size_environment
        old_count = self.all_nSAS[model_number][old_state][action].copy()

        LL_old = self.find_log_likelihood(old_count,
                                          last_count_one_model,
                                          size,
                                          order=True)
        all_LL = np.zeros((len(models_to_test), len(LL_old)))

        LL_old = np.concatenate([LL_old, [0]])
        for ind, new_mod_number in enumerate(models_to_test):
            new_count = self.all_nSAS[new_mod_number][old_state][action].copy()

            LL_new = self.find_log_likelihood(new_count,
                                              last_count_one_model,
                                              size,
                                              order=False)
            LL_new = np.concatenate([[0], LL_new])

            # The new model will contain the last observation (hence the 1:)
            all_LL[ind] = LL_old[1:]+LL_new[1:]
            print("LL_new", LL_new)
        model_ind, position_change = np.argwhere(all_LL == np.min(all_LL))[0]
        best_model = models_to_test[model_ind]
        position_change += 1

        return best_model, position_change

    def find_existing_models(self, action):
        '''Find all non empty models for a given state action.'''
        existing_models = []
        for i in range(self.nb_max_model):
            if self.all_nSA[i][action] > 0:
                existing_models.append(i)

        return existing_models

    # FORGETTING MODELS

    def forget_least_used(self, action):
        '''When we reach the threshold and NO MODEL CAN BE MERGED, we
        forget the one with the least experience, which was not used very
        recently.'''
        all_nSA = self.all_nSA[:, action]
        two_smallest = np.argpartition(all_nSA, 1)[:2]
        smallest = two_smallest[0]
        second_smallest = two_smallest[1]

        if self.current_model[action] != smallest:
            mod_to_forget = smallest
        else:
            mod_to_forget = second_smallest

        self.forget_model(mod_to_forget, action)

    def forget_model(self, model_number, action):
        '''Forgets a model that is not used currently.'''
        if self.current_model[action] != model_number:
            self.all_alpha[model_number][action] = 1
            self.all_beta[model_number][action] = 1

            self.nb_models[action] -= 1
            self.total_forgetting += 1
            self.total_nb_models -= 1

            # Change the last used models to an unused number

            ind_never_used = self.nb_max_model+1
            last_mods = self.last_used_models[action]
            last_mods[last_mods == model_number] = ind_never_used

        # Used for debug
        else:
            raise ValueError("Impossible to forget the current model.")

    # MERGING MODELS
    def from_mod_number_to_jensen_div(self, ind1, ind2, action):
        '''Take two models and compute their jensen-shannon divergence. Used
        to find whether they can be merged.'''

        alpha1 = self.all_alpha[ind1][action]
        alpha2 = self.all_alpha[ind2][action]

        beta1 = self.all_beta[ind1][action]
        beta2 = self.all_beta[ind2][action]

        sum_alpha = alpha1+alpha2
        sum_beta = beta1+beta2

        jen1 = self.kl_divergence_beta(alpha1,
                                       beta1,
                                       sum_alpha,
                                       sum_beta)
        jen2 = self.kl_divergence_beta(alpha2,
                                       beta2,
                                       sum_alpha,
                                       sum_beta)
        jen = 1/2*(jen1+jen2)
        return jen

    def try_to_merge_with_couples(self, action, couples_to_test):
        all_divergences = []
        success = False

        cur_mod = self.current_model[action]
        for (ind1, ind2) in couples_to_test:

            # Computing the jensen-shannon divergence to check for merging
            div = self.from_mod_number_to_jensen_div(ind1,
                                                     ind2,
                                                     action)

            all_divergences.append(div)

        min_div = np.min(all_divergences)
        # Find the best couple (the one with the minimum divergence.)
        argmin_div = np.argmin(all_divergences)

        if min_div < self.merging_threshold:
            couple_to_merge = couples_to_test[argmin_div]

            self.merge_model(couple_to_merge[0],
                             couple_to_merge[1],
                             action)
            success = True

            self.total_merging += 1
            self.total_nb_models -= 1

            if cur_mod in couple_to_merge and self.model_created:
                self.total_merging -= 1
                self.total_creation -= 1

        return success

    def try_to_merge(self, action):
        '''Find if two models can be merged and merge them together.'''
        success = False

        # Models cannot be merged if there is only one model.
        only_one_model = self.nb_models[action] <= 1

        # Current model cannot be merged if it does not have enough experience.
        two_models = self.nb_models[action] == 2
        cur_model = self.current_model[action]
        enough_exp_cur_model = self.check_model_experience(cur_model,
                                                           action)
        two_models_not_enough_exp = two_models and (not enough_exp_cur_model)
        # if only_one_model or two_models_not_enough_exp:
        #     return success  # False
        if only_one_model:
            return success
        existing_models = self.find_existing_models(action)
        # Taking out the current model if not enough experience
        # if not enough_exp_cur_model:
        #     existing_models.remove(cur_model)

        couples_to_test = self.comb2(existing_models)
        success = self.try_to_merge_with_couples(action,
                                                 couples_to_test)

        return success

    def comb2(self, all_mod):
        '''Take a list and return all the (unordered) possible couples 
        made with elements of the list. '''
        list_of_comb2 = []
        for i in range(len(all_mod)):
            for j in range(i+1, len(all_mod)):
                list_of_comb2.append((all_mod[i], all_mod[j]))
        return list_of_comb2

    def merge_model(self, mod_to_keep, mod_to_add, action):
        '''Take two models and merge them count/reward-wise'''

        # Transitions
        count_passage = self.all_nSA[mod_to_add][action]
        self.all_nSA[mod_to_keep][action] += count_passage
        self.all_nSA[mod_to_add][action] = 0

        # Rewards
        alphas_to_add = self.all_alpha[mod_to_add][action]
        betas_to_add = self.all_beta[mod_to_add][action]

        self.all_alpha[mod_to_keep][action] += alphas_to_add
        self.all_beta[mod_to_keep][action] += betas_to_add

        self.all_alpha[mod_to_add][action] = 0
        self.all_beta[mod_to_add][action] = 0

        cur_mod = self.current_model[action]
        cond_cur_mod_add = cur_mod == mod_to_add
        cond_cur_mod_keep = cur_mod == mod_to_keep
        if cond_cur_mod_add or cond_cur_mod_keep:
            self.change_model(mod_to_keep, action)

        for ind, mod in enumerate(self.last_used_models[action]):
            if mod == mod_to_add:
                self.last_used_models[action][ind] = mod_to_keep
        self.nb_models[action] -= 1
