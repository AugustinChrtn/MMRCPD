import numpy as np


class MBMultiModel():

    def __init__(self,
                 environment,
                 gamma,
                 horizon=10,
                 kl_threshold=3,
                 threshold_VI=1e-3,
                 max_iterations=1000,
                 step_update=1,
                 merging_threshold=0.1,
                 delay=10,
                 nb_max_models=5,
                 reassign=True,
                 semi_jensen=False
                 ):

        # RL parameters
        self.environment = environment
        self.gamma = gamma

        # Multi-model parameters
        self.kl_threshold = kl_threshold
        self.horizon = horizon
        self.merging_threshold = merging_threshold
        self.nb_max_model = nb_max_models
        self.delay = delay

        # Value Iteration
        self.threshold = threshold_VI
        self.max_iterations = max_iterations
        self.step_update = step_update

        # Ablation
        self.reassign = reassign
        self.semi_jensen = semi_jensen

        # Basic tables
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.shape_SA = (self.size_environment, self.size_actions)
        self.shape_SAS = (self.size_environment,
                          self.size_actions, self.size_environment)

        # Rewards
        self.R = np.zeros(self.shape_SA)
        self.Rsum = np.zeros(self.shape_SA)

        # Reward for value iteration
        self.R_VI = np.zeros(self.shape_SA)

        # Counts
        self.nSA = np.zeros(self.shape_SA, dtype=int)
        self.nSAS = np.zeros(self.shape_SAS, dtype=int)

        # Transitions
        # self.tSAS = np.ones(self.shape_SAS) / self.size_environment
        self.tSAS = np.zeros(self.shape_SAS)
        for action in range(self.size_actions):
            self.tSAS[:, action, :] = np.eye(self.size_environment)
        # Q-Table
        self.Q = np.zeros(self.shape_SA)

        # Multi-model tables
        shape_all_SAS = (self.nb_max_model, ) + self.shape_SAS
        shape_all_SA = (self.nb_max_model, ) + self.shape_SA
        self.all_nSAS = np.zeros(shape_all_SAS, dtype=int)
        self.all_nSA = np.zeros(shape_all_SA, dtype=int)

        self.all_Rsum = np.zeros(shape_all_SA)

        # Last counts
        shape_last_SA = self.shape_SA+(horizon,)
        self.last_nSAS = np.zeros(shape_last_SA, dtype=int)

        # Values of the last rewards
        self.last_rewards = np.zeros(shape_last_SA)

        # Contains what models were used for the last k iterations
        self.last_used_models = np.zeros(shape_last_SA, dtype=int)

        # Contains what model is currently used (0 at start)
        self.current_model = np.zeros(self.shape_SA, dtype=int)

        # Contains the number of total models for each transition
        self.nb_models = np.ones(self.shape_SA, dtype=int)

        # Contains the last time a model was created
        self.last_creation = np.zeros(self.shape_SA, dtype=int)

        # Contains the last time the model changed for each transition
        self.last_change = np.zeros(self.shape_SA, dtype=int)

        # Contains all the times a change was detected
        self.nb_changes = 0

        # Index to know what index is used for the last counts
        self.rel_ind = np.zeros(self.shape_SA, dtype=int)

        # General counters
        self.counter = 0  # +1 every action
        self.total_nSA = np.zeros(self.shape_SA, dtype=int)
        self.total_R = 0  # used for debug
        self.model_changed = False  # whether a model has just changed or not

        # Store the kl for all current models / not used right now
        self.all_current_kl = np.zeros(self.shape_SA)

        # General info for plotting purposes
        self.total_nb_models = self.size_environment*self.size_actions
        self.total_creation = 0
        self.total_forgetting = 0
        self.total_merging = 0

        # Local info for plotting purpose
        self.creation_per_state = np.zeros(self.size_environment, dtype=int)
        self.model_per_state = self.size_actions*np.ones(self.size_environment,
                                                         dtype=int)

    def choose_action(self, state):
        '''Choose an action with argmax'''
        q_values = self.Q[state]
        return np.random.choice(np.flatnonzero(q_values == np.max(q_values)))

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
        self.last_change[state][action] = 0
        self.nb_changes += 1
        self.current_model[state][action] = new_nb_model

        new_nSAS = self.all_nSAS[new_nb_model][state][action]
        new_nSA = self.all_nSA[new_nb_model][state][action]
        self.nSAS[state][action] = new_nSAS
        self.nSA[state][action] = new_nSA

        self.compute_reward(state, action, new_nb_model)

    def learn_the_model(self, old_state, reward, new_state, action):

        # General
        self.counter += 1
        self.last_change[old_state][action] += 1
        self.last_creation[old_state][action] += 1

        # Update last arrival state, reward and model used at the right index
        ind_rel = self.rel_ind[old_state][action]
        model_number = self.current_model[old_state][action]

        self.last_nSAS[old_state][action][ind_rel] = new_state
        self.last_rewards[old_state][action][ind_rel] = reward
        self.last_used_models[old_state][action][ind_rel] = model_number

        # Update all counts
        self.total_nSA[old_state][action] += 1
        self.all_nSAS[model_number][old_state][action][new_state] += 1
        self.all_nSA[model_number][old_state][action] += 1
        self.nSA[old_state][action] += 1
        self.nSAS[old_state][action][new_state] += 1

        # Reward
        self.update_reward(old_state, reward, action, model_number)

        # Without enough experience for the current model, no creation or swap
        enough_experience = self.check_model_experience(model_number,
                                                        old_state,
                                                        action)
        if enough_experience:
            model_swap = self.check_model_change(old_state, action)
            if not model_swap:
                self.model_created = self.check_model_creation(
                    old_state, action)
            self.model_changed = model_swap or self.model_created

        # Merging part active at all step.
        merging = True
        while merging:
            merging = self.try_to_merge(old_state, action)

    def check_model_experience(self, model_number, old_state, action):
        '''Take a model, a state and an action and return whether the model
        has enough experience or not (>= self.horizon).'''
        nb_exp_model = self.all_nSA[model_number][old_state][action]
        enough_experience = nb_exp_model >= self.horizon
        return enough_experience

    def check_model_change(self, old_state, action):
        '''Check whether models need to be swapped and do it.'''
        cur_model = self.current_model[old_state][action]
        need_to_swap = self.swap_model(old_state, action)

        # If one model seems better than our current model, we find
        # the best model, since when, and rearrange the counts
        if need_to_swap:

            existing_models = self.find_existing_models(old_state,
                                                        action)
            existing_models.remove(cur_model)

            new_model_number = self.swap_with_best_model(old_state,
                                                         action,
                                                         cur_model,
                                                         existing_models)

            self.change_model(new_model_number, old_state, action)

        return need_to_swap

    def check_model_creation(self, old_state, action):
        '''Check whether a model needs to be created and do it.'''
        cur_model = self.current_model[old_state][action]
        cur_kl = self.from_distrib_to_kl(old_state,
                                         action,
                                         cur_model)

        # We checked for swapping before so this condition makes sure that
        # no model has a good enough KL div on last observations.
        model_creation = cur_kl > self.kl_threshold

        if model_creation:
            created_model_number = self.create_new_model(old_state, action)

            # # if the agent creates a model, it swaps with it.
            # existing_models = self.find_existing_models(old_state,
            #                                             action)
            # existing_models.remove(cur_model)
            existing_models = [created_model_number]
            new_model_number = self.swap_with_best_model(old_state,
                                                         action,
                                                         cur_model,
                                                         existing_models)

            self.update_counter_new_model(old_state, action)
            # if new_model_number == created_model_number:
            #     self.update_counter_new_model(old_state,action)
            # else :
            #     print("I did not create any model !")

            # print("Creation at step:",
            #       self.counter,
            #       " of model: ",
            #       new_model_number)

            self.change_model(new_model_number, old_state, action)

        return model_creation

    def get_distrib_cur_model(self, old_state, action):
        ''' Get the histogram corresponding to the last states of the current
        model.'''
        current_model = self.current_model[old_state][action]

        last_one_mod, _ = self.get_last_one_model(old_state,
                                                  action,
                                                  current_model)
        distrib = np.zeros(self.size_environment)

        for state in last_one_mod:
            distrib[state] += 1

        # rhos = [10/(i+1) for i in range(len(last_one_mod))]
        # for ind, state in enumerate(last_one_mod):
        #     distrib[state] += rhos[ind]+1

        # for ind, state in enumerate(last_one_mod):
        #     distrib[state] += len(last_one_mod)-ind+1
        return distrib

    def swap_with_best_model(self,
                             old_state,
                             action,
                             current_model,
                             all_mods_to_test):

        current_model = self.current_model[old_state][action]

        # For finding the change point, we use the current model horizon
        last_count_one_model, indexes = self.get_last_one_model(
            old_state,
            action,
            current_model)

        # The best model and the index of change are found using log likelihood
        new_model, index_change = self.min_log_likelihood(
            old_state,
            action,
            current_model,
            last_count_one_model,
            models_to_test=all_mods_to_test)

        # Reassigns the counts depending on the change point and the new model
        self.reassign_counts(old_state,
                             action,
                             current_model,
                             new_model,
                             last_count_one_model,
                             indexes,
                             index_change)

        return new_model

    def learn(self, old_state, reward, new_state, action):
        '''Main learning function. Called by the function play at every step.'''
        self.learn_the_model(old_state, reward, new_state, action)
        self.compute_reward_VI(old_state, action)
        self.compute_transitions(old_state, action)
        self.value_iteration()
        self.update_relative_index(old_state, action)

    def update_relative_index(self, old_state, action):
        '''Relative index used for filling the horizon tables. Similar to 
        using a queue.'''
        self.rel_ind[old_state][action] += 1
        self.rel_ind[old_state][action] %= self.horizon

    # MODEL CREATION

    def get_number_new_model(self, old_state, action):
        '''Since models can be merged, the number of the new model can be any 
        number. We take the value corresponding to the first empty model.'''
        for index_model in range(self.nb_max_model):
            nSA = self.all_nSA[index_model][old_state][action]
            if nSA == 0:
                return index_model

    def create_new_model(self, old_state, action):
        '''Create an empty model.'''

        # If the agent reached its number of models threshold, it tries to merge
        # models. If it cannot, it forgets the least used model.
        if self.nb_models[old_state][action] == self.nb_max_model:
            # model_merged is not used in practice as we merge models before
            model_merged = self.try_to_merge(old_state, action)
            if not model_merged:
                self.forget_least_used(old_state, action)

        nb_new_model = self.get_number_new_model(old_state, action)
        # self.nb_models[old_state][action] += 1
        # self.last_creation[old_state][action] = 0

        # # Update plot counters
        # self.total_creation += 1
        # self.total_nb_models += 1

        return nb_new_model  # returns the number of the current model

    def update_counter_new_model(self, old_state, action):
        self.nb_models[old_state][action] += 1
        self.last_creation[old_state][action] = 0

        # Update plot counters
        self.total_creation += 1
        self.creation_per_state[old_state] += 1
        self.total_nb_models += 1
        self.model_per_state[old_state] += 1

    # TRANSITIONS

    def compute_transitions(self, old_state, action):
        '''Compute the frequentist transition.'''
        nSAS_SA = self.nSAS[old_state][action]
        nSA_SA = self.nSA[old_state][action]
        self.tSAS[old_state][action] = nSAS_SA / nSA_SA

        if np.sum(nSAS_SA) != nSA_SA:
            print("nSAS", nSAS_SA)
            print("sum nSAS", np.sum(nSAS_SA))
            print("nSA", nSA_SA)
            raise ValueError("nSA should be the sum of nSAS")

    # REWARD

    def update_reward(self, old_state, reward, action, model):
        '''Update the infinite memory reward.'''
        self.all_Rsum[model][old_state][action] += reward
        self.total_R += reward
        self.compute_reward(old_state, action, model)

    def compute_reward(self, old_state, action, model):
        self.Rsum[old_state][action] = self.all_Rsum[model][old_state][action]
        reward_sum = self.Rsum[old_state][action]
        nb_actions = self.nSA[old_state][action]
        self.R[old_state][action] = reward_sum / nb_actions

    def compute_reward_VI(self, old_state, action):
        # for action in range(self.size_actions):

        #     if self.nSA[old_state][action] >= self.horizon:
        #             self.R_VI[old_state][action] = self.R[old_state][action]
        #     else :
        #         self.R_VI[old_state][action] = np.max(self.R)
        self.R_VI[old_state][action] = self.R[old_state][action]

    # COUNTS

    def reassign_counts(self,
                        old_state,
                        action,
                        former_model,
                        new_model,
                        last_count_one_model,
                        indexes,
                        index_change):

        # Get all the states to reassign depending on the index
        if self.reassign:
            to_reassign = last_count_one_model[:index_change]

        else:
            to_reassign = [last_count_one_model[:index_change][0]]

        for ind, state in enumerate(to_reassign):

            # Update count of the old model
            self.all_nSAS[former_model][old_state][action][state] -= 1
            self.all_nSA[former_model][old_state][action] -= 1

            # Update count of the new model
            self.all_nSA[new_model][old_state][action] += 1
            self.all_nSAS[new_model][old_state][action][state] += 1

            # Update rewards for both models
            r = self.last_rewards[old_state][action][indexes[ind]]
            self.all_Rsum[former_model][old_state][action] -= r
            self.all_Rsum[new_model][old_state][action] += r

            # Model used
            self.last_used_models[old_state][action][indexes[ind]] = new_model

        # Used for debug
        # if self.all_Rsum[former_model][old_state][action] < -0.1:
        #     print("negative old_reward",
        #             self.all_Rsum[former_model][old_state][action])
        #     raise ValueError("The reward is negative")

        # If the model is empty, forget it
        if self.all_nSA[former_model][old_state][action] == 0:
            self.forget_model(former_model, old_state, action)

    def get_last_one_model(self, old_state, action, model):
        '''Get the last arrival states with the current model and the 
        corresponding indexes.'''

        last_models = self.last_used_models[old_state][action]
        last_states = self.last_nSAS[old_state][action]
        index_model = []
        last_one_model = []

        # Fill lists of the last states used and corresponding indexes.
        # Stop when another model than the current one was used.
        for i in range(self.horizon):
            ind = self.rel_ind[old_state][action]-i

            if last_models[ind] == model:
                last_one_model.append(last_states[ind])
                index_model.append(ind)
            else:
                break

        # Used for debug
        if len(last_one_model) == 0:
            text = "Current model is not the last used. Index problem."
            raise ValueError(text)

        return last_one_model, index_model

    def count_to_all_distrib(self,
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
        all_distrib = []
        for index in range(len(last_count_one_model)):
            state_to_change = last_count_one_model[index]
            if order:
                new_count[state_to_change] -= 1
                last_new_index = last_count_one_model[index:]
                # Used for debug
                if new_count[state_to_change] < 0:
                    raise ValueError(
                        "An arrival state has a negative number of passages.")
            else:
                new_count[state_to_change] += 1
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

    def from_distrib_to_kl(self, old_state, action, model_number):
        '''Compute the kl between the last transitions of the current model
        and the count of the model indicated. '''
        trans_model = self.all_nSAS[model_number][old_state][action]
        last_trans = self.get_distrib_cur_model(old_state, action)
        nb_experiences = np.sum(last_trans)
        if nb_experiences < self.delay:
            return 0
        else:

            norm = self.all_nSA[model_number][old_state][action]
            trans_model = trans_model / norm

            last_trans = last_trans/nb_experiences
            kl = self.kl_div(last_trans, trans_model)
            return kl

    def kl_div(self, p, q, epsilon=1e-5):
        p += epsilon
        q += epsilon
        p /= np.sum(p)
        q /= np.sum(q)
        kl = np.sum(p * np.log(p / q))
        return kl

    # LOG LIKELIHOOD

    def find_log_likelihood(self, count, last_count_one_model, size, order):
        '''Take a count and the last observations and returns the log likelihood
        for all change points.'''
        all_distrib = self.count_to_all_distrib(count,
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
            # print("LL_new",LL_new)

        model_ind, position_change = np.argwhere(all_LL == np.min(all_LL))[0]
        best_model = models_to_test[model_ind]
        position_change += 1

        # print("LL_old",LL_old)
        # print("all_LL", all_LL)
        # print("las_count",last_count_one_model)
        # print("position_change", position_change)
        # print("")
        return best_model, position_change

    def find_existing_models(self, old_state, action):
        '''Find all non empty models for a given state action.'''
        existing_models = []
        for i in range(self.nb_max_model):
            if self.all_nSA[i][old_state][action] > 0:
                existing_models.append(i)

        # Used for debug
        # if len(existing_models) != self.nb_models[old_state][action]:
        #     print("existing models", existing_models)
        #     print("number models", self.nb_models[old_state][action])
        #     raise ValueError("number of models is wrong")

        return existing_models

    # FORGETTING MODELS

    def forget_least_used(self, old_state, action):
        '''When we reach the threshold and NO MODEL CAN BE MERGED, we
        forget the one with the least experience, which was not used very
        recently.'''
        all_nSA = self.all_nSA[:, old_state, action]
        two_smallest = np.argpartition(all_nSA, 1)[:2]
        smallest = two_smallest[0]
        second_smallest = two_smallest[1]

        if self.current_model[old_state][action] != smallest:
            mod_to_forget = smallest
        else:
            mod_to_forget = second_smallest

        self.forget_model(mod_to_forget, old_state, action)
        # Used for debug
        # print("forgotten model: ", mod_to_forget, "in state:", old_state,
        # "and action: ", action)

    def forget_model(self, model_number, old_state, action):
        '''Forgets a model that is not used currently.'''
        if self.current_model[old_state][action] != model_number:
            self.all_nSA[model_number][old_state][action] = 0
            self.all_nSAS[model_number][old_state][action] = 0
            self.all_Rsum[model_number][old_state][action] = 0
            self.nb_models[old_state][action] -= 1
            self.total_forgetting += 1
            self.total_nb_models -= 1
            self.model_per_state[old_state] -= 1

            # Change the last used models to an unused number

            ind_never_used = self.nb_max_model+1
            last_mods = self.last_used_models[old_state][action]
            last_mods[last_mods == model_number] = ind_never_used

            # Used for debug
            # if ind_never_used in self.last_used_models[old_state][action]:
            #     print(self.last_used_models[old_state][action])

        # Used for debug
        else:
            raise ValueError("Impossible to forget the current model.")

    # MERGING MODELS

    def from_mod_number_to_semi_jensen_div(self, ind1, ind2, old_state, action):
        '''Take two models and compute their jensen-shannon divergence. Used
        to find whether they can be merged.'''

        count1 = self.all_nSAS[ind1][old_state][action].copy()
        count2 = self.all_nSAS[ind2][old_state][action].copy()

        norm1 = self.all_nSA[ind1][old_state][action]
        norm2 = self.all_nSA[ind2][old_state][action]

        distrib1 = count1 / norm1
        distrib2 = count2 / norm2
        distrib_sum = (count2+count1) / (norm1+norm2)

        jen1 = self.kl_div(distrib1, distrib_sum)
        jen2 = self.kl_div(distrib2, distrib_sum)
        jen = 1/2*(jen1+jen2)
        return jen

    def from_mod_number_to_jensen_div(self, ind1, ind2, old_state, action):
        '''Take two models and compute their jensen-shannon divergence. Used
        to find whether they can be merged.'''

        count1 = self.all_nSAS[ind1][old_state][action].copy()
        count2 = self.all_nSAS[ind2][old_state][action].copy()

        norm1 = self.all_nSA[ind1][old_state][action]
        norm2 = self.all_nSA[ind2][old_state][action]

        distrib1 = count1 / norm1
        distrib2 = count2 / norm2
        distrib_sum = 1/2*(distrib1+distrib2)

        jen1 = self.kl_div(distrib1, distrib_sum)
        jen2 = self.kl_div(distrib2, distrib_sum)
        jen = 1/2*(jen1+jen2)
        return jen

    def try_to_merge_with_couples(self, old_state, action, couples_to_test):
        all_divergences = []
        success = False

        cur_mod = self.current_model[old_state][action]
        for (ind1, ind2) in couples_to_test:

            # Computing the sym kl divergence to check for merging
            # div = self.from_mod_number_sym_kl(ind1, ind2, old_state, action)

            # Computing the jensen-shannon divergence to check for merging
            if not self.semi_jensen:
                div = self.from_mod_number_to_jensen_div(ind1,
                                                         ind2,
                                                         old_state,
                                                         action)
            else:
                div = self.from_mod_number_to_semi_jensen_div(ind1,
                                                              ind2,
                                                              old_state,
                                                              action)

            all_divergences.append(div)

        min_div = np.min(all_divergences)
        # Find the best couple (the one with the minimum divergence.)
        argmin_div = np.argmin(all_divergences)

        if min_div < self.merging_threshold:
            couple_to_merge = couples_to_test[argmin_div]

            self.merge_model(couple_to_merge[0],
                             couple_to_merge[1],
                             old_state,
                             action)
            success = True

            self.total_merging += 1
            self.total_nb_models -= 1
            self.model_per_state[old_state] -= 1

            if cur_mod in couple_to_merge and self.model_created:
                self.total_merging -= 1
                self.total_creation -= 1
                self.creation_per_state[old_state] -= 1

            # Used for debug

            # print("Counts of merged models:",
            #       self.all_nSA[couple_to_merge[0]][old_state, action],
            #       self.all_nSA[couple_to_merge[1]][old_state, action],
            #       "Merged models: ", couple_to_merge[0], couple_to_merge[1],
            #       "Current model: ", cur_model)

            # print("Merging at step: ", self.counter,
            #       " of models: ", couple_to_merge)

        return success

    def try_to_merge(self, old_state, action):
        '''Find if two models can be merged and merge them together.'''
        success = False

        # Models cannot be merged if there is only one model.
        only_one_model = self.nb_models[old_state][action] <= 1

        # Current model cannot be merged if it does not have enough experience.
        two_models = self.nb_models[old_state][action] == 2
        cur_model = self.current_model[old_state][action]
        enough_exp_cur_model = self.check_model_experience(cur_model,
                                                           old_state,
                                                           action)
        two_models_not_enough_exp = two_models and (not enough_exp_cur_model)
        # if only_one_model or two_models_not_enough_exp:
        #     return success  # False
        if only_one_model:
            return success
        existing_models = self.find_existing_models(old_state, action)
        # Taking out the current model if not enough experience
        # if not enough_exp_cur_model:
        #     existing_models.remove(cur_model)

        couples_to_test = self.comb2(existing_models)
        success = self.try_to_merge_with_couples(old_state,
                                                 action,
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

    def merge_model(self, mod_to_keep, mod_to_add, old_state, action):
        '''Take two models and merge them count/reward-wise'''

        # Transitions
        count_passage = self.all_nSA[mod_to_add][old_state][action]
        arrival_states = self.all_nSAS[mod_to_add][old_state][action]

        self.all_nSA[mod_to_keep][old_state][action] += count_passage
        self.all_nSAS[mod_to_keep][old_state][action] += arrival_states

        self.all_nSA[mod_to_add][old_state][action] = 0
        self.all_nSAS[mod_to_add][old_state][action] = np.zeros(
            self.size_environment)

        # Rewards

        rewards_to_add = self.all_Rsum[mod_to_add][old_state][action]
        self.all_Rsum[mod_to_keep][old_state][action] += rewards_to_add

        self.all_Rsum[mod_to_add][old_state][action] = 0

        cur_mod = self.current_model[old_state][action]
        cond_cur_mod_add = cur_mod == mod_to_add
        cond_cur_mod_keep = cur_mod == mod_to_keep
        if cond_cur_mod_add or cond_cur_mod_keep:
            self.change_model(mod_to_keep, old_state, action)

        for ind, mod in enumerate(self.last_used_models[old_state][action]):
            if mod == mod_to_add:
                self.last_used_models[old_state][action][ind] = mod_to_keep
        self.nb_models[old_state][action] -= 1

        # Used for debug
        # print("merged",
        #       "old_s:", old_state,
        #       "action:", action,
        #       "mod_to_keep", mod_to_keep,
        #       "mod_to_add", mod_to_add)

    # Value Iteration

    def value_iteration(self):
        # self.counter += 1
        if self.counter % self.step_update == 0:
            converged = False
            nb_iters = 0
            while (not converged and nb_iters < self.max_iterations):
                nb_iters += 1
                max_Q = np.max(self.Q, axis=1)
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)
                diff = np.abs(self.Q - new_Q)
                self.Q = new_Q
                if np.max(diff) < self.threshold:
                    converged = True


class EgreedyMultiModel(MBMultiModel):

    def __init__(self,
                 environment,
                 gamma,
                 epsilon=0.05,
                 horizon=10,
                 kl_threshold=3,
                 threshold_VI=1e-3,
                 max_iterations=1000,
                 step_update=1,
                 merging_threshold=0.1,
                 delay=10,
                 nb_max_models=5,
                 reassign=True,
                 semi_jensen=False):

        super().__init__(environment,
                         gamma,
                         horizon,
                         kl_threshold,
                         threshold_VI,
                         max_iterations,
                         step_update,
                         merging_threshold,
                         delay,
                         nb_max_models,
                         reassign,
                         semi_jensen)

        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.choice(self.environment.actions)
        else:
            q_values = self.Q[state]
            action = np.random.choice(
                np.flatnonzero(q_values == q_values.max()))
        return action


class SoftmaxMultiModel(MBMultiModel):

    def __init__(self,
                 environment,
                 gamma,
                 beta=0.05,
                 horizon=10,
                 kl_threshold=3,
                 threshold_VI=1e-3,
                 max_iterations=1000,
                 step_update=1,
                 merging_threshold=0.1,
                 delay=10,
                 nb_max_models=5,
                 reassign=True,
                 semi_jensen=False):

        super().__init__(environment,
                         gamma,
                         horizon,
                         kl_threshold,
                         threshold_VI,
                         max_iterations,
                         step_update,
                         merging_threshold,
                         delay,
                         nb_max_models,
                         reassign,
                         semi_jensen)

        self.beta = beta

    def choose_action(self, current_state):

        # Substracting by max to avoid high exponentials
        max_Q_values = np.max(self.Q[current_state])
        q_values_to_use = self.Q[current_state] - max_Q_values

        exp_Q = np.exp(q_values_to_use * self.beta)
        probas = exp_Q / np.sum(exp_Q)
        action = np.random.choice(np.arange(self.size_actions),
                                  p=probas)
        return action
