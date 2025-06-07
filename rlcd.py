# import numpy as np


# class RLCD:

#     def __init__(self, environment,
#                  gamma=0.95,
#                  alpha=0.05,
#                  beta=0.1,
#                  horizon=10,
#                  max_iterations=10000,
#                  step_update=1,
#                  threshold_VI=1e-3,
#                  softmax_beta=3):

#         self.environment = environment
#         self.size_environment = len(environment.states)
#         self.size_actions = len(environment.actions)
#         self.gamma = gamma
#         self.softmax_beta = softmax_beta
#         self.horizon = horizon

#         self.counter = 0
#         self.threshold_VI = threshold_VI
#         self.max_iterations = max_iterations
#         self.step_update = step_update

#         #
#         self.threshold = np.log((1 - beta) / alpha)

#         # Initialize uniform model
#         self.models = [self._create_new_model()]
#         self.scores = np.zeros(1)
#         self.score_uniform = 0
#         self.current_model = 0
#         self._set_model(0)

#     def _create_new_model(self):
#         """Create all the useful matrices for a given model. Initialize it with
#         uniform transitions or eye transitions.
#         """
#         model = {}
#         shape_SA = (self.size_environment, self.size_actions)
#         shape_SAS = (self.size_environment,
#                      self.size_actions, self.size_environment)
#         shape_SAH = (self.size_environment,
#                      self.size_actions, self.horizon)
#         model["Q"] = np.zeros(shape_SA)
#         model["R"] = np.zeros(shape_SA)
#         model["Rsum"] = np.zeros(shape_SA)
#         model["R_VI"] = np.zeros(shape_SA)
#         model["nSA"] = np.zeros(shape_SA)
#         model["nSAS"] = np.zeros(shape_SAS)
#         model["R_horizon"] = np.zeros(shape_SAH)
#         model["nSA_horizon"] = np.zeros(shape_SAH, dtype='int')

#         model["tSAS"] = np.zeros(shape_SAS)
#         for a in range(self.size_actions):
#                 model["tSAS"][:, a, :] = np.eye(self.size_environment)
#         model["Q_probas"] = np.zeros(shape_SA)
#         return model

#     def _set_model(self, model_index):
#         """Change all of the matrices to the ones of the current model."""
#         m = self.models[model_index]
#         self.current_model = model_index
#         self.Q = m["Q"]
#         self.R = m["R"]
#         self.Rsum = m["Rsum"]
#         self.R_VI = m["R_VI"]
#         self.nSA = m["nSA"]
#         self.nSAS = m["nSAS"]
#         self.tSAS = m["tSAS"]
#         self.Q_probas = m["Q_probas"]
#         self.nSA_horizon = m["nSA_horizon"]
#         self.R_horizon = m["R_horizon"]

#     def _save_model(self, model_index):
#         m = self.models[model_index]
#         m["Q"] = self.Q
#         m["R"] = self.R
#         m["Rsum"] = self.Rsum
#         m["R_VI"] = self.R_VI
#         m["nSA"] = self.nSA
#         m["nSAS"] = self.nSAS
#         m["tSAS"] = self.tSAS
#         m["Q_probas"] = self.Q_probas
#         m["nSA_horizon"] = self.nSA_horizon
#         m["R_horizon"] = self.R_horizon

#     def choose_action(self, current_state):
#         """Softmax decision-making"""
#         max_Q_values = np.max(self.Q[current_state])
#         q_values_to_use = self.Q[current_state] - max_Q_values
#         exp_Q = np.exp(q_values_to_use * self.softmax_beta)
#         self.Q_probas[current_state, :] = exp_Q / np.sum(exp_Q)
#         action = np.random.choice(np.arange(self.size_actions),
#                                   p=self.Q_probas[current_state, :])
#         return action

#     def learn_the_model(self, old_state, reward, new_state, action):
#         '''Learns the model with respect to the horizon'''
#         self.nSA[old_state][action] += 1
#         self.nSAS[old_state][action][new_state] += 1
#         self.Rsum[old_state][action] += reward

#         ind_to_change = int(self.nSA[old_state][action] % self.horizon)

#         if self.nSA[old_state][action] > self.horizon:
#             state_to_forget = self.nSA_horizon[old_state][action][ind_to_change]
#             reward_to_substract = self.R_horizon[old_state][action][ind_to_change]

#             self.Rsum[old_state][action] -= reward_to_substract
#             self.nSAS[old_state][action][state_to_forget] -= 1

#         self.nSA_horizon[old_state][action][ind_to_change] = new_state
#         self.R_horizon[old_state][action][ind_to_change] = reward

#         self.normalization_factor = min(
#             self.nSA[old_state][action], self.horizon)
#         self.R[old_state][action] = self.Rsum[old_state][action] / \
#             self.normalization_factor

#     def learn(self, s, r, s_next, a):

#         self._detect_context(s, a, s_next)
#         self.learn_the_model(s, r, s_next, a)
#         self._compute_reward_VI(s, a)
#         self._compute_transitions(s, a)
#         self._value_iteration()

#     def _change_model(self, number_new_model):
#         old_model = self.current_model
#         self._save_model(old_model)
#         self._set_model(number_new_model)

#     def _compute_shifts(self, s, a ,s_next, r):

#     def _detect_context(self, s, a, s_next):
#         '''Check whether the context must be changed. If so, change contexts.
#         Uses transitions only.'''
#         t_current = self.tSAS[s, a, s_next]
#         epsilon = 1e-10
#         approx_t_current = t_current + epsilon
#         for i, model in enumerate(self.models):
#             if i != self.current_model:
#                 approx_t_stored = self.models[i]["tSAS"][s, a, s_next]+epsilon
#                 delta = np.log(approx_t_stored / approx_t_current)
#                 self.scores[i] += delta
#                 self.scores[self.scores < 0] = 0

#         best_index = np.argmax(self.scores)
#         best_score = np.max(self.scores)


#         t_uniform = 1/self.size_environment
#         self.score_uniform += np.log(t_uniform / approx_t_current)
#         self.score_uniform = max(self.score_uniform, 0)

#         print(self.score_uniform)

#         change_uniform = self.score_uniform > self.threshold
#         change_stored = self.scores[best_index] > self.threshold
#         if change_uniform or change_stored: #Condition to change model
#             if best_score > self.score_uniform : # change to a stored model
#                 nb_new_model = best_index
#             else : #change to a new model that has to be created
#                 new_model = self._create_new_model()
#                 self.models.append(new_model)
#                 self.scores = np.append(self.scores, 0)
#                 nb_new_model = len(self.models) - 1

#             # Actual model change
#             self._change_model(nb_new_model)

#             # Reset scores after switch
#             self.scores[:] = 0
#             self.score_uniform = 0

#     def _compute_transitions(self, s, a):
#         self.tSAS[s, a] = self.nSAS[s, a] / self.nSA[s, a]

#     def _compute_reward_VI(self, s, a):
#         self.R_VI[s, a] = self.R[s, a]

#     def _value_iteration(self):
#         self.counter += 1
#         if self.counter % self.step_update == 0:
#             threshold = self.threshold_VI
#             converged = False
#             nb_iters = 0
#             while (not converged and nb_iters < self.max_iterations):
#                 nb_iters += 1
#                 max_Q = np.max(self.Q, axis=1)
#                 new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)
#                 diff = np.abs(self.Q - new_Q)
#                 self.Q = new_Q
#                 if np.max(diff) < threshold:
#                     converged = True
import numpy as np


class RLCD:
    def __init__(self, environment,
                 horizon=10,
                 rho=0.3,
                 Omega=0.,
                 Emin=-0.15,
                 gamma=0.95,
                 Rmin=0,
                 Rmax=1,
                 beta=3,
                 threshold_VI=1e-3,
                 max_iterations=1000,
                 step_update=1):

        self.environment = environment
        self.size_environment = len(environment.states)
        self.size_actions = len(environment.actions)

        self.horizon = horizon
        self.rho = rho
        self.Omega = Omega
        self.Emin = Emin
        self.gamma = gamma
        self.Rmin = Rmin
        self.Rmax = Rmax

        self.beta = beta


        self.counter = 0
        self.threshold_VI = threshold_VI
        self.max_iterations = max_iterations
        self.step_update = step_update

        self.models = []
        self.E = []  # quality traces
        self.current_model = None
        self.shape_SA = (self.size_environment, self.size_actions)
        self.shape_SAS = (self.size_environment,
                          self.size_actions, self.size_environment)
        self.shape_SAH = (self.size_environment,
                          self.size_actions, self.horizon)

        self._new_model()

        self.step=0

    def _new_model(self):
        
        # tSAS = np.zeros(self.shape_SAS)
        # for action in range(self.size_actions):
        #         tSAS[:,action,:]=np.eye(self.size_environment)
        model = {
            "Q": np.zeros(self.shape_SA),
            "R": np.zeros(self.shape_SA),
            "Rsum": np.zeros(self.shape_SA),
            "R_VI": np.zeros(self.shape_SA),
            "nSA": np.zeros(self.shape_SA),
            "nSAS": np.zeros(self.shape_SAS),
            # uniform initialization
            "tSAS": np.ones(self.shape_SAS) / self.size_environment,
            #"tSAS":tSAS,
            "Q_probas": np.ones(self.shape_SA) / self.size_actions,
            "R_horizon": np.zeros(self.shape_SAH),
            "nSA_horizon": np.zeros(self.shape_SAH, dtype='int')
        }

        self.models.append(model)
        self.E.append(0.0)
        self._set_model(len(self.models) - 1)

    def _set_model(self, idx):
        model = self.models[idx]
        self.current_model = idx

        self.Q = model["Q"]
        self.R = model["R"]
        self.Rsum = model["Rsum"]
        self.R_VI = model["R_VI"]
        self.nSA = model["nSA"]
        self.nSAS = model["nSAS"]
        self.tSAS = model["tSAS"]
        self.Q_probas = model["Q_probas"]
        self.nSA_horizon = model["nSA_horizon"]
        self.R_horizon = model["R_horizon"]

    def choose_action(self, current_state):

        # if self.step == 1500 :
        #     print(self.tSAS[0,0])
        #     print(self.R)
        #     print(self.Q)

        self.step+=1
        """Softmax decision-making"""
        max_Q_values = np.max(self.Q[current_state])
        q_values_to_use = self.Q[current_state] - max_Q_values
        exp_Q = np.exp(q_values_to_use * self.beta)
        self.Q_probas[current_state, :] = exp_Q / np.sum(exp_Q)
        action = np.random.choice(np.arange(self.size_actions),
                                  p=self.Q_probas[current_state, :])
        return action

    def _learn_the_model(self, old_state, reward, new_state, action):
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

        self.normalization_factor = min(self.nSA[old_state][action],
                                        self.horizon)
        self.R[old_state][action] = self.Rsum[old_state][action] / \
            self.normalization_factor
        

    def learn(self, old_state, reward, new_state, action):
        self._detect_context(old_state, action, new_state, reward)
        self._learn_the_model(old_state, reward, new_state, action)

        self._compute_reward_VI(old_state, action)
        self._compute_transitions(old_state, action)
        self._value_iteration()

    def _compute_transitions(self, old_state, action):
        self.tSAS[old_state][action] = self.nSAS[old_state][action] / \
            self.normalization_factor

    def _compute_reward_VI(self, s, a):
        self.R_VI[s, a] = self.R[s, a]

    def _value_iteration(self):
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

    def _detect_context(self, s, a, s_next, r):
        for i, model in enumerate(self.models):

            t_alt = model["tSAS"][s, a, :]
            r_alt = model["R"][s, a]
            n_alt = model["nSA"][s, a]
            N_alt = min(n_alt + 1, self.horizon)

            consider_transitions = self.Omega!=1
            consider_rewards = self.Omega != 0

            if consider_transitions :
                # Compute delta_T
                target = np.zeros(self.size_environment)
                target[s_next] = 1.0
                delta_T = (target - t_alt) / N_alt
                
                # Compute eT
                ZT = 1/2 * N_alt**2
                eT = 1 - 2 * ZT * np.sum(delta_T**2)
            
            if consider_rewards :
                # Compute delta_R
                delta_R = (r - r_alt) / N_alt

                # Compute eR
                ZR = 1.0 / (self.Rmax - self.Rmin + 1e-10)
                eR = 1 - 2 * ZR * (delta_R ** 2)

            # Compute confidence
            cm = N_alt / self.horizon

            # Compute the weighted sum of eR and eT
            if not consider_transitions :
                em = cm*eR
            elif not consider_rewards : 
                em = cm*eT
            else :
                em = cm * (self.Omega * eR + (1 - self.Omega) * eT)

            # Update the quality trace
            self.E[i] *= (1-self.rho)
            self.E[i] += self.rho* em 


            
        best_idx = np.argmax(self.E)

        if self.E[best_idx] < self.Emin:
            self._new_model()
            # print("created new model at step ", self.step)
        else:
            self._set_model(best_idx)
            # print("changed current model")
