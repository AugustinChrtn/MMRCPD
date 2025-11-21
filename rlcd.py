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

        self.step = 0

    def _new_model(self):
        # uniform initialization
        #tSAS = np.ones(self.shape_SAS) / self.size_environment

        # non-uniform initialization
        tSAS = np.zeros(self.shape_SAS)
        for action in range(self.size_actions):
                tSAS[:,action,:]=np.eye(self.size_environment)

        model = {
            "Q": np.zeros(self.shape_SA),
            "R": np.zeros(self.shape_SA),
            "Rsum": np.zeros(self.shape_SA),
            "R_VI": np.zeros(self.shape_SA),
            "nSA": np.zeros(self.shape_SA),
            "nSAS": np.zeros(self.shape_SAS),
            "tSAS": tSAS,
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

        self.step += 1
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

            consider_transitions = self.Omega != 1
            consider_rewards = self.Omega != 0

            if consider_transitions:
                # Compute delta_T
                target = np.zeros(self.size_environment)
                target[s_next] = 1.0
                delta_T = (target - t_alt) / N_alt

                # Compute eT
                ZT = 1/2 * N_alt**2
                eT = 1 - 2 * ZT * np.sum(delta_T**2)

            if consider_rewards:
                # Compute delta_R
                delta_R = (r - r_alt) / N_alt

                # Compute eR
                ZR = 1.0 / (self.Rmax - self.Rmin + 1e-10)
                eR = 1 - 2 * ZR * (delta_R ** 2)

            # Compute confidence
            cm = N_alt / self.horizon

            # Compute the weighted sum of eR and eT
            if not consider_transitions:
                em = cm*eR
            elif not consider_rewards:
                em = cm*eT
            else:
                em = cm * (self.Omega * eR + (1 - self.Omega) * eT)

            # Update the quality trace
            self.E[i] *= (1-self.rho)
            self.E[i] += self.rho * em

        best_idx = np.argmax(self.E)

        if self.E[best_idx] < self.Emin:
            self._new_model()
            # print("created new model at step ", self.step)
        else:
            self._set_model(best_idx)
            # print("changed current model")
