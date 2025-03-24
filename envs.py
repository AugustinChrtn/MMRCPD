import numpy as np

# ---------------------------------------------------------------------------- #
# Chain problem
# ---------------------------------------------------------------------------- #


class ChainProblem:

    def __init__(self,
                 slip=0.,
                 size_chain=5,
                 step_change=2000,
                 changes=['T']):

        # Defining basic variables
        self.number_states = size_chain
        self.number_actions = 2
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)
        self.step = 0
        self.step_change = step_change
        self.changes = changes

        # Last state is rewarded with 1, first state with 0.1
        self.rewards = np.zeros((self.number_states))
        self.rewards[0] = 0.1
        self.rewards[-1] = 1

        # The agent starts in position 0
        self.initial_state = 0
        self.agent_state = self.initial_state

        # Defining the transitions depending on the slipping probability
        self.slip_change(slip)

        self.nb_changes = 0

        self.all_transitions = {}
        for state in range(self.number_states):
            for action in range(self.number_actions):
                key = (state, action)
                self.all_transitions[key] = []
                self.all_transitions[key].append(list(self.transitions[key]))

    def slip_change(self, new_value):
        self.slip = new_value
        transitions_a = np.zeros((self.number_states, self.number_states))
        transitions_b = np.zeros((self.number_states, self.number_states))

        # Right action for all states
        for i in range(self.number_states-1):
            transitions_a[i, i+1] = 1-self.slip
            transitions_b[i, i+1] = self.slip
        # Right action for last state
        transitions_a[self.number_states-1, self.number_states-1] = 1-self.slip
        transitions_b[self.number_states-1, self.number_states-1] = self.slip
        transitions_b[:, 0] = 1-self.slip
        transitions_a[:, 0] = self.slip

        transitions = np.zeros((self.number_states,
                                self.number_actions,
                                self.number_states))
        transitions[:, 0, :] = transitions_a
        transitions[:, 1, :] = transitions_b
        self.transitions = transitions

    def check_new_model(self):
        for state in range(self.number_states):
            for action in range(self.number_actions):
                key = (state, action)
                if list(self.transitions[key]) not in self.all_transitions[key]:
                    self.all_transitions[key].append(
                        list(self.transitions[key]))

    def new_episode(self):
        if self.step % (self.step_change) == 0:

            # Finding the change cond number
            nb_change = (self.step // self.step_change)
            nb_change %= len(self.changes)
            nb_change = int(nb_change)-1
            self.nb_changes += 1

            # Applying the change
            change = self.changes[nb_change]
            if change == 'T':
                self.total_change()
            elif change == 'S':
                self.change_random_transition()
            elif 0 <= change <= 1:
                self.slip_change(change)
            else:
                raise ValueError(str(change) + " is not a good condition")

            # Adds the new models to all the true models the agent faced.
            self.check_new_model()

        self.agent_state = self.initial_state

    def total_change(self):
        # Invert completely the transition matrix
        tmp_transitions = self.transitions[:, 0].copy()
        self.transitions[:, 0] = self.transitions[:, 1].copy()
        self.transitions[:, 1] = tmp_transitions

    def change_random_transition(self):
        # Invert the transition matrix for one random state
        index_change = np.random.randint(self.number_states)
        tmp_transitions = self.transitions[index_change, 0].copy()
        self.transitions[index_change,
                         0] = self.transitions[index_change, 1].copy()
        self.transitions[index_change, 1] = tmp_transitions

    def make_step(self, action):
        self.step += 1
        transition_probas = self.transitions[self.agent_state][action]
        self.agent_state = np.random.choice(self.states, p=transition_probas)
        reward = self.rewards[self.agent_state]
        return reward, self.agent_state


# ---------------------------------------------------------------------------- #
# Basic Navigation environment
# ---------------------------------------------------------------------------- #


class NavigationEnv():

    def __init__(self, transitions, rewards, initial_state) -> None:
        self.transitions = transitions
        self.rewards = rewards
        self.initial_state = initial_state
        self.number_states = np.shape(transitions)[0]
        self.number_actions = np.shape(transitions)[1]
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)
        self.step = 0
        self.agent_state = self.initial_state
        # print('called grandparent class')

    def new_episode(self):
        self.agent_state = self.initial_state

    def make_step(self, action):
        self.step += 1
        transition_probas = self.transitions[self.agent_state][action]
        self.agent_state = np.random.choice(self.states, p=transition_probas)
        return self.rewards[self.agent_state], self.agent_state
# ---------------------------------------------------------------------------- #
# Environment with a cross in the middle
# ---------------------------------------------------------------------------- #


class CrossEnvironment(NavigationEnv):
    def __init__(self, number, type=''):
        path_transi = 'Env/Transitions/Transitions_' + \
            str(number) + type + '.npy'
        path_rewards = 'Env/Tables/Rewards_'+str(number) + '.npy'
        path_worlds = 'Env/Tables/World_'+str(number)+'.npy'
        transitions = np.load(path_transi, allow_pickle=True)
        rewards = np.load(path_rewards, allow_pickle=True)
        self.world = np.load(path_worlds, allow_pickle=True)
        self.world = self.world.flatten()
        initial_state = np.where(self.world == -2)[0][0]
        self.size = np.shape(transitions)[0]
        self.nb_actions = np.shape(transitions)[2]

        new_shape = (self.size**2, self.nb_actions, self.size**2)
        transitions = transitions.reshape(new_shape)
        rewards = rewards.reshape(self.size**2)
        # print('called parent class')
        super().__init__(transitions, rewards, initial_state)

    def twoD_to_oneD(self, state_2D):
        return state_2D[0]*self.size+state_2D[1]

    def one_to_twoD(self, state_1D):
        return (state_1D//self.size, state_1D % self.size)

# ---------------------------------------------------------------------------- #
# Non-stationary version of the cross-environment
# ---------------------------------------------------------------------------- #


class ChangingCrossEnvironment(CrossEnvironment):
    def __init__(self, number, step_change, conds=['', '_B']):
        self.nb_changes = 0
        self.step_change = step_change
        self.number = number
        self.conds = conds
        self.current_cond = 0
        self.total_counter = 0
        super().__init__(self.number, type=self.conds[self.current_cond])

        self.all_transitions = {}
        for state in range(self.number_states):
            for action in range(self.number_actions):
                key = (state, action)
                self.all_transitions[key] = []
                self.all_transitions[key].append(list(self.transitions[key]))

    def new_episode(self):
        change_cond = (self.total_counter //
                       self.step_change) % len(self.conds)
        if change_cond != self.current_cond:
            self.nb_changes += 1
            change_cond = int(change_cond)
            self.current_cond = change_cond
            new_cond = self.conds[change_cond]
            super().__init__(self.number, type=new_cond)

        self.agent_state = self.initial_state

    def make_step(self, action):
        self.step += 1
        self.total_counter += 1
        transition_probas = self.transitions[self.agent_state][action]

        # uncertainty
        # walls = self.world == -1
        # noise = np.random.random(size = len(transition_probas))
        # noise[walls] = 0
        # noise = noise/np.sum(noise)

        # noise = np.zeros(self.number_states)
        # noise[self.initial_state]=1
        # ratio = 0.1
        # transition_probas = (1-ratio)*transition_probas + ratio*noise
        # print(self.transitions[self.agent_state][action])
        # print(transition_probas)
        self.agent_state = np.random.choice(self.states, p=transition_probas)
        return self.rewards[self.agent_state], self.agent_state


class PartiallyChangingCrossEnvironment():

    def __init__(self,
                 number,
                 step_change,
                 conds=['', '_C'],
                 value_change=0.5,
                 uncertain=False):

        all_paths_transi = ['Env/Transitions/Transitions_' +
                            str(number) + typ + '.npy' for typ in conds]
        path_rewards = 'Env/Tables/Rewards_'+str(number) + '.npy'
        path_world = 'Env/Tables/World_'+str(number)+'.npy'
        all_transitions2 = [np.load(path_transi, allow_pickle=True)
                            for path_transi in all_paths_transi]
        rewards = np.load(path_rewards, allow_pickle=True)

        self.world = np.load(path_world, allow_pickle=True)
        self.world = self.world.flatten()
        self.initial_state = np.where(self.world == -2)[0][0]
        shape_all_transitions2 = np.shape(all_transitions2)
        self.size = shape_all_transitions2[1]
        self.number_states = self.size**2
        self.number_actions = shape_all_transitions2[3]
        new_shape = (self.number_states,
                     self.number_actions,
                     self.number_states)

        self.all_transitions2 = []
        for transi in all_transitions2:
            transi = transi.reshape((new_shape))
            self.all_transitions2.append(transi)

        self.all_transitions2 = np.stack(self.all_transitions2)
        self.transitions = self.all_transitions2[0].copy()
        self.rewards = rewards.reshape(self.number_states)
        self.nb_changes = 0
        self.step_change = step_change
        self.value_change = value_change
        self.number = number
        self.total_counter = 0

        self.states = np.arange(self.number_states)

        self.actions = np.arange(self.number_actions)
        self.step = 0
        self.agent_state = self.initial_state

        self.current_transis = np.zeros(self.number_states, dtype='int')

        self.conds = conds
        self.uncertain=uncertain

        self.all_transitions = {}
        for state in range(self.number_states):
            for action in range(self.number_actions):
                key = (state, action)
                self.all_transitions[key] = []
                self.all_transitions[key].append(list(self.transitions[key]))

    def twoD_to_oneD(self, state_2D):
        return state_2D[0]*self.size+state_2D[1]

    def one_to_twoD(self, state_1D):
        return (state_1D//self.size, state_1D % self.size)

    def check_new_model(self):
        for state in range(self.number_states):
            for action in range(self.number_actions):
                key = (state, action)
                if list(self.transitions[key]) not in self.all_transitions[key]:
                    self.all_transitions[key].append(
                        list(self.transitions[key]))

    def new_episode(self):
        change = (self.total_counter % self.step_change) == 0
        if change:
            self.nb_changes += 1
            total_elements = self.number_states
            if self.value_change == 1:
                random_indices = np.arange(total_elements)
            else:
                num_changes = int(self.value_change * self.number_states)
                random_indices = np.random.choice(total_elements,
                                                  num_changes,
                                                  replace=False)

            self.current_transis[random_indices] += 1
            self.current_transis %= len(self.conds)
            for index, state in enumerate(random_indices):
                new_transitions = self.all_transitions2[self.current_transis[index], state].copy(
                )
                self.transitions[state] = new_transitions

            self.check_new_model()
        self.agent_state = self.initial_state

    def make_step(self, action):
        # if self.rewards[self.agent_state]==1:
        #     self.agent_state = self.initial_state
        #     return 0, self.agent_state

        self.step += 1
        self.total_counter += 1
        transition_probas = self.transitions[self.agent_state][action]

        if self.uncertain :
            noise = np.zeros(self.number_states)
            noise[self.initial_state]=1
            ratio = 0.05
            transition_probas = (1-ratio)*transition_probas + ratio*noise

        self.agent_state = np.random.choice(self.states, p=transition_probas)
        reward = self.rewards[self.agent_state]
        return reward, self.agent_state


class SwappingCrossEnvironment(CrossEnvironment):
    def __init__(self, number, step_change, conds=['S']):
        self.step_change = step_change
        self.number = number
        self.conds = conds
        self.current_cond = 0
        self.total_counter = 0
        # print('called own class')
        super().__init__(self.number, type=self.conds[self.current_cond])

    def total_change(self):
        print('Step '+str(self.step)+': Total Task change')
        # Invert the two actions
        tmp_transitions = self.transitions[:, 0].copy()
        self.transitions[:, 0] = self.transitions[:, 1].copy()
        self.transitions[:, 1] = tmp_transitions

        tmp_rewards = self.rewards[:, 0].copy()
        self.rewards[:, 0] = self.rewards[:, 1].copy()
        self.rewards[:, 1] = tmp_rewards

    def change_random_transition(self):
        # print('Step '+str(self.step)+': Small Task change')
        index_change = np.random.randint(self.number_states)
        tmp_transitions = self.transitions[index_change, 0].copy()
        self.transitions[index_change,
                         0] = self.transitions[index_change, 1].copy()
        self.transitions[index_change, 1] = tmp_transitions

        tmp_rewards = self.rewards[index_change, 0].copy()
        self.rewards[index_change, 0] = self.rewards[index_change, 1].copy()
        self.rewards[index_change, 1] = tmp_rewards

    def new_episode(self):
        change_cond = (self.total_counter //
                       self.step_change) % len(self.conds)
        if change_cond != self.current_cond:
            self.nb_changes += 1
            change_cond = int(change_cond)
            self.current_cond = change_cond
            new_cond = self.conds[change_cond]
            super().__init__(self.number, type=new_cond)
            # print("Step" + str(self.total_counter) +
            #       ":Task change, new cond is "+str(new_cond))
            # print(self.step)
        self.agent_state = self.initial_state

    def make_step(self, action):
        self.step += 1
        self.total_counter += 1
        transition_probas = self.transitions[self.agent_state][action]
        self.agent_state = np.random.choice(self.states, p=transition_probas)
        return self.rewards[self.agent_state], self.agent_state

# ---------------------------------------------------------------------------- #
# Three states environment
# ---------------------------------------------------------------------------- #


class ThreeStates:

    def __init__(self, slip=0, step_change=2000, uncertain=False):

        self.slip = slip
        self.step_change = step_change
        self.uncertain = uncertain

        self.number_states = 3
        self.number_actions = 2
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)
        self.step = 0

        self.rewards = np.zeros((self.number_states))

        self.rewards[2] = 1
        self.initial_state = 0
        self.agent_state = self.initial_state

        # transitions_left = np.zeros((self.number_states, self.number_states))
        # transitions_right = np.zeros((self.number_states, self.number_states))
        # # transitions_left[0,1] = self.slip
        # # transitions_left[0,2] = 1-self.slip
        # # transitions_right[0,1] = 1-self.slip
        # # transitions_right[0,2] = self.slip

        # transitions_left[0, 1] = 0
        # transitions_left[0, 2] = 1
        # transitions_right[0, 1] = 1
        # transitions_right[0, 2] = 0

        # # transitions_left[0,1] = 1
        # # transitions_left[0,2] = 0
        # # transitions_right[0,1] = 0.99
        # # transitions_right[0,2] = 0.01

        # # Staying in position
        # transitions_left[1, 1] = 1
        # transitions_right[1, 1] = 1

        # transitions_left[2, 2] = 1
        # transitions_right[2, 2] = 1

        # transitions = np.zeros((self.number_states,
        #                         self.number_actions,
        #                         self.number_states))
        # transitions[:, 0, :] = transitions_left
        # transitions[:, 1, :] = transitions_right
        # self.transitions = transitions
        self.update_probas()

    def update_probas(self):
        transitions_left = np.zeros((self.number_states, self.number_states))
        transitions_right = np.zeros((self.number_states, self.number_states))

        transitions_left[0, 2] = self.slip
        transitions_left[0, 1] = 1-self.slip
        transitions_right[0, 2] = 1-self.slip
        transitions_right[0, 1] = self.slip


        # transitions_left[0, 2] = self.probas[self.cond][0]
        # transitions_left[0, 1] = 1-self.probas[self.cond][0]
        # transitions_right[0, 2] = self.probas[self.cond][1]
        # transitions_right[0, 1] = 1-self.probas[self.cond][1]

        # Staying in position
        transitions_left[1, 1] = 1
        transitions_right[1, 1] = 1

        transitions_left[2, 2] = 1
        transitions_right[2, 2] = 1

        transitions = np.zeros((self.number_states,
                                self.number_actions,
                                self.number_states))

        transitions[:, 0, :] = transitions_left
        transitions[:, 1, :] = transitions_right
        self.transitions = transitions
        # print(self.transitions)


    def new_episode(self):
        if self.step % (self.step_change) == 0:
            self.slip = 1-self.slip
            self.update_probas()

        # transitions_left = self.transitions[:,0,:].copy()
        # transitions_right = self.transitions[:,1,:].copy()

        # self.transitions[:,0,:] = transitions_right
        # self.transitions[:,1,:] = transitions_left

        self.agent_state = self.initial_state

    def make_step(self, action):
        self.step += 1
        # if np.random.random() < self.slip:
        #     action = (action+1) % 2

        transition_probas = self.transitions[self.agent_state][action]
        self.agent_state = np.random.choice(self.states, p=transition_probas)

        # Unfrequent reward
        if self.uncertain:
            if self.agent_state == 2 and np.random.random() < 0.1:
                reward = 10
            else:
                reward = 0

        else:
            reward = self.rewards[self.agent_state]
        return reward, self.agent_state


class DiffThreeStates:

    def __init__(self, step_change=2000, uncertain=False, probas=[[0.1, 0.2]]):

        self.probas = probas
        self.step_change = step_change
        self.uncertain = uncertain

        self.number_states = 3
        self.number_actions = 2
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)
        self.step = 0


        self.initial_state = 0
        self.agent_state = self.initial_state
        self.cond = 0
        self.update_probas()

    def update_probas(self):
        transitions_left = np.zeros((self.number_states, self.number_states))
        transitions_right = np.zeros((self.number_states, self.number_states))

        transitions_left[0, 2] = self.probas[self.cond][0]
        transitions_left[0, 1] = 1-self.probas[self.cond][0]
        transitions_right[0, 2] = self.probas[self.cond][1]
        transitions_right[0, 1] = 1-self.probas[self.cond][1]

        # Staying in position
        transitions_left[1, 1] = 1
        transitions_right[1, 1] = 1

        transitions_left[2, 2] = 1
        transitions_right[2, 2] = 1

        transitions = np.zeros((self.number_states,
                                self.number_actions,
                                self.number_states))

        transitions[:, 0, :] = transitions_left
        transitions[:, 1, :] = transitions_right
        self.transitions = transitions

        self.rewards = np.zeros((self.number_states))
        self.rewards[2] = 1/np.max(self.probas[self.cond])

    def new_episode(self):

        if self.step % (self.step_change) == 0:
            self.cond+=1
            self.cond%=len(self.probas)
            self.update_probas()

        self.agent_state = self.initial_state

    def make_step(self, action):
        self.step += 1
        transition_probas = self.transitions[self.agent_state][action]
        self.agent_state = np.random.choice(self.states, p=transition_probas)

        # Unfrequent reward
        if self.uncertain:
            if self.agent_state == 2 and np.random.random() < 0.05:
                reward = 20
            else:
                reward = 0

        else:
            reward = self.rewards[self.agent_state]
        return reward, self.agent_state
    



class FourStates:

    def __init__(self, step_change=2000, uncertain=False, slip=0.1):

        self.slip = slip
        self.step_change = step_change
        self.uncertain = uncertain
        self.swap = False

        self.number_states = 4
        self.number_actions = 2
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)
        self.step = 0


        self.initial_state = 0
        self.agent_state = self.initial_state
        self.cond = 0
        self.update_probas()

    def update_probas(self, swap=False):
        transitions_left = np.zeros((self.number_states, self.number_states))
        transitions_right = np.zeros((self.number_states, self.number_states))

        transitions_left[0, 2] = 1-self.slip
        transitions_left[0, 3] = self.slip
        transitions_right[0, 2] = self.slip
        transitions_right[0, 1] = 1-self.slip

        # Staying in position
        transitions_left[1, 1] = 1
        transitions_right[1, 1] = 1

        transitions_left[2, 2] = 1
        transitions_right[2, 2] = 1

        transitions_left[3, 3] = 1
        transitions_right[3, 3] = 1

        transitions = np.zeros((self.number_states,
                                self.number_actions,
                                self.number_states))

        # transitions[:, 0, :] = transitions_left
        # transitions[:, 1, :] = transitions_right

        if self.swap : 
            transitions[:, 0, :] = transitions_left
            transitions[:, 1, :] = transitions_right
        else :
            transitions[:, 0, :] = transitions_right
            transitions[:, 1, :] = transitions_left
        self.transitions = transitions

        self.rewards = np.zeros((self.number_states))
        self.rewards[3] = 10

    def new_episode(self):

        if self.step % (self.step_change) == 0:
            self.swap = not self.swap
            self.update_probas()

        self.agent_state = self.initial_state

    def make_step(self, action):
        self.step += 1
        transition_probas = self.transitions[self.agent_state][action]
        self.agent_state = np.random.choice(self.states, p=transition_probas)

        # Unfrequent reward
        # if self.uncertain:
        #     if self.agent_state == 2 and np.random.random() < 0.05:
        #         reward = 20
        #     else:
        #         reward = 0

        # else:
        reward = self.rewards[self.agent_state]
        return reward, self.agent_state


# ---------------------------------------------------------------------------- #
# Two-step-task
# ---------------------------------------------------------------------------- #
class TwoStepTask:

    def __init__(self):
        self.number_states = 5
        self.number_actions = 2
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)
        self.step = 0

        self.rewards = np.zeros((self.number_states, self.number_actions))

        reward_1 = 1
        reward_2 = 0.5

        self.rewards[:, 1] = 0.2
        self.rewards[-1, 0] = 1

        self.initial_state = 0
        self.agent_state = self.initial_state

        tmp_transitions = np.zeros(self.number_states)
        tmp_transitions[0] = 1
        transitions_a = np.zeros((self.number_states, self.number_states))
        transitions_b = np.zeros((self.number_states, self.number_states))
        transitions_b[:, 0] = 1
        for i in range(self.number_states-1):
            transitions_a[i, i+1] = 1
        transitions_a[self.number_states-1, self.number_states-1] = 1

        transitions = np.zeros(
            (self.number_states, self.number_actions, self.number_states))
        transitions[:, 0, :] = transitions_a
        transitions[:, 1, :] = transitions_b
        self.transitions = transitions
        # print(transitions)

    def new_episode(self):
        self.agent_state = self.initial_state

    def change_of_task(self):
        if self.step % (50*1500) == 0:
            # print('Total Task change')
            # Invert the two actions
            self.transitions[:, 0], self.transitions[:,
                                                     1] = self.transitions[:, 1], self.transitions[:, 0].copy()
            self.rewards[:, 1], self.rewards[:,
                                             0] = self.rewards[:, 0], self.rewards[:, 1].copy()

    def make_step(self, action):
        self.step += 1
        # print(action)
        # In the chain problem, the two actions are inverted with a 20% rate
        if np.random.random() < self.slip:
            action = (action+1) % 2

        reward = self.rewards[self.agent_state, action]
        transition_probas = self.transitions[self.agent_state][action]
        self.agent_state = np.random.choice(self.states, p=transition_probas)

        self.change_of_task()
        return reward, self.agent_state

# ---------------------------------------------------------------------------- #
# MAB
# ---------------------------------------------------------------------------- #


class MAB:

    def __init__(self, number_arms=2, step_change=50):

        self.number_states = 1
        self.number_actions = number_arms
        self.step_change = step_change
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)
        self.step = 0

        self.number_of_situations = number_arms
        self.all_means = np.random.uniform(0, 1, size=(self.number_of_situations,
                                                       self.number_states,
                                                       self.number_actions))

        for i in range(self.number_of_situations):
            # print(np.shape(self.all_means))
            self.all_means[i, :, i] = 1

        # print(self.all_means, self.all_stds)

        self.current_situation = 0
        self.means = self.all_means[self.current_situation]

        self.initial_state = 0
        self.agent_state = self.initial_state

    def new_episode(self):
        self.agent_state = self.initial_state

    def change_of_task(self):
        if self.step % self.step_change == 0:
            self.current_situation = np.random.randint(
                self.number_of_situations)
            self.means = self.all_means[self.current_situation]

    def make_step(self, action):
        self.step += 1

        mean = self.means[self.agent_state, action]
        reward = int(np.random.random() < mean)

        self.change_of_task()
        return reward, self.agent_state
