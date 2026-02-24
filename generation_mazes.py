import numpy as np
from const_maze import reward_distance, wall_rate, size
from const_maze import pattern, reward_value, probas_transis
from plots import plot_maze, plot_one_transition
import sys
sys.setrecursionlimit(5000)


def connexe_wall(states, size, min_rate=0):
    '''Check whether the states are next to walls or not.'''
    for i in range(size):
        for j in range(size):
            r = 0
            if i-1 > 0 and states[i-1, j] == -1:
                r += 0.25
            if i+1 < size-1 and states[i+1, j] == -1:
                r += 0.25
            if j-1 > 0 and states[i, j-1] == -1:
                r += 0.25
            if j+1 < size-1 and states[i, j+1] == -1:
                r += 0.25
            if r > min_rate and np.random.random() < r:
                states[i, j] = -1
    return states


def world_with_walls(size=size, wall_rate=wall_rate, pattern=pattern):
    '''Generate the walls of the environment following a three steps
    procedure'''
    states = np.zeros((size, size))
    # 1st phase: generate walls randomly
    
    # nb_walls = int(wall_rate * size * size)
    # walls = np.random.choice(size*size,
    #                             nb_walls,
    #                             replace=False)
    # states = np.zeros(size**2)
    # states[walls] = -1
    # states = states.reshape((size,size))

    for i in range(size):
        for j in range(size):
            if np.random.random() < wall_rate:
                states[i, j] = -1

    # 2nd phase: If walls are closeby, add a wall
    states = connexe_wall(states, size, min_rate=0)
    for i in range(2):
        states = connexe_wall(states, size, min_rate=0.25)
    states = connexe_wall(states, size, min_rate=0.999)
    # 3rd phase: Add the pattern
    for cell, value in pattern.items():
        states[cell] = value
    return states


def state_initial(states):
    ''' Choose an initial state randomly where there is no wall 
    and where there is no pattern.'''
    states_init = []
    for i in range(len(states)):
        for j in range(len(states)):
            if states[i, j] == 0 and (i, j) not in pattern.keys():
                states_init.append((i, j))
    states_init = np.array(states_init)
    indices = np.arange(states_init.shape[0])
    state_initial = states_init[np.random.choice(indices)]
    states[state_initial[0], state_initial[1]] = 1
    return states


def distance_state_initial(states):
    '''Compute the distance between the initial state and the other states.
    Uses the fact that the initial state has a value of 1.'''
    for k in range(1, 50):
        for i in range(len(states)):
            for j in range(len(states[i])):
                if states[i][j] == k:
                    if i > 0 and states[i-1][j] == 0:
                        states[i-1][j] = k + 1
                    if j > 0 and states[i][j-1] == 0:
                        states[i][j-1] = k + 1
                    if i < len(states)-1 and states[i+1][j] == 0:
                        states[i+1][j] = k + 1
                    if j < len(states[i])-1 and states[i][j+1] == 0:
                        states[i][j+1] = k + 1
    states = np.array(states, dtype='int')
    return states


def generate_distance(size=size, wall_rate=wall_rate):
    world = world_with_walls(size, wall_rate)
    world_initial = state_initial(world)
    world_distance = distance_state_initial(world_initial)
    # keep worlds where there exists cell at the reward_distance
    # from the initial state
    good_distance_reward = (world_distance == reward_distance).any()
    return world_distance, good_distance_reward


def find_possible_reward_cells(world):
    equal_to_max_distance = []
    for row in range(size):
        for col in range(size):
            if world[row, col] == reward_distance:
                equal_to_max_distance.append((row, col))
    equal_to_max_distance = np.array(equal_to_max_distance)
    return equal_to_max_distance


def get_cells_on_optimal_path(distance_initial,
                              distance_max, high_reward):
    optimal_path = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            if distance_initial[row, col] >= 1 and distance_max[row, col] >= 1:
                cond1 = distance_initial[row, col]+distance_max[row, col]
                cond2 = distance_initial[high_reward[0], high_reward[1]]+1
                optimal_path[row, col] = cond1 <= cond2
            else:
                optimal_path[row, col] = 0
    return optimal_path


def generate_world(size=size,
                   wall_rate=wall_rate,
                   reward=reward_value,
                   pattern=pattern):
    valid = False
    while not valid:
        world, valid = generate_distance()
    distance_initial = world.copy()

    # Find array of cells at the max distance
    possible_rewards = find_possible_reward_cells(world)
    indexes = np.arange(len(possible_rewards))
    high_reward = possible_rewards[np.random.choice(indexes)]

    # Add reward value and change initial state to -2
    for row in range(size):
        for col in range(size):
            if world[row, col] not in [-1, 1]:
                world[row, col] = 0
            elif world[row, col] == 1:
                world[row, col] = -2
            world[high_reward[0], high_reward[1]] = reward

    # distance from the max reward
    world_highest_reward = world.copy()
    for row in range(size):
        for col in range(size):
            if world_highest_reward[row, col] != -1:
                world_highest_reward[row, col] = 0
    world_highest_reward[high_reward[0], high_reward[1]] = 1
    distance_max = distance_state_initial(world_highest_reward)

    # cells on an optimal path
    cells_on_optimal_path = get_cells_on_optimal_path(distance_initial,
                                                      distance_max,
                                                      high_reward)

    # Create a dic of cells on the optimal path and their distance to the
    # initial state

    distance_init_optimal = dict()
    for row in range(size):
        for col in range(size):
            if cells_on_optimal_path[row, col] == 1 and (row, col) not in pattern.keys():
                distance_init_optimal[row, col] = distance_initial[row, col]

    world_valid = True

    # # Checking that the pattern is on the optimal way and far enough from the
    # # rewarded area
    # for cell, value in pattern.items():
    #     if value == 0:  # not a wall
    #         # No other path to go to the reward
    #         cond = len([k for k, v in distance_init_optimal.items()
    #                     if v == distance_initial[cell[0], cell[1]]]) == 0
    #         pattern_on_optimal_path = bool(cells_on_optimal_path[cell])
    #         if not cond or not pattern_on_optimal_path:
    #             world_valid = False

    # # Checking that there is no other path
    # world_2 = world.copy()
    # for cell, value in pattern.items():
    #     world_2[cell] = -1
    # world_2[high_reward[0], high_reward[1]] = 0
    # for row in range(size):
    #     for col in range(size):
    #         if world_2[row, col] == -2:
    #             world_2[row, col] = 1

    # distance_world_2 = distance_state_initial(world_2)
    # dist_reward_new_world = distance_world_2[high_reward[0], high_reward[1]]
    # cond_distance = dist_reward_new_world == 0

    # if not cond_distance:
    #     world_valid = False

    # # Checking that the reward is not in the pattern
    # if tuple(high_reward) in pattern.keys():
    #     world_valid = False

    if not world_valid:
        return generate_world(size=size,
                              wall_rate=wall_rate,
                              reward=reward_value,
                              pattern=pattern)

    return world


# exemple = generate_world()
# gridworld = Monde(exemple, rewards)
# pygame.display.flip()
# pygame.time.delay(1000)
# pygame.quit()


# def save_gridworld(world, path, rewards=[]):
#     gridworld = Monde(world, rewards)
#     gridworld.show()
#     gridworld.save(path)
#     gridworld.quit()


def generate_worlds(number=20):
    for i in range(number):
        world = generate_world()
        np.save('Env/Tables/World_'+str(i)+'.npy', world)
        path = "Env/Images/World_"+str(i)+".pdf"
        plot_maze(world, path)


def generate_obstructed_worlds(number=20):
    for i in range(number):
        world = np.load('Env/Tables/World_' + str(i) + '.npy')
        for row in range(len(world)):
            for col in range(len(world)):
                if (row, col) in pattern.keys():
                    world[row, col] = -1
        np.save('Env/Tables/World_'+str(i)+'_B.npy', world)
        path = "Env/Images/World_"+str(i)+"_B.pdf"
        plot_maze(world, path)
        new_world = world.copy()
        for row in range(len(world)):
            for col in range(len(world)):
                if world[row][col] not in [-1, -2]:
                    new_world[row, col] = 0
                if world[row][col] == -2:
                    new_world[row, col] = 1
        world_distance = distance_state_initial(new_world)
        path = "Env/Images/World_"+str(i)+"_distance_B.pdf"
        plot_maze(world, path, world_distance)


def generate_uncertainty(probabilities):

    # probas =[np.random.randint(1,100),
    #          np.random.randint(100,1000),
    #          np.random.randint(500,2000),
    #          np.random.randint(100,1000),
    #          np.random.randint(1,100),
    #          np.random.randint(1,1000)]

    # probas = [0.0,
    #           0.0,
    #           0.9,
    #           0.0,
    #           0.0,
    #           0.0]

    # probas = [0.1*np.random.random(),
    #           0.1*np.random.random(),
    #           0.8,
    #           0.1*np.random.random(),
    #           0.1*np.random.random(),
    #           0.1*np.random.random()]

    # probas = np.random.random(size=6)

    # probas = [np.random.random(),
    #           3*np.random.random(),
    #           5*np.random.random(),
    #           3*np.random.random(),
    #           np.random.random(),
    #           2*np.random.random()]

    probas = np.array(probabilities)/np.sum(probabilities)

    return probas


def incertitude_transition(world):
    walls = []
    size = len(world)
    for row in range(size):
        for col in range(size):
            if world[row][col] == -1:
                walls.append((row, col))
    UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
    transitions = np.zeros((size, size, 5, size, size))
    for row in range(size):
        for col in range(size):
            if world[row][col] != -1:
                for action in range(4):

                    probas = generate_uncertainty(probabilities=probas_transis)

                    row_0 = row == 0
                    col_0 = col == 0
                    row_10 = row == size-1
                    col_10 = col == size-1

                    # Defining the limit to consider depending on the action
                    limit_in_front = [row_0, row_10, col_0, col_10]
                    limit_on_left = [col_0, col_10, row_10, row_0]
                    limit_on_right = [col_10, col_0, row_0, row_10]

                    # Defining the six interesting cells depending on the action
                    if action == UP:
                        cells = [(row, col-1), (row-1, col-1), (row-1, col),
                                 (row-1, col+1), (row, col+1), (row, col)]

                    elif action == DOWN:
                        cells = [(row, col+1), (row+1, col+1), (row+1, col),
                                 (row+1, col-1), (row, col-1), (row, col)]

                    elif action == LEFT:
                        cells = [(row+1, col), (row+1, col-1), (row, col-1),
                                 (row-1, col-1), (row-1, col), (row, col)]

                    elif action == RIGHT:
                        cells = [(row-1, col), (row-1, col+1), (row, col+1),
                                 (row+1, col+1), (row+1, col), (row, col)]

                    if limit_in_front[action]:  # limit in front
                        probas[5] += probas[1] + probas[2]+probas[3]
                        probas[1] = 0
                        probas[2] = 0
                        probas[3] = 0

                    if limit_on_left[action]:  # limit on left
                        probas[5] += probas[0]+probas[1]
                        probas[0] = 0
                        probas[1] = 0

                    if limit_on_right[action]:  # limit on right
                        probas[5] += probas[3]+probas[4]
                        probas[3] = 0
                        probas[4] = 0

                    for w in range(5):  # if wall -> stay
                        if cells[w] in walls:
                            probas[5] += probas[w]
                            probas[w] = 0

                    # wall in front and left -> stay top left
                    if cells[2] in walls and cells[0] in walls:
                        probas[5] += probas[1]
                        probas[1] = 0

                    # wall in front and right -> stay top right
                    if cells[2] in walls and cells[4] in walls:
                        probas[5] += probas[3]
                        probas[3] = 0

                    arriving_probas = np.zeros((size, size))
                    sum = 0
                    for idx, cell in enumerate(cells):
                        if probas[idx] != 0:
                            round_proba = np.round(probas[idx], 3)
                            sum += round_proba
                            arriving_probas[cell] = round_proba
                    arriving_probas[row, col] += 1-sum
                    transitions[row, col, action] = arriving_probas
                transitions[row, col, STAY, row, col] = 1  # STAY ACTION

    return transitions


def generate_worlds_uncertain(number=20):
    for i in range(number):
        world = np.load('Env/Tables/World_' + str(i)+'.npy')
        transitions = incertitude_transition(world)

        np.save('Env/Transitions/Transitions_'+str(i)+'.npy', transitions)


def generate_distance_world(number=20):
    for i in range(number):
        world = np.load('Env/Tables/World_' + str(i) + '.npy')
        new_world = world.copy()
        for row in range(len(world)):
            for col in range(len(world)):
                if world[row][col] not in [-1, -2]:
                    new_world[row, col] = 0
                if world[row][col] == -2:
                    new_world[row, col] = 1
        world_distance = distance_state_initial(new_world)
        path = "Env/Images/World_"+str(i)+"_distance"+".pdf"
        plot_maze(world, path, world_distance)

# ADD stay transition !


# def show_transition(world_number, action, row, col, trans_type=''):
#     """Show a transition for a given cell and action, in a given world

#     Parameters
#     ----------
#     world_number : int
#         number of the world
#     action : int
#         between 0 and 4
#     row : int
#         row number
#     col : int
#         column number
#     trans_type : str, optional
#         '', '_U' or '_B', by default ''
#     """

#     change_world = {'_U': '', '_B': '_B', '': '', '_C': ''}
#     end_path = change_world[trans_type]
#     path_world = 'Env/Tables/World_' + str(world_number) + end_path + '.npy'
#     world = np.load(path_world)
#     path_transi = 'Env/Transitions/Transitions_' + \
#         str(world_number)+trans_type+'.npy'
#     all_transitions = np.load(path_transi, allow_pickle=True)
#     transition = all_transitions[row][col][action]
#     # Adding walls
#     walls = {(i, j): 0 for i in range(row-1, row+2)
#              for j in range(col-1, col+2)}
#     for cell in walls.keys():
#         cond_ext_wall_1 = cell[0] >= len(world) or cell[1] >= len(world)
#         cond_ext_wall_2 = cell[0] < 0 or cell[1] < 0
#         if cond_ext_wall_1 or cond_ext_wall_2 or world[cell[0], cell[1]] == -1:
#             walls[cell] = 1

#     actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
#     title = "Action " + actions[action] + \
#         ", cell ("+str(row)+","+str(col)+")"+" in world "+str(world_number)
#     # print(transition)
#     transi = Transition((row, col), title, transition, walls, action)
#     transi.show()
#     transi.save("Env/"+str(title)+'.pdf')
#     transi.quit()


def generate_pattern_uncertain(number=20):
    for i in range(number):
        transitions = np.load(
            'Env/Transitions/Transitions_' + str(i) + '.npy',
            allow_pickle=True)
        for (row, col), value in pattern.items():
            action_to_effect = [(row+1, col),
                                (row-1, col),
                                (row, col-1),
                                (row, col+1),
                                (row, col)]
            if value == 0:
                for action in range(5):
                    det = transitions[row, col,
                                      action][action_to_effect[action]]
                    transitions[row, col, action] = np.zeros((size, size))
                    if det > 0:
                        transitions[row, col, action, row, col] = 0.9
                        transitions[row, col,
                                    action][action_to_effect[action]] += 0.1
                    else:
                        transitions[row, col, action, row, col] = 1
        np.save('Env/Transitions/Transitions_'+str(i)+'_U.npy', transitions)



def generate_transitions_obstructed(number=20):
    for i in range(number):
        transitions = np.load(
            'Env/Transitions/Transitions_' + str(i) + '.npy',
            allow_pickle=True)
        size = np.shape(transitions)[0]
        for row in range(size):
            for col in range(size):
                for action in range(5):
                    for key in pattern.keys():
                        value = transitions[row, col][action][key]
                        if value > 0:
                            transitions[row, col][action][row, col] += value
                            transitions[row, col][action][key] = 0
        for (row, col) in pattern.keys():
            for action in range(5):
                transitions[row, col][action] = np.zeros((size, size))
        np.save('Env/Transitions/Transitions_' +
                str(i) + '_B.npy', transitions)


def cyclic_permutation(lst):
    if len(lst) <= 1:
        return lst
    shift = np.random.randint(1, len(lst) - 1)
    # shift = 1
    return lst[-shift:] + lst[:-shift]


def generate_transitions_cyclic(number=20):
    for i in range(number):
        transitions = np.load(
            'Env/Transitions/Transitions_' + str(i) + '.npy',
            allow_pickle=True)
        size = np.shape(transitions)[0]
        for row in range(size):
            for col in range(size):
                if (row, col) in pattern.keys():
                    action_permutation = cyclic_permutation(
                        [i for i in range(5)])
                    t_copy = transitions[row, col].copy()
                    for action in range(5):
                        new_action = action_permutation[action]
                        # print(new_action)
                        # print(row,col)
                        transitions[row, col][action] = t_copy[new_action]
        np.save('Env/Transitions/Transitions_' +
                str(i) + '_C.npy', transitions)


def generate_all_transitions_cyclic(number=20):
    for i in range(number):
        transitions = np.load(
            'Env/Transitions/Transitions_' + str(i) + '.npy',
            allow_pickle=True)
        size = np.shape(transitions)[0]
        for row in range(size):
            for col in range(size):
                action_permutation = cyclic_permutation([i for i in range(5)])
                t_copy = transitions[row, col].copy()
                for action in range(5):
                    new_action = action_permutation[action]
                    transitions[row, col][action] = t_copy[new_action]
        np.save('Env/Transitions/Transitions_' +
                str(i) + '_D.npy', transitions)


def generate_reward(number=20, rewards=reward_value):
    for i in range(number):
        world = np.load('Env/Tables/World_'+str(i)+'.npy', allow_pickle=True)
        mask = np.isin(world, rewards, invert=True)
        world[mask] = 0
        path = 'Env/Tables/Rewards_'+str(i)+'.npy'
        np.save(path, world)


def generate_optimal_policies(number=20):
    for i in range(number):
        world = np.load('Env/Tables/World_'+str(i)+'.npy', allow_pickle=True)
        # rewards = np.load('Env/Tables/Rewards_'+str(i)+'.npy')
        # transitions = np.load('Env/Transitions/Transitions_'+str(i)+'.npy')
        # transitions_U = np.load('Env/Transitions/Transitions_'+str(i)+'_U.npy')
        _, optimal_policy = value_iteration(i, cond='')
        _, optimal_policy_C = value_iteration(i, cond='_C')
        _, optimal_policy_D = value_iteration(i, cond='_D')
        path = 'Env/Optimal_policy/World_'+str(i)+'.pdf'
        path_C = 'Env/Optimal_policy/World_'+str(i)+'_C.pdf'
        path_U = 'Env/Optimal_policy/World_'+str(i)+'_U.pdf'
        path_U_blue = 'Env/Optimal_policy/World_'+str(i)+'_U_blue.pdf'
        path_blue = 'Env/Optimal_policy/World_'+str(i)+'_blue.pdf'

        # % change
        change_rate = 0.2
        nb_states = len(world.flatten())

        no_walls = world.flatten() != -1

        indexes_no_wall = np.arange(nb_states)
        indexes_no_wall = indexes_no_wall[no_walls]
        nb_states_no_wall = len(indexes_no_wall)

        num_changes = int(change_rate * nb_states_no_wall)
        random_indices = np.random.choice(indexes_no_wall,
                                            num_changes,
                                            replace=False)
        pattern_array = np.zeros(nb_states, dtype=bool)
        pattern_array[random_indices] = True
        pattern_array = pattern_array.reshape(np.shape(world))

        optimal_policy_D_20 = np.where(pattern_array, 
                                       optimal_policy_D, 
                                       optimal_policy)
        plot_maze(world, path, arrows=optimal_policy)
        plot_maze(world, path_C, arrows=optimal_policy_C)
        plot_maze(world, path_U, arrows=optimal_policy, 
                  uncertain=optimal_policy_D_20)
        plot_maze(world, path_blue, arrows=optimal_policy,
                  blue_circle=True)
        plot_maze(world, path_U_blue, arrows=optimal_policy, 
                  uncertain=optimal_policy_D_20, blue_circle=True)


def generate_all(number=10):
    generate_worlds(number)
    generate_worlds_uncertain(number)
    generate_obstructed_worlds(number)
    generate_distance_world(number)
    generate_pattern_uncertain(number)
    generate_transitions_obstructed(number)
    generate_transitions_cyclic(number)
    generate_all_transitions_cyclic(number)
    generate_reward(number)
    generate_optimal_policies(number)


def value_iteration(world_number, cond='', gamma=0.95, accuracy=1e-3):
    transitions = np.load('Env/Transitions/Transitions_' +
                          str(world_number) + cond+'.npy')
    rewards = np.load('Env/Tables/Rewards_'+str(world_number)+'.npy')
    size = np.shape(transitions)[0]
    nb_actions = np.shape(transitions)[2]

    new_shape = (size**2, nb_actions, size**2)
    transitions = transitions.reshape(new_shape)
    rewards = rewards.reshape(size**2)
    Q = np.zeros((size**2, nb_actions))
    V = np.zeros(size**2)
    converged = False
    while not converged:
        Q = np.dot(transitions, rewards+gamma*V)
        new_V = np.max(Q, axis=1)
        diff = np.abs(V - new_V)
        V = new_V
        if np.max(diff) < accuracy:
            converged = True
    policy = np.argmax(Q, axis=1)
    policy = policy.reshape((size, size))
    return V, policy


if __name__ == "__main__":
    np.random.seed(1)
    generate_all(number=10)
    plot_one_transition(world_number=4, cond='', col=3, row=3, action=0)
    plot_one_transition(world_number=5, cond='', col=3, row=3, action=0)
    plot_one_transition(world_number=5, cond='_C', col=3, row=3, action=0)
