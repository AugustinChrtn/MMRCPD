size = 7
#wall_rate = 0.1
wall_rate = 0.15
reward_distance = 10
reward_value = 1

max_pattern = size//2+1
min_pattern = size//2-1
pattern = {(i, j): 0 for i in range(min_pattern, max_pattern+1) 
           for j in range(min_pattern, max_pattern+1)}


probas_transis = [0.05,
              0.05,
              0.75,
              0.05,
              0.05,
              0.05]

pattern={}
# for i in [min_pattern, max_pattern]:
#     for j in [min_pattern, max_pattern]:
#         pattern[i, j] = -1