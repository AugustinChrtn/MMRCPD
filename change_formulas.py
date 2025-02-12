import numpy as np
import matplotlib.pyplot as plt


def relative_change(x, y):
    if x > 0:
        return abs(x-y)/abs(x)
    if x == 0:
        return False


def arithmetic_mean_change(x, y):
    return 2*abs(x-y)/(abs(x)+abs(y))


def geo_mean_change(x, y):
    print(2*abs(x-y)/(abs(x)+abs(y)))


liste = [0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 1, 0, 1, 2, 2, 1, 0, 1, 0, 1, 0, 1]


def KL_div(p, q, epsilon=1e-3):
    p += epsilon
    q += epsilon
    kl = np.sum(p * np.log(p / q))
    return kl


def log_likelihood(p, distrib_counts, epsilon=1e-5):
    p += epsilon
    p /= np.sum(p)
    log_p = -np.log(p)
    log_p[log_p < 0] = 0
    log_ll = np.sum(log_p*distrib_counts)
    # print(log_ll)
    return log_ll/np.sum(distrib_counts)


def last_to_count(size, arr):
    unique_indices, counts = np.unique(arr, return_counts=True)
    unique_indices = unique_indices.astype(int)
    transformed_arr = np.zeros(size, dtype=int)
    transformed_arr[unique_indices] = counts
    return transformed_arr


def from_count_to_all_distrib(count, last, size):
    all_distrib = []
    for index, item in enumerate(last):
        new_count = count
        new_count[item] -= 1
        interesting = last[:index+1]
        # interesting = last
        count_last = last_to_count(size, interesting)
        if np.sum(count_last) != 0:
            all_distrib.append((np.array(new_count)/np.sum(new_count),
                                count_last/np.sum(count_last)))
    return all_distrib


def from_count2_to_all_distrib(count, last, size):
    all_distrib = []
    last = last[::-1]
    for index, item in enumerate(last):
        new_count = count
        new_count[item] += 1
        interesting = last[:index+1]
        # interesting = last
        count_last = last_to_count(size, interesting)
        # print(count_last)
        if np.sum(count_last) != 0:
            all_distrib.append(
                (np.array(new_count)/np.sum(new_count),
                 count_last/np.sum(count_last)))
    return all_distrib


def find_kl(count, last, size, order):
    if order:
        all_distrib = from_count_to_all_distrib(count, last, size)
    else:
        all_distrib = from_count2_to_all_distrib(count, last, size)
    all_kl = []
    for distrib in all_distrib:
        q = distrib[0]
        p = distrib[1]
        kl = KL_div(p, q, epsilon=1e-5)
        all_kl.append(kl)
    return all_kl
    # plt.plot(all_kl)
    # plt.savefig('plots/formula.png')

def find_sum_kl(count1, count2, last, size):
    all_kl_count1 = find_kl(count1, last, size, order=True)
    all_kl_count2 = find_kl(count2, last, size, order=False)
    # print(all_kl_count1, all_kl_count2)


def find_distribution(ranging, liste):
    probas = np.zeros(ranging)
    for value in liste:
        probas[value] += 1/len(liste)
    return probas


probas_1 = [0.5, 0.49, 0.01]


def get_all_probas(ranging, liste):
    all_probas = []
    # liste = liste[::-1]
    length = len(liste)
    for a in range(length):
        new_list = liste[:a+1]
        new_list_probas = find_distribution(ranging, new_list)
        all_probas.append(new_list_probas)
    return np.array(all_probas)


# probas = get_all_probas(len(probas_1), liste)

# all_res = []
# for i in range(len(probas)):
#     a = KL_div(probas[:i], np.array(probas_1))
#     print(a)
#     all_res.append(a)

# plt.plot(all_res)
# plt.savefig('plots/formula.png')

#
last = [5,3,1,1,4,4,2,4]

# print(len(last))
count = [6,8,5,7,4,4]
count2 = [1,2,4,1,8,1]
size = 6

# count = [252,  0,  0,  0,  0,   0, 2239]
# count2 = [53,  0, 0, 0, 0, 0, 10]
# last = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0]
# size = 7


all_kl = find_kl(count, last, size, True)
all_kl_2 = find_kl(count2, last, size, False)
# print(all_kl)
# print(all_kl_2)
print("min distrib 1:", np.argmin(all_kl))
print("min distrib 2:", np.argmin(all_kl_2))
plt.plot(all_kl)
plt.plot(all_kl_2)
plt.savefig('plots/formula.png')
plt.close()

sum = np.array(all_kl)+np.array(all_kl_2)
plt.plot(sum)
# print("")
# print(sum)
print("min total:", np.argmin(sum))
plt.savefig('plots/sum.png')

# old_count [135   0   0   0 679   0]
# new_count [0 0 0 0 0 0]
# last [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4]
# kl old [1.80289109 1.809151   1.81546592 1.82183671 1.82826425 1.83474944
#  1.84129319 1.84789644 1.85456015 1.86128531 1.86807293 1.87492403
#  1.88183968 1.88882096 1.52774987 1.55259667 1.57565872 1.59721854
#  1.61750379 1.41427677]
# kl new [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# sum [1.80289109 1.809151   1.81546592 1.82183671 1.82826425 1.83474944
#  1.84129319 1.84789644 1.85456015 1.86128531 1.86807293 1.87492403
#  1.88183968 1.88882096 1.52774987 1.55259667 1.57565872 1.59721854
#  1.61750379 1.41427677]

# min_index 19
# to_reassign [4]


from scipy.stats import wasserstein_distance
u = np.array([6,8,5,7,4,4])
# u = np.array([0,2,0,1,0,1])
obs_u = np.sum(u)   
v = np.array([1,2,4,1,8,1])
v = np.array([0,2,0,1,0,1])
obs_v = np.sum(v)
sum_uv = (u+v)/np.sum(u+v)
factor_u = obs_u/(obs_v+obs_u)
factor_v = obs_v/(obs_v+obs_u)
print(sum_uv)
u = u/obs_u
v = v/obs_v
uv = (u+v)/2
was = wasserstein_distance(u,v)
print("was:", was)
print("KL u-v:", (KL_div(u,v)))
print("KL v-u:", (KL_div(v,u)))
print("sim_KL:", (KL_div(u,v)+KL_div(v,u)) /2)
print("Jensen:", 1/2*KL_div(u, uv)+1/2*KL_div(v,uv))
print("Our metric:",1/2*KL_div(u, sum_uv)+1/2*KL_div(v,sum_uv))
print("Weighted Jensen:",factor_u*KL_div(u, sum_uv)+factor_v*KL_div(v,sum_uv))
print("min_KL", min(KL_div(u,v),KL_div(v,u)))