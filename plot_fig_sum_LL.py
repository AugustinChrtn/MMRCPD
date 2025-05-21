import numpy as np

first_model = [6, 8, 5, 7, 4, 4]
second_model = [2, 1, 4, 1, 8, 1]
last_observations = [4, 2, 4, 4, 1, 1, 3,
                     5]  # from most recent to oldest

first_mod_LL = []
second_mod_LL = []


def compute_distrib(list_of_occurences):
    distrib = np.array(list_of_occurences)
    return distrib / np.sum(distrib)


def compute_LL(distrib, occurence):
    return -np.log(distrib[occurence])


for index, observation in enumerate(last_observations):
    if index != 0:
        distrib_first_mod = compute_distrib(first_model)
        distrib_second_mod = compute_distrib(second_model)
        logL_to_add_first = compute_LL(distrib_first_mod, observation)
        logL_to_add_second = compute_LL(distrib_second_mod, observation)
        first_mod_LL.append(logL_to_add_first)
        second_mod_LL.append(logL_to_add_second)

        print(logL_to_add_first, logL_to_add_second, distrib_first_mod, distrib_second_mod)
    second_model[observation] += 1
    first_model[observation] -= 1

first_mod_LL = first_mod_LL+[0]
second_mod_LL = [0]+second_mod_LL

first_mod_LL = first_mod_LL[::-1]

sum_first_model = np.cumsum(first_mod_LL)
sum_second_model = np.cumsum(second_mod_LL)[::-1]

sum_LL_models = sum_first_model+sum_second_model
change_point = np.argmin(sum_LL_models)

print(change_point)
print(sum_first_model, sum_second_model)

import matplotlib.pyplot as plt

x_axis = np.arange(0, len(sum_first_model))
# plt.scatter(x_axis, sum_first_model)
# plt.scatter(x_axis, sum_second_model)
plt.rcParams.update({'font.size': 13})
plt.fill_between(x_axis, sum_second_model, sum_second_model+sum_first_model, color='#6C8EBF', alpha=0.5, label="Former model")
plt.fill_between(x_axis, sum_second_model, color='#D6B656', alpha=0.5, label ="New model")
plt.plot(x_axis, sum_LL_models, color='black', alpha=0.3)
plt.scatter(x_axis, sum_LL_models, color='black', alpha=1,s=10)
plt.vlines(x=change_point, ymin=0, ymax=sum_LL_models[change_point],color='black', alpha=0.3, linestyles='--')
plt.ylabel("Log-likelihood of the change point")
plt.xlabel("Position of the change point")
plt.ylim(0,16)
labels_used = np.arange(1,9)[::-1]
plt.xticks(ticks = x_axis, labels = labels_used)
plt.savefig("plots/change-point-likelihood.svg")
plt.close()


x_axis = np.arange(0, len(sum_first_model))
plt.fill_between(x_axis, sum_second_model, color='#D6B656', alpha=0.5, label ="New model")
plt.scatter(x_axis, sum_second_model, color='black', alpha=1,s=10)
plt.plot(x_axis, sum_second_model, color='black', alpha=0.3)
plt.ylabel("Sum of log-likelihood for the new model")
plt.xlabel("Position of the change point")
labels_used = np.arange(1,9)[::-1]
plt.xticks(ticks = x_axis, labels = labels_used)
plt.savefig("plots/change-point-likelihood-new.svg")
plt.close()

x_axis = np.arange(0, len(sum_first_model))
plt.fill_between(x_axis, sum_first_model, color='#6C8EBF', alpha=0.5, label ="Former model")
plt.scatter(x_axis, sum_first_model, color='black', alpha=1,s=10)
plt.plot(x_axis, sum_first_model, color='black', alpha=0.3)
plt.ylabel("Sum of log-likelihood for the old model")
plt.xlabel("Position of the change point")
labels_used = np.arange(1,9)[::-1]
plt.xticks(ticks = x_axis, labels = labels_used)
plt.savefig("plots/change-point-likelihood-former.svg")
plt.close()