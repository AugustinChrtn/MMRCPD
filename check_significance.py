import scipy.stats as st
import numpy as np


def compute_z_score(p1, p2, n1, n2):
    p = (n1*p1+n2*p2)/(n1+n2)
    diff = abs(p1-p2)
    product = p*(1-p)*(1/n1+1/n2)
    return diff/np.sqrt(product), p1 > p2


def get_p_value(z_score):
    return (1-st.norm.cdf(z_score))*2


def get_stats(all_means,
              key_agent1='SoftmaxFiniteHorizon',
              key_agent2='SoftmaxMultiModel',
              n=4000):
    means_1 = all_means[key_agent1]
    means_2 = all_means[key_agent2]
    list_significance = []
    bigger = []
    for (mean1, mean2) in zip(means_1, means_2):
        z_score, p1_bigger_p2 = compute_z_score(p1=mean1,
                                                p2=mean2,
                                                n1=n,
                                                n2=n)
        p_value = get_p_value(z_score)
        list_significance.append(p_value)
        bigger.append(p1_bigger_p2)
    return list_significance, bigger
