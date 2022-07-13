from numpy import load
import numpy as np
from scipy.special import expit
import pandas as pd


def multivar_continue_KL_divergence(p, q):
    a = np.log(np.linalg.det(q[1]) / np.linalg.det(p[1]))
    b = np.trace(np.dot(np.linalg.inv(q[1]), p[1]))
    c = np.dot(np.dot(np.transpose(q[0] - p[0]), np.linalg.inv(q[1])), (q[0] - p[0]))
    n = p[1].shape[0]
    return 0.5 * (a - n + b + c)


#
# def format_covariates(file_path=''):
#     df = pd.read_csv(file_path, index_col='sample_id', header=0, sep=',')
#     return df
#
#
# train_rate = 0.8
# p = r"./data/Twin_Data.csv"
# with open(p, encoding='UTF-8')as f:
#     ori_data = np.loadtxt(f, delimiter=",", skiprows=1)
#     x = ori_data[:, :30]
#     no, dim = x.shape
#
#     # Define potential outcomes
#     potential_y = ori_data[:, 30:]
#     # Die within 1 year = 1, otherwise = 0
#     potential_y = np.array(potential_y < 9999, dtype=float)
#
#     ## Assign treatment
#     coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
#     prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))
#
#     prob_t = prob_temp / (2 * np.mean(prob_temp))
#     prob_t[prob_t > 1] = 1
#
#     t = np.random.binomial(1, prob_t, [no, 1])
#     t = t.reshape([no, ])
#
#     ## Define observable outcomes
#     y = np.zeros([no, 1])
#     mu0, mu1 = potential_y[:, 0], potential_y[:, 1]
#     y = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
#     y = np.reshape(np.transpose(y), [no, ])
#
#     ## Train/test division
#     idx = np.random.permutation(no)
#     train_idx = idx[:int(train_rate * no)]
#     test_idx = idx[int(train_rate * no):]
#
#     train_x = x[train_idx, :]
#     train_t = t[train_idx]
#     train_y = y[train_idx]
#     train_potential_y = potential_y[train_idx, :]
#
#     test_x = x[test_idx, :]
#     test_potential_y = potential_y[test_idx, :]
#     np.savez_compressed('twins.npz', t=t[:, None], y=y[:, None], y_cf=None, mu0=mu0[:, None], mu1=mu1[:, None],
#                         x=x[:, :, None])
#
# data = load('./data/ihdp_npci.npz')
#
# data_in = load('./data/ihdp_npci_1-100.train.npz')
# data_out = load('./data/ihdp_npci_1-100.test.npz')
# print(data_in)
# data = {'x': np.concatenate((data_in['x'], data_out['x']), axis=0),
#         'yf': np.concatenate((data_in['yf'], data_out['yf']), axis=0),
#         'ycf': np.concatenate((data_in['ycf'], data_out['ycf']), axis=0),
#         'mu0': np.concatenate((data_in['mu0'], data_out['mu0']), axis=0),
#         'mu1': np.concatenate((data_in['mu1'], data_out['mu1']), axis=0),
#         't': np.concatenate((data_in['t'], data_out['t']), axis=0)
#         }
#
# data['HAVE_TRUTH'] = not data['ycf'] is None
# data['dim'] = data['x'].shape[1]
# data['n_x'] = data['x'].shape[0]
# data['n_files'] = data['x'].shape[0]
# # for i in enumerate(data):
# data_all = [*data_in, *data_out]
# np.savez_compressed('ihdp_npci.npz', t=t, y=y, y_cf=None, mu0=mu0, mu1=mu1, x=x)

# cigma = np.random.uniform(low=-1, high=1, size=[10, 10])
# cigmaT = cigma.transpose()
# cigma0 = (cigma + cigmaT) * 0.5
# # cigma0 = np.matmul(cigma, cigmaT) * 0.5
# xl, tl, mu0l, mu1l, yl = [], [], [], [], []
# kl = []
# for k in [1, 2, 3, 4]:
#     #  control samples
#     w = np.random.uniform(low=-1, high=1, size=[10, 2])
#
#     x0 = np.random.multivariate_normal(mean=np.zeros([10]), cov=cigma0, size=5000)[:, :, None]
#
#     y0 = (np.matmul(w.transpose(), x0) + np.random.multivariate_normal(mean=(0, 0), cov=[[1, 0], [0, 1]],
#                                                                            size=5000)[:, :, None]).squeeze()
#     mu0, mu1 = y0[:, 0][:, None], y0[:, 1][:, None]
#     #  treatment samples
#     x1 = np.random.multivariate_normal(mean=np.ones([10]) * k, cov=cigma0, size=2500)[:, :, None]
#     y1 = (np.matmul(w.transpose(), x1) + np.random.multivariate_normal(mean=(0, 0), cov=[[1, 0], [0, 1]],
#                                                                            size=2500)[:, :, None]).squeeze()
#     mu0, mu1 = np.concatenate((mu0, y1[:, 0][:, None]), axis=0), np.concatenate((mu1, y1[:, 1][:, None]), axis=0)
#
#     t = np.concatenate((np.zeros([5000, 1]), np.ones([2500, 1])), axis=0)
#     y = np.concatenate((y0[:, 0][:, None], y1[:, 1][:, None]), axis=0)
#     x = np.concatenate((x0, x1), axis=0)
#
#     p = (np.zeros([10]), cigma0)
#     q = (np.ones([10]) * k, cigma0)
#     kldiv = multivar_continue_KL_divergence(p, q)
#     xl.append(x)
#     yl.append(y)
#     tl.append(t)
#     mu0l.append(mu0)
#     mu1l.append(mu1)
#     kl.append(kldiv)
#     print(kldiv)
# x = np.array(xl).squeeze().transpose([1, 2, 0])
# t = np.array(tl).squeeze().transpose()
# y = np.array(yl).squeeze().transpose()
# mu0 = np.array(mu0l).squeeze().transpose()
# mu1 = np.array(mu1l).squeeze().transpose()
# np.savez_compressed('simu_bias.npz', t=t, y=y, y_cf=None,
#                     mu0=mu0, mu1=mu1, x=x, kl=kl)
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

def norm_dist_prob(theta):
    y = norm.pdf(theta, loc=3, scale=2)
    return y

T = 5000
pi = [0 for i in range(T)]
sigma = 5
t = 0
while t < T-1:
    t = t + 1
    pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
    alpha = min(1, (norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t - 1])))

    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = pi_star[0]
    else:
        pi[t] = pi[t - 1]


plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
num_bins = 50
plt.hist(pi, num_bins, density=True, facecolor='red', alpha=0.7)
plt.show()
