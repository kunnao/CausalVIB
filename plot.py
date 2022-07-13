import matplotlib.pyplot as plt
import numpy as np
from evalution import Evaluator
from semi_parametric_estimation.ate import *
from numpy import load
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import glob
import os

mode = 1  # 0:held out ;;; 1:held in  ;;; 2:test=train
n_replication = 1


def load_truth(ufid, network_type, dataset='ihdp'):
    """
    loading ground truth data
    """
    file_path = 'D:\\study_research\\Causal Effect Inference\\dragonnet-master\\dat\\' + dataset + '\\output\\' \
                + network_type + '/simu_truth/simulation_outputs_{}.npz'.format(ufid)
    data = load(file_path)
    mu_0 = data['mu_0']
    mu_1 = data['mu_1']
    return mu_1, mu_0


def load_param(file_path, ufid, folder='censoring'):
    """
    loading ground truth data
    """
    df = pd.read_csv(file_path + folder + "_params.csv")
    res = df[df['ufid'] == str(ufid)]
    truth = np.squeeze(res.effect_size.values)

    return truth


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = np.log(2.0 * np.pi)
    ret_val = np.sum(
        -0.5 * ((sample - mean) ** 2.0 * np.exp(-logvar) + logvar + log2pi),
        axis=raxis,
    )
    return ret_val


def load_data(network_type='default', replication=0, model='baseline', train_test='test', dataset='ihdp', ufid=0,
              folder=None):
    """
    loading train test experiment results
    """
    if folder:
        file_path = './data/' + dataset + '/output' \
                    + '/{}/{}/{}/{}/'.format(network_type, model, folder, ufid)
    else:
        file_path = './data/' + dataset + '/output' \
                    + '/{}/{}/{}/'.format(network_type, model, ufid)
    data = load(file_path + '{}_{}_{}.npz'.format(network_type, replication, train_test))
    return data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), data['g'].reshape(-1, 1), \
           data['t'].reshape(-1, 1), data['y'].reshape(-1, 1), data['index'].reshape(-1, 1), data['eps'].reshape(-1, 1)


def load_guss_result(replication=0, train_test='test_guss', dataset='ihdp',
                     ufid=0, network_type='causalvib',
                     folder=None, model='targeted_regularization'):
    """
    loading train test experiment results
    """
    if folder:
        file_path = './data/' + dataset + '/output' \
                    + '/{}/{}/{}/{}/'.format(network_type, model, folder, ufid)
    else:
        file_path = './data/' + dataset + '/output' \
                    + '/{}/{}/{}/'.format(network_type, model, ufid)
    data = load(file_path + '{}_{}_{}.npz'.format('causalvib', replication, train_test))
    return data['ate_mean'].reshape(1, 1), data['ate_var'].reshape(1, 1), \
           data['ite_mean'].reshape(-1, 1), data['ite_var'].reshape(-1, 1)


def get_estimate(q_t0, q_t1, g, t, y_dragon, index, eps, truncate_level=0.0):
    """
    getting the back door adjustment & TMLE estimation
    """
    psi_n = psi_naive(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
    psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(q_t0, q_t1, g, t,
                                                                                              y_dragon,
                                                                                              truncate_level=truncate_level)
    return psi_n, psi_tmle, initial_loss, final_loss, g_loss


class std_eval:
    def __init__(self, dataset='twin', k=1, rep=1):
        self.__dataset = dataset
        self.__k = k
        self.rep = rep

    def make_table(self):
        data_in = np.load('./data/simu_bias.npz')
        kldiv = data_in['kl'].round()
        train_test = 'test'
        dict = {'dragonnet': {'baseline': 0, 'targeted_regularization': 0},
                'causalvib': {'baseline': 0, 'targeted_regularization': 0},
                'cevae': {'baseline': 0, 'targeted_regularization': 0},
                'tarnet': {'baseline': {'back_door': 0, }, 'targeted_regularization': 0},
                'nednet': {'baseline': 0, 'targeted_regularization': 0}}

        tmle_dict = copy.deepcopy(dict)
        ite_dict = copy.deepcopy(dict)
        ate_dict = copy.deepcopy(dict)
        pehe_dict = copy.deepcopy(dict)
        kl = []
        data1, data2, data3, data4 = [], [], [], []
        # data = pd.DataFrame({'kl': np.zeros([1]), 'ate': np.zeros([1]), 'model': np.zeros([1])})
        name = {'dragonnet': 'dragonnet', 'causalvib': 'CEVIB'}
        mnam = {'baseline': '', 'targeted_regularization': '-tarreg'}
        for dataset in [self.__dataset]:
            for network_type in ['dragonnet', 'causalvib']:  #
                print('-----', network_type, '-----')
                for model in ['baseline', 'targeted_regularization']:  #
                    simple_errors, tmle_errors = [], []
                    ite, ate, pehe = [], [], []
                    for idx in range(self.__k):
                        tem = []
                        for rep in range(self.rep):
                            q_t0, q_t1, g, t, y, index, eps = load_data(network_type, rep, model, train_test,
                                                                        dataset, idx)
                            q_t0_, q_t1_, g_, t_, y_, index_, eps_ = load_data(network_type, rep, model, 'train',
                                                                               dataset, idx)
                            if mode == 1:
                                q_t0 = np.concatenate([q_t0, q_t0_], axis=0)
                                q_t1 = np.concatenate([q_t1, q_t1_], axis=0)
                                g = np.concatenate([g, g_], axis=0)
                                t = np.concatenate([t, t_], axis=0)
                                y = np.concatenate([y, y_], axis=0)
                                index = np.concatenate([index, index_], axis=0)
                                eps = np.concatenate([eps, eps_], axis=0)
                            if mode == 2:
                                q_t0 = q_t0_
                                q_t1 = q_t1_
                                g = g_
                                t = t_
                                y = y_
                                index = index_
                                eps = eps_

                            a, b = load_truth(idx, network_type, dataset)
                            mu_1, mu_0 = a[index].reshape((-1, 1)), b[index].reshape((-1, 1))
                            evaluator = Evaluator(y=y, t=t, mu0=mu_1, mu1=mu_0)
                            truth = (mu_1 - mu_0).mean()

                            psi_n, psi_tmle, initial_loss, final_loss, g_loss = \
                                get_estimate(q_t0, q_t1, g, t,
                                             y, index, eps,
                                             truncate_level=0.01)

                            err = abs(truth - psi_n).mean()
                            tmle_err = abs(truth - psi_tmle).mean()
                            simple_errors.append(err)
                            tmle_errors.append(tmle_err)
                            ite1, ate1, pehe1, relpehe = evaluator.calc_stats(ypred1=q_t0, ypred0=q_t1)
                            ite.append(ite1)
                            ate.append(ate1)
                            tem.append(ate1)
                            pehe.append(pehe1)
                            # print(idx, ':', err)
                            if network_type == 'causalvib' and model == 'targeted_regularization':  # 'baseline'
                                break
                            data1.append(kldiv[idx])
                            data2.append(ate1)
                            data3.append(name[network_type] + mnam[model])
                            data4.append(pehe1)
                    dict[network_type][model] = np.nanmean(simple_errors)
                    tmle_dict[network_type][model] = np.nanmean(tmle_errors)
                    ite_dict[network_type][model] = np.nanmean(ite)
                    ate_dict[network_type][model] = np.nanmean(ate)
                    print(ate)
                    print(pehe)
                    pehe_dict[network_type][model] = np.nanmean(pehe)
                # print(kl)
        data = pd.DataFrame({'Kullback-Leibler divergence': data1, '∈ate': data2, 'model': data3})
        sns.boxplot(x="Kullback-Leibler divergence", y="∈ate", data=data, hue="model", width=0.3, linewidth=1.0,
                    palette="Set3")
        plt.savefig("simu_result_ate.pdf", dpi=333)
        plt.show()

        data = pd.DataFrame({'Kullback-Leibler divergence': data1, '∈pehe': data4, 'model': data3})
        sns.boxplot(x="Kullback-Leibler divergence", y="∈pehe", data=data, hue="model", width=0.3, linewidth=1.0,
                    palette="Set3")
        plt.savefig("simu_result_pehe.pdf", dpi=333)
        plt.show()
        return dict, tmle_dict, ite_dict, ate_dict, pehe_dict


def main():
    simu = std_eval('simu', 4, rep=50)

    print("SIMU_BIAS:")
    dict, tmle_dict, ite_dict, ate_dict, pehe_dict = simu.make_table()
    print("The back door adjustment result is below")
    print(dict)

    print("the tmle estimator result is this ")
    print(tmle_dict)

    print("the ite estimator result is this ")
    print(ite_dict)

    print("the ate estimator result is this ")
    print(ate_dict)

    print("the pehe estimator result is this ")
    print(pehe_dict)


if __name__ == '__main__':
    main()
