from evalution import Evaluator
from semi_parametric_estimation.ate import *
from numpy import load
import pandas as pd
import numpy as np
import copy
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pygal
from pygal.style import CleanStyle
mode = 0  # 0:held out ;;; 1:held in  ;;; 2:test=train
varm = 0
value, model_n = [], []
name = {'dragonnet': 'dragonnet', 'causalvib': 'VIBNet', 'tarnet': 'TARNet', 'cevae': 'CEVAE'}
mnam = {'baseline': '', 'targeted_regularization': '-tarreg'}
model2polt = {'dragonnet': 'dragonnet', 'dragonnet-tarreg': 'dragonnet-tarreg',
              'VIBNet': 0, 'VIBNet-tarreg': 'CEVIB',
              'TARNet': 0, 'TARNet-tarreg': 0,
              'CEVAE': 0, 'CEVAE-tarreg': 0}


def plotAsRadar(values, model, n=50, title='pehe'):
    # 用于正常显示中文
    # 调用Radar这个类，并设置雷达图的填充，及数据范围
    maxx = 0
    l = len(model)
    val = np.array(values)
    val = np.reshape(val, [l, -1, 2])
    values = np.mean(val, axis=-1).tolist()
    for i in range(l):
        maxt = max(values[i])
        if maxx < maxt:
            maxx = maxt
    radar_chart = pygal.Radar(fill=False, range=(0, maxx+0.1), style=CleanStyle)
    # 添加雷达图的标题
    radar_chart.title = title
    # 添加雷达图各顶点的含义
    radar_chart.x_labels = np.arange(n//2, dtype=float)

    # 绘制两条雷达图区域
    for i in range(l):
        # if model[i] == 'VIBNet-tarreg':
        #     radar_chart.add(model[i], [{'value': values[i], 'style': 'fill: red; stroke: black; stroke-dasharray: 15, 10, 5, 10, 15'}])
        # else:
        radar_chart.add(model[i], values[i])
    # radar_chart.add(model1, values1)
    # radar_chart.add(model2, values2)

    # 保存图像
    radar_chart.render_to_file('./radar_chart.svg')
    return


def load_truth(ufid, network_type, dataset='ihdp'):
    """
    loading ground truth data
    """
    file_path = './data' + dataset + '/output/' \
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
        file_path = './data' + dataset + '/output' \
                    + '/{}/{}/{}/{}/'.format(network_type, model, folder, ufid)
    else:
        file_path = './data' + dataset + '/output' \
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


class ihdp_eval:
    def __init__(self, rep=1):
        self.__dataset = 'ihdp'
        self.rep = rep

    def make_table(self):
        kn = 50
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
        peherel_dict = copy.deepcopy(dict)
        kl = []
        data1, data2, data3 = [], [], []
        for dataset in ['ihdp']:
            for network_type in ['dragonnet', 'causalvib']:  #
                print('-----', network_type, '-----')
                for model in ['targeted_regularization', 'baseline']:  #
                    simple_errors, tmle_errors = [], []
                    ite, ate, pehe, peherel = [], [], [], []
                    ate_var, pehe_var, peherel_var = np.zeros(shape=[kn, self.rep]),\
                                                     np.zeros(shape=[kn, self.rep]), np.zeros(shape=[kn, self.rep])
                    for idx in range(kn):
                        for rep in range(self.rep):
                            q_t0, q_t1, g, t, y, index, eps = load_data(network_type, rep, model, train_test,
                                                                        dataset, idx)
                            q_t0_, q_t1_, g_, t_, y_, index_, eps_ = load_data(network_type, rep, model, 'train',
                                                                               dataset, idx)
                            # if network_type == 'dragonnet' and n_replication == 1 and model != 'baseline':
                            #     ate_mean, ate_var, ite_mean, ite_var = load_guss_result(rep, 'test_guss', dataset, idx)
                            #     ite_hat = q_t1 - q_t0
                            #     kl_loss = log_normal_pdf(ite_hat, ite_mean, ite_var)
                            #     # print(kl_loss)
                            #     kl.append(kl_loss.mean())
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
                            ite1, ate1, pehe1, peherel1 = evaluator.calc_stats(ypred1=q_t0, ypred0=q_t1)
                            ite.append(ite1)
                            ate.append(ate1)
                            pehe.append(pehe1)
                            peherel.append(np.squeeze(peherel1))
                            ate_var[idx, rep] = ate1
                            pehe_var[idx, rep] = pehe1
                            peherel_var[idx, rep] = np.mean(peherel1)
                        pe = np.squeeze(np.mean(peherel, axis=0))
                        pe = np.log(pe)
                        for i, es in enumerate(pe):
                            # data1.append(1)
                            nm = model2polt[name[network_type] + mnam[model]]
                            if idx <= 9 and nm is not 0:
                                data1.append(idx+1)
                                data2.append(es)
                                data3.append(nm)

                    if varm == 0:
                        dict[network_type][model] = np.nanmean(simple_errors)
                        tmle_dict[network_type][model] = np.nanmean(tmle_errors)
                        ite_dict[network_type][model] = np.nanmean(ite)
                        ate_dict[network_type][model] = np.nanmean(ate)
                        pehe_dict[network_type][model] = np.nanmean(pehe)
                        peherel_dict[network_type][model] = np.nanmean(peherel)
                    else:
                        dict[network_type][model] = np.nanstd(simple_errors)
                        tmle_dict[network_type][model] = np.nanstd(tmle_errors)
                        ite_dict[network_type][model] = np.nanstd(ite)
                        ate_dict[network_type][model] = np.nanmean(ate_var, axis=0).std()
                        pehe_dict[network_type][model] = np.nanmean(pehe_var, axis=0).std()
                        peherel_dict[network_type][model] = np.nanmean(peherel_var, axis=0).std()
                    value.append(np.mean(peherel_var, axis=-1))
                    model_n.append(name[network_type] + mnam[model])
                print(kl)

            plotAsRadar(value, model_n, kn)
            plt.clf()
            data = pd.DataFrame({'id': data1, '∈relpehe': data2, 'model': data3})
            sns.boxplot(x="id", y="∈relpehe", data=data, hue="model", width=0.7,
                        linewidth=1.0, palette="Set3")
            plt.savefig("ihdp_result_relpehe.pdf", dpi=333)
            # plt.show()

        return dict, tmle_dict, ite_dict, ate_dict, pehe_dict, peherel_dict


class std_eval:
    def __init__(self, dataset='twin', k=1, rep=1):
        self.__dataset = dataset
        self.__k = k
        self.rep = rep

    def make_table(self):

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
        peherel_dict = copy.deepcopy(dict)
        kl = []
        data1, data2, data3 = [], [], []
        for dataset in [self.__dataset]:
            for network_type in ['dragonnet', 'causalvib', 'cevae', 'tarnet']:  #
                print('-----', network_type, '-----')
                for model in ['targeted_regularization', 'baseline']:  #
                    simple_errors, tmle_errors = [], []
                    ite, ate, pehe, peherel = [], [], [], []
                    ate_var, pehe_var, peherel_var = np.zeros(shape=[self.__k, self.rep]), \
                                                     np.zeros(shape=[self.__k, self.rep]), \
                                                     np.zeros(shape=[self.__k, self.rep])
                    for idx in range(self.__k):
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
                            ite1, ate1, pehe1, peherel1 = evaluator.calc_stats(ypred1=q_t0, ypred0=q_t1)
                            ite.append(ite1)
                            ate.append(ate1)
                            pehe.append(pehe1)
                            peherel.append(peherel1)
                            ate_var[idx, rep] = ate1
                            pehe_var[idx, rep] = pehe1
                            peherel_var[idx, rep] = np.mean(peherel1)
                        pe = np.squeeze(np.mean(peherel, axis=0))
                        pe = np.log(pe)
                        for i, es in enumerate(pe):
                            # data1.append(1)
                            nm = model2polt[name[network_type] + mnam[model]]
                            if idx <= 9 and nm is not 0:
                                data1.append(idx + 1)
                                data2.append(es)
                                data3.append(nm)

                    if varm == 0:
                        dict[network_type][model] = np.nanmean(simple_errors)
                        tmle_dict[network_type][model] = np.nanmean(tmle_errors)
                        ite_dict[network_type][model] = np.nanmean(ite)
                        ate_dict[network_type][model] = np.nanmean(ate)
                        pehe_dict[network_type][model] = np.nanmean(pehe)
                        peherel_dict[network_type][model] = np.nanmean(peherel)
                    else:
                        dict[network_type][model] = np.nanstd(simple_errors)
                        tmle_dict[network_type][model] = np.nanstd(tmle_errors)
                        ite_dict[network_type][model] = np.nanstd(ite)
                        ate_dict[network_type][model] = np.nanmean(ate_var, axis=0).std()
                        pehe_dict[network_type][model] = np.nanmean(pehe_var, axis=0).std()
                        peherel_dict[network_type][model] = np.nanmean(peherel_var, axis=0).std()
                print(kl)
            data = pd.DataFrame({'id': data1, '∈relpehe': data2, 'model': data3})
            plt.clf()
            sns.boxplot(x="id", y="∈relpehe", data=data, hue="model", width=0.7,
                        linewidth=1.0, palette="Set3")
            plt.savefig(self.__dataset+"_result_relpehe.pdf", dpi=333)
            # plt.show()
        return dict, tmle_dict, ite_dict, ate_dict, pehe_dict, peherel_dict


class acic_eval:
    def __init__(self, rep=1):
        self.__dataset = 'acic'
        self.rep = rep

    def make_table(self):
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
        peherel_dict = copy.deepcopy(dict)
        data1, data2, data3 = [], [], []
        for dataset in ['acic']:
            for network_type in ['dragonnet', 'causalvib', 'tarnet']:  # , 'cevae'
                print('-----', network_type, '-----')
                for model in ['baseline', 'targeted_regularization']:

                    for fld in ['scaling']:  # 'censoring',
                        sim_dir = './data/' + dataset + '/output' \
                                  + '/{}/{}/{}'.format(network_type, model, fld)
                        ufids = sorted(glob.glob("{}/*".format(sim_dir)))
                        ufid_simple = pd.Series(np.zeros(len(ufids)))
                        ufid_tmle = pd.Series(np.zeros(len(ufids)))
                        ufid_ite = pd.Series(np.zeros(len(ufids)))
                        ufid_ate = pd.Series(np.zeros(len(ufids)))
                        ufid_pehe = pd.Series(np.zeros(len(ufids)))
                        ufid_peherel = pd.Series(np.zeros(len(ufids)))
                        ate_var, pehe_var, peherel_var = np.zeros(shape=[len(ufids), self.rep]), \
                                                         np.zeros(shape=[len(ufids), self.rep]), \
                                                         np.zeros(shape=[len(ufids), self.rep])
                        i = 0
                        for j in range(len(ufids)):
                            ufid = os.path.basename(ufids[j])
                            all_psi_n, all_psi_tmle = [], []
                            ite, ate, pehe, peherel = [], [], [], []
                            a, b = load_truth(ufid, network_type, dataset)
                            truth_dir = './data/' + dataset + '\\'
                            truth = load_param(truth_dir, ufid, fld)
                            for rep in range(self.rep):
                                q_t0, q_t1, g, t, y, index, eps = load_data(network_type, rep, model, train_test,
                                                                            dataset, ufid, fld)
                                q_t0_, q_t1_, g_, t_, y_, index_, eps_ = load_data(network_type, rep, model,
                                                                                   'train', dataset, ufid, fld)
                                # if network_type == 'dragonnet' and n_replication == 1:
                                #     ate_mean, ate_var, ite_mean, ite_var = load_guss_result(rep, 'test_guss', dataset,
                                #                                                             ufid, folder=fld)
                                #     ite_hat = q_t1 - q_t0
                                #     kl_loss = log_normal_pdf(ite_hat, ite_mean, ite_var)
                                #     # print(kl_loss)
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
                                mu_1, mu_0 = a[index].reshape((-1, 1)), b[index].reshape((-1, 1))
                                evaluator = Evaluator(y=y, t=t, mu0=mu_1, mu1=mu_0)

                                psi_n, psi_tmle, initial_loss, final_loss, g_loss = \
                                    get_estimate(q_t0, q_t1, g, t,
                                                 y, index, eps,
                                                 truncate_level=0.01)
                                ite1, ate1, pehe1, peherel1 = evaluator.calc_stats(ypred1=q_t0, ypred0=q_t1)
                                ite.append(ite1)
                                ate.append(ate1)
                                pehe.append(pehe1)
                                peherel.append(peherel1)
                                all_psi_n.append(psi_n)
                                all_psi_tmle.append(psi_tmle)
                                ate_var[i, rep] = ate1
                                pehe_var[i, rep] = pehe1
                                peherel_var[i, rep] = np.mean(peherel1)

                            err = abs(np.nanmean(all_psi_n) - truth)
                            tmle_err = abs(np.nanmean(all_psi_tmle) - truth)
                            ufid_simple[j] = err
                            ufid_tmle[j] = tmle_err
                            ufid_ite[j] = np.nanmean(ite)
                            ufid_ate[j] = np.nanmean(ate)
                            ufid_pehe[j] = np.nanmean(pehe)
                            ufid_peherel[j] = np.nanmean(peherel)
                            i = i+1
                            pe = np.squeeze(np.mean(peherel, axis=0))
                            pe = np.log(pe)
                            for k, es in enumerate(pe):
                                # data1.append(1)
                                nm = model2polt[name[network_type] + mnam[model]]
                                if i >= 11 and i <= 20 and nm is not 0:
                                    data1.append(i-10)
                                    data2.append(es)
                                    data3.append(nm)

                            # print(ufid, ':', err)
                    nan = np.isnan(ufid_simple)
                    if varm == 0:
                        dict[network_type][model] = ufid_simple.mean()
                        tmle_dict[network_type][model] = ufid_tmle.mean()
                        ite_dict[network_type][model] = ufid_ite.mean()
                        ate_dict[network_type][model] = ufid_ate.mean()
                        pehe_dict[network_type][model] = ufid_pehe.mean()
                        peherel_dict[network_type][model] = ufid_peherel.mean()
                    else:
                        dict[network_type][model] = ufid_simple.mean()
                        tmle_dict[network_type][model] = ufid_tmle.mean()
                        ite_dict[network_type][model] = ufid_ite.mean()
                        ate_dict[network_type][model] = np.nanmean(np.array(ate_var), axis=0).std()
                        pehe_dict[network_type][model] = np.nanmean(np.array(pehe_var), axis=0).std()
                        peherel_dict[network_type][model] = np.nanmean(np.array(peherel_var), axis=0).std()
            data = pd.DataFrame({'id': data1, '∈relpehe': data2, 'model': data3})
            plt.clf()
            sns.boxplot(x="id", y="∈relpehe", data=data, hue="model", width=0.7,
                        linewidth=1.0, palette="Set3")
            plt.savefig("acic_result_relpehe.pdf", dpi=333)
            plt.show()
        return dict, tmle_dict, ite_dict, ate_dict, pehe_dict, peherel_dict


def main():
    ihdp = ihdp_eval(rep=10)
    acic = acic_eval(rep=5)
    twin = std_eval(rep=20)
    simu = std_eval('simu', 4, rep=5)
    dict, tmle_dict, ite_dict, ate_dict, pehe_dict, peherel_dict = ihdp.make_table()
    print("IHDP:")
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

    print("the relative pehe estimator result is this ")
    print(peherel_dict)
    print("--------------------------------------")
    print("ACIC:")
    dict, tmle_dict, ite_dict, ate_dict, pehe_dict, peherel_dict = acic.make_table()
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

    print("the relative pehe estimator result is this ")
    print(peherel_dict)
    print("--------------------------------------")
    print("TWIN:")
    dict, tmle_dict, ite_dict, ate_dict, pehe_dict, peherel_dict = twin.make_table()
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

    print("the relative pehe estimator result is this ")
    print(peherel_dict)
    print("--------------------------------------")
    print("SIMU_BIAS:")
    dict, tmle_dict, ite_dict, ate_dict, pehe_dict, peherel_dict = simu.make_table()
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

    print("the relative pehe estimator result is this ")
    print(peherel_dict)


if __name__ == '__main__':
    main()
