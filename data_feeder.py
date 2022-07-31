import glob
import os

import numpy as np
import pandas as pd

# batch_size: the number of rows fed into the network at once.
# crop: the number of rows in the data set to be used in total.
# chunk_size: the number of lines to read from the file at once.
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class SpecDataGenerator:

    def __init__(self, file_name, output_file, load_func=None, shuffle=True, test_size=0.1):
        self.__data_base_dir = file_name
        self.__output_dir = output_file
        self.__shuffle = shuffle
        self.__total_size = 100
        self.__test_size = test_size
        self.__load_func = load_func

    @property
    def total_num_samples(self):
        return self.__total_num_samples

    @total_num_samples.setter
    def total_num_samples(self, value):
        self.__total_num_samples = value

    def load_dataset(self, folder):
        def load():
            self.__load_func(self.__data_base_dir, self.__output_dir, folder)

        return load()

    def get_ihdp_inputsize(self, data_base_dir):
        return [1, 25]

    def load_ihdp(self, data_base_dir, output_dir, folder, rep=1):
        simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
        simu_output_dir = os.path.join(output_dir, 'simu_truth')
        os.makedirs(simu_output_dir, exist_ok=True)
        for idx, simulation_file in enumerate(simulation_files):
            t, y, y_f = np.empty(shape=[0, 1]), np.empty(shape=[0, 1]), np.empty(shape=[0, 1])
            mu_0, mu_1, x = np.empty(shape=[0, 1]), np.empty(shape=[0, 1]), np.empty(shape=[0, 25])

            # simulation_output_dir = os.path.join(output_dir, str(idx))
            # os.makedirs(simulation_output_dir, exist_ok=True)
            data = np.loadtxt(simulation_file, delimiter=',')

            t = np.concatenate((t, data[:, 0][:, None]))
            y = np.concatenate((y, data[:, 1][:, None]))
            y_f = np.concatenate((y_f, data[:, 2][:, None]))
            mu_0 = np.concatenate((mu_0, data[:, 3][:, None]))
            mu_1 = np.concatenate((mu_1, data[:, 4][:, None]))
            x = np.concatenate((x, data[:, 5:]))

            y_scaler = StandardScaler().fit(y)
            y = y_scaler.transform(y)
            np.savez_compressed(os.path.join(simu_output_dir, "simulation_outputs_{}.npz".format(idx)),
                                t=t, y=y, y_cf=y_f, mu_0=mu_0, mu_1=mu_1)
            for i in range(rep):
                train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)

                x_train, x_test = np.squeeze(x[train_index]), np.squeeze(x[test_index])
                y_train, y_test = np.expand_dims(np.squeeze(y[train_index]), axis=1), np.expand_dims(
                    np.squeeze(y[test_index]), axis=1)
                t_train, t_test = np.expand_dims(np.squeeze(t[train_index]), axis=1), np.expand_dims(
                    np.squeeze(t[test_index]), axis=1)
                yield [[x_train, y_train, t_train, train_index],
                       [x_test, y_test, t_test, y_scaler, test_index],
                       output_dir, idx]
        return None

    def get_acic_inputsize(self, data_base_dir):
        covariate_csv = os.path.join(data_base_dir, 'x.csv')
        x_raw = format_covariates(covariate_csv)
        x = x_raw.values[:, :]
        return x.shape

    def load_acic(self, data_base_dir, output_dir, folder, rep=1):
        # Returns an array of the content from the CSV file.
        covariate_csv = os.path.join(data_base_dir, 'x.csv')
        x_raw = format_covariates(covariate_csv)
        simulation_dir = os.path.join(data_base_dir, folder)

        simu_output_dir = os.path.join(output_dir, 'simu_truth')
        os.makedirs(simu_output_dir, exist_ok=True)

        simulation_files = sorted(glob.glob("{}/*".format(simulation_dir)))
        for idx, simulation_file in enumerate(simulation_files):
            counter_fact = "_cf"
            file_extension = ".csv"
            if simulation_file.endswith(counter_fact + file_extension):
                continue

            ufid = os.path.basename(simulation_file)[:-4]
            t, y, sample_id, x, mu_0, mu_1 = load_treatment_outcome(x_raw, simulation_file)

            y_scaler = StandardScaler().fit(y)
            y = y_scaler.transform(y)
            for i in range(rep):
                train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)

                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                t_train, t_test = t[train_index], t[test_index]
                np.savez_compressed(os.path.join(simu_output_dir, "simulation_outputs_{}.npz".format(ufid)),
                                    t=t, y=y, sample_id=sample_id, x=x, mu_0=mu_0, mu_1=mu_1)
                yield [[x_train, y_train, t_train, train_index],
                       [x_test, y_test, t_test, y_scaler, test_index],
                       output_dir, ufid]
        return

    def get_twin_inputsize(self, data_base_dir):
        covariate_csv = os.path.join(data_base_dir, 'X.csv')
        x_raw = format_covariates(covariate_csv)
        x = x_raw.values[:, :]
        return x.shape

    def load_twin(self, data_base_dir, output_dir, folder, rep=1):
        # Returns an array of the content from the CSV file.
        covariate_csv = os.path.join(data_base_dir, 'X.csv')
        x_raw = format_covariates(covariate_csv)
        treatment_csv = os.path.join(data_base_dir, 'T.csv')
        t_raw = format_covariates(treatment_csv)
        outcome_csv = os.path.join(data_base_dir, 'Y.csv')
        y_raw = format_covariates(outcome_csv)

        dataset = x_raw.join(t_raw, how='inner')
        dataset = dataset.join(y_raw, how='inner')

        simu_output_dir = os.path.join(output_dir, 'simu_truth')
        os.makedirs(simu_output_dir, exist_ok=True)

        ufid = '0'
        y0, y1, sample_id, x, t0, t1 = dataset['mort_0'].values.reshape(-1, 1), dataset['mort_1'].values.reshape(-1, 1), \
                                       dataset.index, dataset.values[:, :-4], dataset['dbirwt_0'].values.reshape(-1, 1), \
                                       dataset['dbirwt_0'].values.reshape(-1, 1)
        t = t0
        y_scaler = StandardScaler().fit(y0)
        y = y_scaler.transform(y0)
        mu_0 = y0
        mu_1 = y1
        x_scaler = StandardScaler().fit(x)
        x = x_scaler.transform(x)
        for i in range(rep):
            train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            t_train, t_test = t[train_index], t[test_index]
            np.savez_compressed(os.path.join(simu_output_dir, "simulation_outputs_{}.npz".format(ufid)),
                                t=t, y=y, sample_id=sample_id, x=x, mu_0=mu_0, mu_1=mu_1)
            yield [[x_train, y_train, t_train, train_index],
                   [x_test, y_test, t_test, y_scaler, test_index],
                   output_dir, ufid]
        return


# use ihdp dataset defaultly
class StandardNumpyLoader:
    def __init__(self, file_name, output_file, load_func=None, shuffle=True, test_size=0.1):
        self.__data_base_dir = file_name
        self.__output_dir = output_file
        self.__shuffle = shuffle
        self.__test_size = test_size
        self.__load_func = load_func
        data_in = np.load(self.__data_base_dir)
        print(data_in)
        data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['y'], 'mu0': data_in['mu0'], 'mu1': data_in['mu1']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
        data['HAVE_TRUTH'] = not data['ycf'] is None
        data['dim'] = data['x'].shape[1]
        data['n'] = data['x'].shape[0]
        data['n_files'] = data['x'].shape[2]
        self.data = data

    def get_xsize(self):
        return [self.data['n'], self.data['dim']]

    def load_data(self, output_dir, folder=None, rep=1):

        simu_output_dir = os.path.join(output_dir, 'simu_truth')
        os.makedirs(simu_output_dir, exist_ok=True)
        data = self.data
        for ufid in range(data['n_files']):
            if ufid >= 100:
                break
            t = data['t'][:, ufid].reshape(-1, 1)
            y = data['yf'][:, ufid].reshape(-1, 1)
            if data['ycf'] is not None:
                y_cf = data['ycf'][:, ufid].reshape(-1, 1)
            else:
                y_cf = None
            mu_0 = data['mu0'][:, ufid].reshape(-1, 1)
            mu_1 = data['mu1'][:, ufid].reshape(-1, 1)
            x = data['x'][:, :, ufid].reshape(-1, data['dim'], 1)
            y_scaler = StandardScaler().fit(y)
            y = y_scaler.transform(y)
            np.savez_compressed(os.path.join(simu_output_dir, "simulation_outputs_{}.npz".format(ufid)),
                                t=t, y=y, y_cf=y_cf, mu_0=mu_0, mu_1=mu_1)

            for i in range(rep):
                train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)

                x_train, x_test = np.squeeze(x[train_index]), np.squeeze(x[test_index])
                y_train, y_test = np.expand_dims(np.squeeze(y[train_index]), axis=1), np.expand_dims(
                    np.squeeze(y[test_index]), axis=1)
                t_train, t_test = np.expand_dims(np.squeeze(t[train_index]), axis=1), np.expand_dims(
                    np.squeeze(t[test_index]), axis=1)
                mu0_train, mu0_test = np.expand_dims(np.squeeze(mu_0[train_index]), axis=1), np.expand_dims(
                    np.squeeze(mu_0[test_index]), axis=1)
                mu1_train, mu1_test = np.expand_dims(np.squeeze(mu_1[train_index]), axis=1), np.expand_dims(
                    np.squeeze(mu_1[test_index]), axis=1)
                yield [[x_train, y_train, t_train, train_index, mu0_train, mu1_train],
                       [x_test, y_test, t_test, y_scaler, test_index, mu0_test, mu1_test],
                       output_dir, ufid]
        return None


def format_covariates(file_path=''):
    df = pd.read_csv(file_path, index_col='sample_id', header=0, sep=',')
    return df


def load_treatment_outcome(covariates, file_path, standardize=True):
    output = pd.read_csv(file_path, index_col='sample_id', header=0, sep=',')
    cf_file_path = file_path.replace('.csv', '_cf.csv')
    cf_res = pd.read_csv(cf_file_path, index_col='sample_id', header=0, sep=',')
    dataset = covariates.join(output, how='inner')
    dataset = dataset.join(cf_res, how='inner')
    t = dataset['z'].values
    y = dataset['y'].values
    y0 = dataset['y0'].values
    y1 = dataset['y1'].values
    x = dataset.values[:, :-4]
    if standardize:
        normal_scalar = preprocessing.StandardScaler()
        x = normal_scalar.fit_transform(x)
    nan = np.isnan(y)
    return t.reshape(-1, 1), y.reshape(-1, 1), dataset.index, x, \
           y0.reshape(-1, 1), y1.reshape(-1, 1)
    # return t[~nan].reshape(-1, 1), y[~nan].reshape(-1, 1), dataset.index[~nan], x[~nan], \
    #        y0[~nan].reshape(-1, 1), y1[~nan].reshape(-1, 1)
