import math
import copy
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TerminateOnNaN, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
from data_feeder import SpecDataGenerator, StandardNumpyLoader
from model_structure import save_model, create_nednet_model, create_tarnet_model, dragonnet, causalvib, CEVAE, \
    dragonnet_loss_binarycross, Metrics
from tensorflow.keras.layers import Layer

adam_model = None


class Trainer:

    def __init__(self, network, data_loader, network_type, dataset, targeted_regularization, batch_size, directory,
                 save_model_dir,
                 output_dir,
                 folder,
                 plot_result=True, auto=0,
                 epoch=100, validation_frequency=1, patience=8, min_delta=1e-5, verbose=1, rep=1):
        self.network = network
        self.__network_type = network_type
        self.__dataset = dataset
        self.__patience = patience
        self.__min_delta = min_delta
        self.__verbose = verbose
        self.__batch_size = batch_size
        self.__epoch = epoch
        self.__loss = "mse"
        self.__metrics = ["mse"]
        self.__learning_rate = 0.001
        self.__beta_1 = 0.9
        self.__beta_2 = 0.999
        self.__save_model_dir = save_model_dir
        self.__output_dir = output_dir
        self.__targeted_regularization = targeted_regularization
        self.__directory = directory
        self.__validation_frequency = validation_frequency
        self.__val_split = 0.3
        self.__folder = folder
        self.__train_outputs = []
        self.__test_outputs = []
        self.vib_mean_var = []
        self.__training_chunker = SpecDataGenerator(file_name=self.__directory, output_file=self.__output_dir,
                                                    shuffle=True, test_size=0.3)
        self.__standard_loader = data_loader
        self.__rep = rep
        self.__auto = auto
        # StandardNumpyLoader(file_name='./data/ihdp_npci.npz', output_file=self.__output_dir,
        #                                              shuffle=True, test_size=0.3)

    def train_predict(self, model=None, ratio=1.0, knob_loss=dragonnet_loss_binarycross, val_split=0.3):
        # ----------------------tensorboard and checkpoint------------------------#
        # tf.config.experimental_run_functions_eagerly(True)

        verbose = 1
        logdir = os.path.join("logs")  # 记录回调tensorboard日志打印记录
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True,
                                                     write_images=True)
        modeldir = os.path.join(self.__save_model_dir)
        modeldir, filename = os.path.split(modeldir)
        modeldir = os.path.join(modeldir, self.__network_type + '_best.h5')  #
        print(modeldir)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=modeldir, monitor='val_loss',
                                                        save_best_only=True)  # False

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.compat.v1.enable_eager_execution()

        #  callbacks = [early_stopping, checkpoint]  # , tensorboard
        rep = self.__rep
        # Load data set
        if self.__dataset == 'acic':
            data_feeder = self.__training_chunker.load_acic(self.__directory, self.__output_dir, self.__folder, rep)
            shape = self.__training_chunker.get_acic_inputsize(self.__directory)
            print(shape)
        elif self.__dataset == 'ihdp':
            # data_feeder = self.__training_chunker.load_ihdp(self.__directory, self.__output_dir, self.__folder, rep)
            # shape = self.__training_chunker.get_ihdp_inputsize(self.__directory)
            # print(shape)
            data_feeder = self.__standard_loader.load_data(self.__output_dir, self.__folder, rep)
            shape = self.__standard_loader.get_xsize()
            print(shape)
        else:
            data_feeder = self.__standard_loader.load_data(self.__output_dir, self.__folder, rep)
            shape = self.__standard_loader.get_xsize()
            print(shape)

        # ----------------------Keras Model------------------------#

        metrics = Metrics()
        import time
        start_time = time.time()

        while True:
            self.__train_outputs = []
            self.__test_outputs = []
            test_output, train_output = [], []
            self.vib_mean_var = []

            for i in range(rep):
                dataset = next(data_feeder)
                if dataset is None:
                    break
                train, test, ufid_output_file, ufid = dataset[0], dataset[1], dataset[2], dataset[3]
                # self.__batch_size = math.ceil(train[1].shape[0] * 0.7 // 10)
                sgd = 1
                if self.__auto:
                    model = self.network.self_train(np.concatenate((train[0], train[1], train[2]), axis=1),
                                                    np.concatenate((train[1], train[2]), axis=1),
                                                    batch_size=self.__batch_size, epochs=self.__epoch,
                                                    verbose=self.__verbose, input_dim=shape[1])
                ##################
                else:
                    model = self.network.create_model(shape[1], self.__batch_size)
                    adam_model = self.network.create_model(shape[1], self.__batch_size)
                    loss = self.network.getloss()
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.__min_delta,
                                                                      patience=5, verbose=self.__verbose,
                                                                      mode="auto")
                    adam_callbacks = [TerminateOnNaN(), early_stopping, checkpoint,
                                      ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, verbose=verbose,
                                                        mode='auto',
                                                        min_delta=1e-8, cooldown=0, min_lr=0)]

                    model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=metrics)
                    training_history = model.fit(x=np.concatenate((train[0], train[1], train[2]), axis=1),
                                                 y=np.concatenate((train[1], train[2]), axis=1),
                                                 batch_size=self.__batch_size, epochs=self.__epoch,
                                                 verbose=self.__verbose, callbacks=adam_callbacks,
                                                 validation_split=self.__val_split)

                    model.save_weights('./adam_model.h5')

                    adam_model.load_weights('./adam_model.h5')

                    sgd_callbacks = [
                        TerminateOnNaN(),
                        EarlyStopping(monitor='val_loss', patience=10, min_delta=0.),
                        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, verbose=verbose, mode='auto',
                                          min_delta=0., cooldown=0, min_lr=0)
                    ]
                    sgd_lr = 1e-5
                    while sgd:
                        model.compile(optimizer=SGD(lr=sgd_lr, momentum=0.9, nesterov=True), loss=loss,
                                      metrics=metrics)
                        history1 = model.fit(x=np.concatenate((train[0], train[1], train[2]), axis=1),
                                             y=np.concatenate([train[1], train[2]], 1), callbacks=sgd_callbacks,
                                             validation_split=self.__val_split, epochs=self.__epoch,
                                             batch_size=self.__batch_size, verbose=self.__verbose)
                        loss_hst = history1.history["loss"][-1]
                        if sgd_lr < 1e-10:
                            model.load_weights('./adam_model.h5')
                            break
                        elif np.isnan(loss_hst) or np.isinf(loss_hst):
                            model.load_weights('./adam_model.h5')
                            sgd_lr = sgd_lr * 0.2
                        else:
                            break

                yt_hat_test = model.predict(np.concatenate((test[0], test[1], test[2]), axis=1))
                yt_hat_train = model.predict(np.concatenate((train[0], train[1], train[2]), axis=1))
                dic_test = _split_output(yt_hat_test, test[2], test[1], test[3], test[0], test[4], test[5], test[6])
                dic_train = _split_output(yt_hat_train, train[2], train[1], test[3], train[0], train[3], train[4], train[5])
                self.__test_outputs += [dic_test]
                self.__train_outputs += [dic_train]
                self.plot_test_data(dic_train, dic_test, training_history)
                # if 'causalvib' == self.__network_type:
                #     test_t = 10
                #     for i in range(test_t):
                #         yt_test = model.predict(np.concatenate((test[0], test[1], test[2]), axis=1))
                #         yt_train = model.predict(np.concatenate((train[0], train[1], train[2]), axis=1))
                #         test_output_rep = _split_output(yt_test, test[2], test[1], test[3], test[0], test[4])
                #         train_output_rep = _split_output(yt_train, train[2], train[1], test[3], train[0], train[3])
                #         ypred0 = test_output_rep['q_t0'].reshape(-1, 1)
                #         ypred1 = test_output_rep['q_t1'].reshape(-1, 1)
                #         # idx1, idx0 = np.where(test[2] == 1), np.where(test[2] == 0)
                #         # ite1, ite0 = test[1][idx1] - ypred0[idx1], ypred1[idx0] - test[1][idx0]
                #         # pred_ite = np.zeros_like(ypred0)
                #         # pred_ite[idx1] = ite1
                #         # pred_ite[idx0] = ite0
                #         pred_ite = ypred1 - ypred0
                #         test_output += [pred_ite]
                #     tst = np.array(test_output).reshape(test_t, -1)
                #     ite_mean = np.nanmean(np.array(tst), axis=0)
                #     ite_var = np.mean(np.square(tst), axis=0) - np.square(
                #         np.mean(tst, axis=0))
                #     ate_mean = np.nanmean(tst)
                #     ate_var = np.mean(np.square(tst)) - np.square(
                #         np.mean(tst))
                #     self.vib_mean_var += [_calculate(ate_mean, ate_var, ite_mean, ite_var)]
                K.clear_session()
            if self.__dataset == 'acic':
                if self.__targeted_regularization:
                    train_output_dir = os.path.join(ufid_output_file,
                                                    "targeted_regularization/{}/{}".format(self.__folder, ufid))
                else:
                    train_output_dir = os.path.join(ufid_output_file, "baseline/{}/{}".format(self.__folder, ufid))
            elif self.__dataset == 'ihdp' or self.__dataset == 'twin' or self.__dataset == 'simu':
                if self.__targeted_regularization:
                    train_output_dir = os.path.join(ufid_output_file, "targeted_regularization/{}".format(ufid))
                else:
                    train_output_dir = os.path.join(ufid_output_file, "baseline/{}".format(ufid))
            os.makedirs(train_output_dir, exist_ok=True)
            for num, output in enumerate(self.__test_outputs):
                np.savez_compressed(
                    os.path.join(train_output_dir, "{}".format(self.__network_type) + "_{}_test.npz".format(num)),
                    **output)

            for num, output in enumerate(self.__train_outputs):
                np.savez_compressed(
                    os.path.join(train_output_dir, "{}".format(self.__network_type) + "_{}_train.npz".format(num)),
                    **output)
            for num, output in enumerate(self.vib_mean_var):
                np.savez_compressed(
                    os.path.join(train_output_dir, "{}".format(self.__network_type) + "_{}_test_guss.npz".format(num)),
                    **output)
            save_model(model, self.__network_type, self.__save_model_dir)

    def plot_test_data(self, dic_train, dic_test, training_history):
        ntrain, ntest = dic_train['q_t0'].shape[0], dic_test['q_t0'].shape[0]
        q_t0 = np.concatenate((dic_train['q_t0'], dic_test['q_t0']), axis=0).reshape(-1)
        q_t1 = np.concatenate((dic_train['q_t1'], dic_test['q_t1']), axis=0).reshape(-1)
        t = np.concatenate((dic_train['t'], dic_test['t']), axis=0).reshape(-1)
        g = np.concatenate((dic_train['g'], dic_test['g']), axis=0).reshape(-1)
        x = np.concatenate((dic_train['x'], dic_test['x']), axis=0)
        y = np.concatenate((dic_train['y'], dic_test['y']), axis=0).reshape(-1)
        mu0 = np.concatenate((dic_train['mu0'], dic_test['mu0']), axis=0).reshape(-1)
        mu1 = np.concatenate((dic_train['mu1'], dic_test['mu1']), axis=0).reshape(-1)
        # evaluator = Evaluator(y=y, t=t, mu0=mu1, mu1=mu0)

        # calculate ite and ate
        ite_true = mu1 - mu0
        ite = np.zeros_like(ite_true)
        idx1, idx0 = np.where(t == 1), np.where(t == 0)
        ite1, ite0 = y[idx1] - q_t0[idx1], q_t1[idx0] - y[idx0]
        ite[idx1] = ite1
        ite[idx0] = ite0

        ate = np.mean(q_t1 - q_t0)
        ate_true = np.mean(ite_true)

        nrows, rrows = 3, 2
        ms = 18
        # plot
        sp1 = Trainer.new_subplot([nrows, rrows, 1], xl='y0', yl='y0_predict', title='scatter for y0')
        sp1.scatter(mu0, q_t0, c='r', edgecolors='darkslategray')

        sp2 = Trainer.new_subplot([nrows, rrows, 2], xl='y1', yl='y1_predict', title='scatter for y1')
        sp2.scatter(mu1, q_t1, c='r', edgecolors='darkslategray')

        train_loss = Trainer.new_subplot([nrows, rrows, (3, 4)], xl='epoch', yl='loss', title='MSE (Training and Validation Loss)')
        train_loss.plot(np.array(training_history.history["loss"]), label='loss', marker="x", color='darkslategray')
        train_loss.plot(np.array(training_history.history["val_loss"]), label='val_loss', marker="o", color='forestgreen')
        plt.legend()

        effect = Trainer.new_subplot([nrows, rrows, (5, 6)], xl='patient', yl='Treatment effect', title='ATE and ITE')
        effect.scatter(np.arange(0, ite.shape[0]), ite, label='ITE_predict', marker="x", color='firebrick')
        effect.scatter(np.arange(0, ite.shape[0]), ite_true, label='ITE_true', marker="+", color='darkslategray')
        effect.plot(np.arange(0, ite.shape[0]), np.ones_like(ite) * ate, label='ATE_predict', marker=",", color='forestgreen')
        effect.plot(np.arange(0, ite.shape[0]), np.ones_like(ite) * ate_true, label='ATE_true', marker=",", color='r')

        plt.legend()
        plt.grid()  # 添加网格
        plt.show()
        return

    def train_model(self, auto=1):
        if auto:
            self.train_predict()
        else:
            self.train_predict_self()

    def train_predict_self(self):
        return

    def plot_training_results(self, training_history):
        plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
        plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
        plt.title('Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def save_model(self, model, network_type, algorithm, appliance, save_model_dir):

        # model_path = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
        model_path = save_model_dir

        if not os.path.exists(model_path):
            open((model_path), 'a').close()

        model.save(model_path)

    @staticmethod
    def new_subplot(pos=[1, 1, 1], xl='', yl='', title='', x_margin=0, y_margin=0):
        ax = plt.subplot(pos[0], pos[1], pos[2])
        ax.margins(x=x_margin, y=y_margin)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        return ax


def _split_output(yt_hat, t, y, y_scaler, x, index, mu0, mu1):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(1, -1).copy()).reshape(-1)
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(1, -1).copy()).reshape(-1)
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] >= 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())
    var = "average propensity for treated: {} and untreated: {}".format(g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'mu0': mu0, 'mu1': mu1,
            'index': index, 'eps': eps}


def _calculate(ate_mean, ate_var, ite_mean, ite_var):
    return {'ate_mean': ate_mean, 'ate_var': ate_var, 'ite_mean': ite_mean, 'ite_var': ite_var}

