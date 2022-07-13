import math
import copy
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TerminateOnNaN, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
from data_feeder import TrainSlidingWindowGenerator, StandardNumpyLoader
from model_structure import save_model, create_nednet_model, create_tarnet_model, dragonnet, dragonvib, CEVAE, \
    dragonnet_loss_binarycross, metrics
from tensorflow.keras.layers import Layer
adam_model = None


class Trainer:

    def __init__(self, network, data_loader, network_type, dataset, targeted_regularization, batch_size, directory, save_model_dir,
                 output_dir,
                 folder,
                 plot_result=True,
                 epoch=100, validation_frequency=1, patience=5, min_delta=1e-5, verbose=1, rep=1):
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
        self.__training_chunker = TrainSlidingWindowGenerator(file_name=self.__directory, output_file=self.__output_dir,
                                                              shuffle=True, test_size=0.3)
        self.__standard_loader = data_loader
        self.__rep = rep
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
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.__min_delta,
                                                          patience=self.__patience, verbose=self.__verbose, mode="auto")
        '''config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.compat.v1.Session(config=config)'''

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

        metrics = metrics()
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
                # if self.__network_type == 'dragonnet':
                #     dag = dragonnet(data_feeder, self.__targeted_regularization, ratio)
                #     model = dag.create_dragonnet_model(shape[1])
                #     adam_model = dag.create_dragonnet_model(shape[1])
                #     loss = dag.getloss()
                # elif self.__network_type == 'dragonvib':
                #     vib = dragonvib(data_feeder, self.__targeted_regularization, ratio)
                #     model = vib.create_dragonvib_model(shape[1])
                #     adam_model = vib.create_dragonvib_model(shape[1])
                #     loss = vib.getloss()
                # elif self.__network_type == 'cevae':
                #     cevae = CEVAE(data_feeder)
                #     model = cevae.get_model(shape[1])
                #     loss = cevae.getloss()
                # elif self.__network_type == 'nednet':
                #     model = create_nednet_model(shape[1])
                # elif self.__network_type == 'tarnet':
                #     model = create_tarnet_model(shape[1])
                model = self.network.create_model(shape[1])
                adam_model = self.network.create_model(shape[1])
                loss = self.network.getloss()
                adam_callbacks = [TerminateOnNaN(), early_stopping, checkpoint,
                                  ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=verbose,
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
                    EarlyStopping(monitor='val_loss', patience=20, min_delta=0.),
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
                        model = adam_model
                        break
                    elif np.isnan(loss_hst) or np.isinf(loss_hst):
                        model = adam_model
                        sgd_lr = sgd_lr * 0.2
                    else:
                        break

                yt_hat_test = model.predict(np.concatenate((test[0], test[1], test[2]), axis=1))
                yt_hat_train = model.predict(np.concatenate((train[0], train[1], train[2]), axis=1))
                self.__test_outputs += [_split_output(yt_hat_test, test[2], test[1], test[3], test[0], test[4])]
                self.__train_outputs += [
                    _split_output(yt_hat_train, train[2], train[1], test[3], train[0], train[3])]
                # if 'dragonvib' == self.__network_type:
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
            # for num, output in enumerate(self.vib_mean_var):
            #     np.savez_compressed(
            #         os.path.join(train_output_dir, "{}".format(self.__network_type) + "_{}_test_guss.npz".format(num)),
            #         **output)
            # save_model(model, self.__network_type, self.__save_model_dir)

    def train_model(self):
        self.train_predict()

    def plot_training_results(self, training_history):

        plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
        plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
        plt.title('Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def save_model(model, network_type, algorithm, appliance, save_model_dir):

        # model_path = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
        model_path = save_model_dir

        if not os.path.exists(model_path):
            open((model_path), 'a').close()

        model.save(model_path)


def _split_output(yt_hat, t, y, y_scaler, x, index):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] >= 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())
    var = "average propensity for treated: {} and untreated: {}".format(g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


def _calculate(ate_mean, ate_var, ite_mean, ite_var):
    return {'ate_mean': ate_mean, 'ate_var': ate_var, 'ite_mean': ite_mean, 'ite_var': ite_var}

