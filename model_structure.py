import functools
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.callbacks import TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from collections import namedtuple
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
import math
from utils import *

tfd = tfp.distributions
epsilon = 1e-25


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def dragonnet_loss_binarycross(y_true, y_pred):
    return regression_loss(y_true, y_pred) + binary_classification_loss(y_true, y_pred)


class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]

    def get_config(self):
        config = super(EpsilonLayer, self).get_config()
        return config


class VIB(Layer):
    """变分信息瓶颈层
    """
    def __init__(self, lamb, **kwargs):
        self.lamb = lamb
        super(VIB, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        u = tf.random.normal(shape=K.shape(z_mean))
        kl_loss = - 0.5 * K.sum(K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        self.add_loss(self.lamb * kl_loss)
        u = K.in_train_phase(u, tf.zeros_like(u))
        return z_mean + K.exp(z_log_var / 2) * u

    def get_config(self):
        config = super(VIB, self).get_config()
        config.update({
            'lamb': self.lamb
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class causalvib:

    def __init__(self, targeted_regularization=True):
        self.elbo = None
        self.pred_log_prob = None
        self.loss = None
        self.train_op = None
        self.ae_model = None
        self.weight_decay = 1.0e-3
        self.cf_dim = 25
        self.input_dim = 25
        self.__targeted_regularization = targeted_regularization

    def create_model(self, input_dim, batch_size=32):
        t_l2 = 0.01
        self.input_dim = input_dim
        inputs = Input(shape=(input_dim + 2,), name='input')
        x_ori = x = inputs[:, :-2]
        y_ori = tf.reshape(inputs[:, -2], (-1, 1))
        t_ori = tf.reshape(inputs[:, -1], (-1, 1))
        # representation
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        mean = Dense(units=100, activation=None)(x)
        logVar = Dense(units=100, activation=None)(x)
        vib = VIB(0.2, name='qz_output')([mean, logVar])
        x = Dense(units=100, activation='elu', kernel_initializer='RandomNormal')(vib)
        t = Dense(units=100, activation='elu', kernel_initializer='RandomNormal')(vib)
        t_predictions = Dense(units=1, activation=tf.nn.sigmoid)(t)
        # HYPOTHESIS
        y0_predictions = tf.keras.Sequential([
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=1, activation=None, kernel_regularizer=l2(t_l2))
        ], name='y0_predictions')(x)
        y1_predictions = tf.keras.Sequential([
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=1, activation=None, kernel_regularizer=l2(t_l2))
        ], name='y1_predictions')(x)
        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name='epsilon')
        concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
        model = Model(inputs=inputs, outputs=concat_pred)
        return model

    def create_model_123(self, input_dim, batch_size=32):
        t_l2 = 0.01
        h = input_dim
        self.input_dim = input_dim
        inputs = Input(shape=(input_dim + 2,), name='input')
        x_ori = x = inputs[:, :-2]
        y_ori = tf.reshape(inputs[:, -2], (-1, 1))
        t_ori = tf.reshape(inputs[:, -1], (-1, 1))
        # representation
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        mean = Dense(units=100, activation=None)(x)
        logVar = Dense(units=100, activation=None)(x)
        vib = VIB(0.2, name='qz_output')([mean, logVar])
        x = Dense(units=100, activation='elu', kernel_initializer='RandomNormal')(vib)
        t = Dense(units=100, activation='elu', kernel_initializer='RandomNormal')(vib)
        t_predictions = Dense(units=1, activation=tf.nn.sigmoid)(t)
        # HYPOTHESIS
        y0_predictions = tf.keras.Sequential([
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=1, activation=None, kernel_regularizer=l2(t_l2))
        ], name='y0_predictions')(x)
        y1_predictions = tf.keras.Sequential([
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=1, activation=None, kernel_regularizer=l2(t_l2))
        ], name='y1_predictions')(x)
        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name='epsilon')
        concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
        model = Model(inputs=inputs, outputs=concat_pred)
        return model

    def create_model_simu(self, input_dim, batch_size=32):
        t_l2 = 0.01
        h = input_dim
        self.input_dim = input_dim
        inputs = Input(shape=(input_dim + 2,), name='input')
        x_ori = x = inputs[:, :-2]
        y_ori = tf.reshape(inputs[:, -2], (-1, 1))
        t_ori = tf.reshape(inputs[:, -1], (-1, 1))
        # representation
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        mean = Dense(units=h*3, activation='elu')(x)
        logVar = Dense(units=h*3, activation=None)(x)
        vib = VIB(0.1, name='qz_output')([mean, logVar])
        x = Dense(units=100, activation='elu', kernel_initializer='RandomNormal')(vib)
        t = Dense(units=100, activation='elu', kernel_initializer='RandomNormal')(vib)
        t_predictions = Dense(units=1, activation=tf.nn.sigmoid)(t)
        # HYPOTHESIS
        y0_predictions = tf.keras.Sequential([
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=1, activation=None, kernel_regularizer=l2(t_l2))
        ], name='y0_predictions')(x)
        y1_predictions = tf.keras.Sequential([
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2)),
            Dense(units=1, activation=None, kernel_regularizer=l2(t_l2))
        ], name='y1_predictions')(x)
        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name='epsilon')
        concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
        model = Model(inputs=inputs, outputs=concat_pred)
        return model

    def create_model1(self, input_dim, batch_size=32):
        """
        Neural net predictive model. The dragon has three heads.
        :param input_dim:
        :return:
        """
        t_l2 = 0.01
        h = input_dim
        self.input_dim = input_dim
        inputs = Input(shape=(input_dim + 2,), name='input')
        x_ori = x = inputs[:, :-2]
        y_ori = tf.reshape(inputs[:, -2], (-1, 1))
        t_ori = tf.reshape(inputs[:, -1], (-1, 1))
        # representation
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        mean = Dense(units=h * 3, activation=None)(x)
        logVar = Dense(units=h * 3, activation=None)(x)
        x = VIB(0.1, name='qz_output')([mean, logVar])
        t_rep, h_rep, y_rep = tf.split(x, 3, -1)
        y_confounder = Concatenate(1)([h_rep, y_rep])
        t_confounder = Concatenate(1)([t_rep, h_rep])
        ix = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(y_confounder)
        t_predictions = Dense(units=1, activation=sigmoid_k)(t_confounder)
        # p = PIPre()([t_predictions, t_ori])
        # p_error = tf.reduce_sum(-tf.math.log(p)) * 0.1
        # t_predictions = Dense(units=1, activation=tf.nn.sigmoid)(t_confounder)
        # HYPOTHESIS
        y0_h = x = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(ix)
        x = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(x)
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=l2(t_l2))(x)

        y1_h = x = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(ix)
        x = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(x)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=l2(t_l2))(x)
        # imb_error = mmd2_lin(y0_h, y1_h, t_predictions) * 1e-3

        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name='epsilon')
        concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
        model = Model(inputs=inputs, outputs=concat_pred)
        # model.add_loss(imb_error)
        return model

    def self_train(self, train, test, batch_size=32, epochs=10, verbose=True, input_dim=25, hyperPara=None):
        model = self.create_model(input_dim)
        # model = self.create_pre_train(train, test, batch_size, epochs, verbose, input_dim)
        callbacks = [TerminateOnNaN(),
                     tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=7, verbose=verbose,
                                                      mode="auto"),
                     ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose,
                                       mode='auto', min_delta=1e-8, cooldown=0, min_lr=0)]
        adam_lr = 1e-3
        model.save_weights('./adam_model.h5')
        while 1:
            model.compile(optimizer=Adam(lr=adam_lr), loss=self.getloss(), metrics=Metrics())
            history1 = model.fit(x=train, y=test, batch_size=batch_size, epochs=epochs, verbose=verbose,
                                 callbacks=callbacks, validation_split=0.3)
            loss_hst = history1.history["loss"][-1]
            if adam_lr < 1e-5:
                model.load_weights('./adam_model.h5')
                break
            elif np.isnan(loss_hst) or np.isinf(loss_hst):
                model.load_weights('./adam_model.h5')
                adam_lr = adam_lr * 0.5
            else:
                break
        sgd_lr = 1e-5
        sgd_callbacks = [
            TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)
        ]
        model.save_weights('./adam_model.h5')
        while 1:
            model.compile(optimizer=SGD(lr=sgd_lr, momentum=0.9, nesterov=True), loss=self.getloss(),
                          metrics=Metrics())
            history1 = model.fit(x=train, y=test, callbacks=sgd_callbacks,
                                 validation_split=0.3, epochs=epochs,
                                 batch_size=batch_size, verbose=verbose)
            loss_hst = history1.history["loss"][-1]
            if sgd_lr < 1e-12:
                model.load_weights('./adam_model.h5')
                break
            elif np.isnan(loss_hst) or np.isinf(loss_hst):
                model.load_weights('./adam_model.h5')
                sgd_lr = sgd_lr * 0.2
            else:
                break
        return model

    def getloss(self):
        if self.__targeted_regularization:
            # loss_drag = make_tarreg_loss(ratio=self.ratio, dragonnet_loss=dragonnet_loss_binarycross)

            def orthog_loss(concat_true, concat_pred):
                r = 0.5
                y_true = concat_true[:, 0]
                t = t_true = concat_true[:, 1]
                y0_pred = concat_pred[:, 0]
                y1_pred = concat_pred[:, 1]
                t_pred = concat_pred[:, 2]

                epsilons = concat_pred[:, 3]
                g = t_pred = (t_pred + 0.001) / 1.002
                # t_pred = tf.clip_by_value(t_pred, 0.01, 0.99, name='t_pred')
                # /--------------------------------------------------
                y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
                # h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
                h = t_true - t_pred
                y_pret = y_pred + epsilons * h
                # regular = tf.reduce_sum(tf.square(y_true - y_pret))  # * (t_true - g)
                regular = tf.reduce_sum(tf.square((y_true - y_pret) * (t_true - g)))
                # regular = tf.reduce_sum(tf.abs((y_true - y_pred) * (t_true - g)))
                # /--------------------------------------------------
                loss0 = (1. - t_true) * tf.square(y_true - y0_pred)  # / (1-g)
                loss1 = t_true * tf.square(y_true - y1_pred)  # / g

                w = 1. + (tf.reduce_mean(t_true) / (1 - tf.reduce_mean(t_true))) * ((1 - g) / g)
                regress_loss = tf.reduce_sum((loss0 + loss1))
                T_loss = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))
                # /--------------------------------------------------
                loss = T_loss + regress_loss + r * regular  #
                return loss

            loss_drag = orthog_loss
        else:
            loss_drag = dragonnet_loss_binarycross
        return loss_drag


class dragonnet:

    def __init__(self, targeted_regularization=True):
        self.elbo = None
        self.pred_log_prob = None
        self.loss = None
        self.train_op = None

        self.weight_decay = 1.0e-3
        self.cf_dim = 25
        self.__targeted_regularization = targeted_regularization

    def create_model(self, input_dim):
        """
        Neural net predictive model. The dragon has three heads.
        :param input_dim:
        :return:
        """
        t_l1 = 0.01
        t_l2 = 0.01
        inputs = Input(shape=(input_dim + 2,), name='input')
        x = inputs[:, :-2]
        # representation
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)

        t_predictions = Dense(units=1, activation='sigmoid')(x)

        # HYPOTHESIS
        y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(x)
        y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(x)

        # second layer
        y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(y0_hidden)
        y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(y1_hidden)

        # third
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=l2(t_l2), name='y0_predictions')(
            y0_hidden)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=l2(t_l2), name='y1_predictions')(
            y1_hidden)

        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name='epsilon')
        # logging.info(epsilons)
        concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
        model = Model(inputs=inputs, outputs=concat_pred)

        return model

    def getloss(self):
        if self.__targeted_regularization:
            # loss_drag = make_tarreg_loss(ratio=self.ratio, dragonnet_loss=dragonnet_loss_binarycross)
            loss_drag = orthog_ATE_unbounded_domain_loss  # tarreg_ATE_unbounded_domain_loss
        else:
            loss_drag = dragonnet_loss_binarycross
        return loss_drag


class tarnet:

    def __init__(self, targeted_regularization=True):
        self.elbo = None
        self.pred_log_prob = None
        self.loss = None
        self.train_op = None

        self.weight_decay = 1.0e-3
        self.cf_dim = 25
        self.__targeted_regularization = targeted_regularization

    def create_model(self, input_dim):
        """
        Neural net predictive model. The dragon has three heads.
        :param input_dim:
        :return:
        """
        t_l1 = 0.01
        t_l2 = 0.01
        inputs = Input(shape=(input_dim + 2,), name='input')
        x = inputs[:, :-2]
        # t_p = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        t_predictions = Dense(units=1, activation='sigmoid')(x)
        # representation
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)

        # HYPOTHESIS
        y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(x)
        y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(x)

        # second layer
        y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(y0_hidden)
        y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=l2(t_l2))(y1_hidden)

        # third
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=l2(t_l2), name='y0_predictions')(
            y0_hidden)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=l2(t_l2), name='y1_predictions')(
            y1_hidden)

        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name='epsilon')
        # logging.info(epsilons)
        concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
        model = Model(inputs=inputs, outputs=concat_pred)

        return model

    def getloss(self):
        if self.__targeted_regularization:
            # loss_drag = make_tarreg_loss(ratio=self.ratio, dragonnet_loss=dragonnet_loss_binarycross)
            loss_drag = tarreg_ATE_unbounded_domain_loss
        else:
            loss_drag = dragonnet_loss_binarycross
        return loss_drag


def orthog_ATE_unbounded_domain_loss(concat_true, concat_pred):
    r = 0.1
    y_true = concat_true[:, 0]
    t = t_true = concat_true[:, 1]
    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    t_pred = concat_pred[:, 2]

    epsilons = concat_pred[:, 3]
    g = t_pred = (t_pred + 0.001) / 1.002
    # t_pred = tf.clip_by_value(t_pred, 0.01, 0.99, name='t_pred')
    # /--------------------------------------------------
    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
    # h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
    h = t_true - t_pred
    y_pret = y_pred + epsilons * h
    regular = tf.reduce_sum(tf.square(y_true - y_pret) * (t_true - g))  #
    # regular = tf.reduce_sum(tf.abs((y_true - y_pred) * (t_true - g)))
    # /--------------------------------------------------
    loss0 = (1. - t_true) * tf.square(y_true - y0_pred)  # / (1-g)
    loss1 = t_true * tf.square(y_true - y1_pred)  # / g

    w = 1. + (tf.reduce_mean(t_true) / (1 - tf.reduce_mean(t_true))) * ((1 - g) / g)
    regress_loss = tf.reduce_sum((loss0 + loss1))
    T_loss = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))
    # /--------------------------------------------------
    y = y0_pred * (1 - t_true) + y1_pred * t_true
    y0_cf = y - t_true * tf.reduce_mean(y1_pred - y0_pred)
    y0_ep = y0_pred + epsilons * (t_true / t_pred - (1 - t_true) / (1 - t_pred))
    ort_regular = tf.reduce_sum(tf.square(y0_cf - y0_ep))
    # /--------------------------------------------------
    loss = T_loss + regress_loss + r * regular  #
    return loss


def fc_layer_multi(x, layers=[200, 200, 200], noa=False):
    for i, lyr in enumerate(layers):
        if lyr == 1:
            x = Dense(units=lyr, activation=None, kernel_initializer='RandomNormal', kernel_regularizer=l2(0.01))(x)
            continue
        if noa:
            x = Dense(units=lyr, activation=None, kernel_initializer='RandomNormal', kernel_regularizer=l2(0.01))(x)
            continue
        x = Dense(units=lyr, activation=tf.nn.elu, kernel_initializer='RandomNormal', kernel_regularizer=l2(0.01))(x)
    return x


def fc_layer_dup(x, layers=[200, 200, 200]):
    output = []
    for i, lyr in enumerate(layers):
        tem = Dense(units=lyr, activation=tf.nn.softplus, kernel_initializer='RandomNormal',
                    kernel_regularizer=l2(0.01))(x)
        output.append(tem)
    return output


def create_nednet_model(input_dim):
    return


def create_tarnet_model(input_dim):
    return


def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
    ratio = 1
    vanilla_loss = dragonnet_loss_binarycross(concat_true, concat_pred)

    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    x_true = concat_true[:, 2:-1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    t_pred = concat_pred[:, 2]
    x_pred = concat_pred[:, 4:-1]

    x_loss = tf.reduce_mean(tf.square(x_true - x_pred))

    epsilons = concat_pred[:, 3]
    t_pred = (t_pred + 0.01) / 1.02
    # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

    y_pert = y_pred + epsilons * h
    targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

    # final
    loss = vanilla_loss + ratio * targeted_regularization + x_loss * 0
    return loss


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.002) / 1.002
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]
    return tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)


def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


def Metrics():
    metrics = [regression_loss, binary_classification_loss, treatment_accuracy, orthog_ATE_unbounded_domain_loss,
               tarreg_ATE_unbounded_domain_loss,
               track_epsilon]
    return metrics


class CEVAE:

    def __init__(self):
        self.train = None
        self.test = None

        self.elbo = None
        self.pred_log_prob = None
        self.loss = None
        self.train_op = None

        self.xshape = 25
        self.yshape = 1
        self.weight_decay = 1.0e-3
        self.cf_dim = 25
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.contfeats = [i for i in range(self.xshape) if i not in self.binfeats]

    def make_inference_networks(self, h=200):
        xshape = self.xshape
        regularizer = l2(0.01)
        x_size, y_size = self.xshape, self.yshape
        dense = functools.partial(tf.keras.layers.Dense, units=h,
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  activation=tf.nn.elu)

        g1 = tf.keras.Sequential([
            dense(input_shape=(x_size + y_size,)),
            dense(),
            dense(units=self.cf_dim)
        ], name='g1')

        g2 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=self.cf_dim * 2),
        ], name='g2')

        g3 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=self.cf_dim * 2),
        ], name='g3')

        def encoder(x, t, y):
            shared_rep = g1(tf.concat([x, y], axis=-1))
            #  Bernoulli
            latent_code_t1 = g2(shared_rep)
            latent_code_t0 = g3(shared_rep)

            params = (1 - t) * latent_code_t0 + t * latent_code_t1
            params, var = tf.split(params, num_or_size_splits=2, axis=1)
            approx_posterior = tfd.Bernoulli(logits=params, dtype=tf.float32)

            # Normal
            t1_mu, t1_var = tf.split(
                g2(shared_rep), num_or_size_splits=2, axis=1
            )
            t0_mu, t0_var = tf.split(
                g3(shared_rep), num_or_size_splits=2, axis=1
            )
            mu = (1 - t) * t0_mu + t * t1_mu
            logvar = (1 - t) * t0_var + t * t1_var
            approx_posterior = tfd.MultivariateNormalDiag(loc=mu, scale_diag=logvar)
            return approx_posterior

        return encoder

    def make_decoder_networks(self, h=200):
        regularizer = l2(0.01)
        x_size, y_size = self.xshape, self.yshape
        dense = functools.partial(tf.keras.layers.Dense, units=h,
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  activation=tf.nn.elu)

        proxy_gen_net = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=2 * x_size),
        ], name='proxy_gen_net')

        outcome_gen_net_t1 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=2 * y_size)
        ], name='outcome_gen_net_t1')

        outcome_gen_net_t0 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=2 * y_size)
        ], name="outcome_gen_net_t0")

        treatment_gen_net = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(units=1, activation=tf.nn.sigmoid),
        ], name="treatment_gen_net")
        EPS = 1e-03

        def decoder(z):
            """

            :param z: sampled from posterior or prior
            :return: posterior predictive distributions
            """
            print_ops = []

            # p(x|z)
            def x_decoder():
                x_dist_params = proxy_gen_net(z)
                x_dist = tfd.MultivariateNormalDiag(loc=x_dist_params[..., 0:x_size],
                                                    scale_diag=x_dist_params[..., x_size:] + EPS,
                                                    name='x_post')
                return x_dist

            # p(t|z)
            def t_decoder():
                treatment_dist_params = treatment_gen_net(z)
                t_dist = tfd.Bernoulli(logits=treatment_dist_params, dtype=tf.float32, name='t_post')
                return t_dist, tf.round(treatment_dist_params)

            # p(y| t, z)
            def y_decoder(t):
                y_dist_params = (1 - t) * outcome_gen_net_t0(z) + t * outcome_gen_net_t1(z)
                y_dist = tfd.MultivariateNormalDiag(loc=y_dist_params[..., 0:y_size],
                                                    scale_diag=y_dist_params[..., y_size:] + EPS,
                                                    name='y_post')
                return y_dist

            return x_decoder, t_decoder, y_decoder, print_ops

        return decoder

    def make_prediction_networks(self, x_size, y_size, h=200):
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        regularizer = l2(0.01)
        dense = functools.partial(tf.keras.layers.Dense, units=h,
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  activation=tf.nn.elu)

        g4 = tf.keras.Sequential([
            dense(input_shape=(x_size,)),
            dense(units=1, activation=tf.nn.sigmoid),
        ], name='g4')

        g5 = tf.keras.Sequential([
            dense(input_shape=(x_size,)),
            dense(),
            dense(units=self.cf_dim)
        ], name='g5')

        g6 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=2 * y_size, activation=tf.nn.sigmoid)
        ], name='g6')

        g7 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=2 * y_size, activation=tf.nn.sigmoid)
        ], name='g7')

        def treatment_pred(x):
            treatment_dist_param = g4(x)
            treatment = tfd.Bernoulli(logits=treatment_dist_param, dtype=tf.float32, name="t_pred")
            return treatment

        def outcome_pred(t, x):
            shared_rep = g5(x)
            outcome_dist_params = t * g6(shared_rep) + (1 - t) * g7(shared_rep)
            loc, scale_diag = tf.split(outcome_dist_params, axis=-1, num_or_size_splits=[y_size, y_size])
            outcome = tfd.MultivariateNormalDiag(loc=loc,
                                                 scale_diag=scale_diag,
                                                 name='y_pred')
            return outcome

        return treatment_pred, outcome_pred

    def latent_prior(self, cf_dim):
        # return tfd.Bernoulli(probs=0.5 * tf.ones(cf_dim, dtype=tf.float32), dtype=tf.float32)
        return tfd.MultivariateNormalDiag(loc=tf.zeros(cf_dim, dtype=tf.float32), scale_diag=tf.ones(cf_dim,
                                                                                                     dtype=tf.float32))

    def create_model(self, input_dim):
        inputs = Input(shape=(input_dim + 2,), name='input')
        self.xshape = input_dim
        x, y, t = tf.split(inputs, axis=-1, num_or_size_splits=[self.xshape, 1, 1])
        prior = self.latent_prior(self.cf_dim)
        encoder = self.make_inference_networks()
        decoder = self.make_decoder_networks()

        # predict t and y just from x
        t_predictor, y_predictor = self.make_prediction_networks(y_size=1, x_size=self.xshape, h=200)

        qz = encoder(x=x, t=t, y=y)

        # within-sample posterior predictive distributions, use observed treatment
        x_dec, t_dec, y_dec, print_ops = decoder(z=qz.sample())
        x_post = x_dec()
        t_post, tp = t_dec()
        y_post = y_dec(t)

        # observational y prediction is not the same as interventional y prediction
        t_predictions, y_predictions = t_predictor(x=x), y_predictor(t=t, x=x)
        y0_predictions = y_dec(t=tf.zeros_like(t))
        y1_predictions = y_dec(t=tf.ones_like(t))
        ycf_pred = y_dec(t=t)
        y0 = tf.ones_like(t)
        y1 = tf.ones_like(t)
        for i in range(0):
            if t[i, -1] == 1:
                y0[i, -1] = ycf_pred[i, -1].mean()
                y1[i, -1] = y[i, -1]
            else:
                y0[i, -1] = y[i, -1]
                y1[i, -1] = ycf_pred[i, -1].mean()
        # t = y0_predictions.mean()
        pred_log_prob = tf.squeeze(t_predictions.log_prob(t)) + \
                        tf.squeeze(y_predictions.log_prob(y))
        elbo = tf.squeeze(x_post.prob(x)) \
               + tf.squeeze(t_post.prob(t)) \
               + tf.squeeze(y_post.prob(y)) \
               - tf.squeeze(tf.reduce_mean(tfd.kl_divergence(qz, prior), axis=-1))

        loss = - elbo - pred_log_prob
        epsilons = EpsilonLayer()(t_predictions.mean(), name='epsilon')
        self.loss = tf.reshape(loss, [-1, 1])
        self.elbo = tf.reshape(elbo, [-1, 1])
        self.pred_log_prob = tf.reshape(pred_log_prob, [-1, 1])
        concat_pred = Concatenate(1, name='output_pred')(
            [t_predictions.mean(), y0_predictions.mean(), y1_predictions.mean(), epsilons])
        model = Model(inputs=inputs, outputs=concat_pred)
        model.add_loss(tf.reduce_mean(self.loss))

        return model

    def getloss(self):
        def loss(y_true, y_pred):
            # pred, ploss = tf.split(y_pred, axis=-1, num_or_size_splits=[4, 1])
            # ploss = tf.reduce_mean(ploss)
            return 0

        return loss


class DenseVariational(Layer):
    def __init__(self,
                 units,
                 kl_weight=0.05,
                 activation='relu',
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5, **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = tf.keras.activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=tf.keras.initializers.RandomNormal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=tf.keras.initializers.RandomNormal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=tf.keras.initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=tf.keras.initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))


'''
class CEVAE:

    def __init__(self, data_feeder, xshape=25):
        self.train = None
        self.test = None

        self.elbo = None
        self.pred_log_prob = None
        self.loss = None
        self.train_op = None

        self.xshape = xshape
        self.yshape = 1
        self.weight_decay = 1.0e-3
        self.cf_dim = 20
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.contfeats = [i for i in range(self.xshape) if i not in self.binfeats]

    def make_inference_networks(self, h=200):
        regularizer = l2(0.01)
        x_size, y_size = self.xshape, self.yshape
        dense = functools.partial(tf.keras.layers.Dense, units=h,
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  activation=tf.nn.elu)

        g1 = tf.keras.Sequential([
            dense(input_shape=(x_size + y_size,)),
            dense(),
            dense(units=self.cf_dim)
        ], name='g1')

        g2 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=self.cf_dim),
        ], name='g2')

        g3 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=self.cf_dim),
        ], name='g3')

        def encoder(x, t, y):
            shared_rep = g1(tf.concat([x, y], axis=-1))
            latent_code_t1 = g2(shared_rep)
            latent_code_t0 = g3(shared_rep)

            params = (1 - t) * latent_code_t0 + t * latent_code_t1
            approx_posterior = tfd.Bernoulli(logits=params, dtype=tf.float32)
            return approx_posterior

        return encoder

    def make_decoder_networks(self, h=200):
        regularizer = l2(0.01)
        x_size, y_size = self.xshape, self.yshape
        dense = functools.partial(tf.keras.layers.Dense, units=h,
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  activation=tf.nn.elu)

        proxy_gen_net = tf.keras.Sequential([
            dense(input_shape=(self.xshape,)),
            dense(),
            dense(units=2 * x_size),
        ], name='proxy_gen_net')

        outcome_gen_net_t1 = tf.keras.Sequential([
            dense(input_shape=(self.xshape,)),
            dense(),
            dense(units=2 * y_size)
        ], name='outcome_gen_net_t1')

        outcome_gen_net_t0 = tf.keras.Sequential([
            dense(input_shape=(self.xshape,)),
            dense(),
            dense(units=2 * y_size)
        ], name="outcome_gen_net_t0")

        treatment_gen_net = dense(units=1, activation=tf.nn.sigmoid, name="treatment_gen_net")
        EPS = 1e-03

        def decoder(z):
            """

            :param z: sampled from posterior or prior
            :return: posterior predictive distributions
            """
            print_ops = []

            # p(x|z)
            def x_decoder():
                x_dist_params = proxy_gen_net(z)
                x_dist = tfd.MultivariateNormalDiag(loc=x_dist_params[..., 0:x_size],
                                                    scale_diag=x_dist_params[..., x_size:] + EPS,
                                                    name='x_post')
                return x_dist

            # p(t|z)
            def t_decoder():
                treatment_dist_params = treatment_gen_net(z)
                t_dist = tfd.Bernoulli(logits=treatment_dist_params, dtype=tf.float32, name='t_post')
                return t_dist

            # p(y| t, z)
            def y_decoder(t):
                y_dist_params = (1 - t) * outcome_gen_net_t0(z) + t * outcome_gen_net_t1(z)
                y_dist = tfd.MultivariateNormalDiag(loc=y_dist_params[..., 0:y_size],
                                                    scale_diag=y_dist_params[..., y_size:] + EPS,
                                                    name='y_post')
                return y_dist

            return x_decoder, t_decoder, y_decoder, print_ops

        return decoder

    def make_prediction_networks(self, x_size, y_size, h=200):
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        regularizer = l2(0.01)
        dense = functools.partial(tf.keras.layers.Dense, units=h,
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  activation=tf.nn.elu)

        g4 = dense(input_shape=(x_size,), units=1, activation=tf.nn.sigmoid, name='g4')

        g5 = tf.keras.Sequential([
            dense(input_shape=(x_size,)),
            dense(),
            dense(units=self.cf_dim)
        ], name='g5')

        g6 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=2 * y_size)
        ], name='g6')

        g7 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=2 * y_size)
        ], name='g7')

        def treatment_pred(x):
            treatment_dist_param = g4(x)
            treatment = tfd.Bernoulli(logits=treatment_dist_param, dtype=tf.float32, name="t_pred")
            return treatment

        def outcome_pred(t, x):
            shared_rep = g5(x)
            outcome_dist_params = t * g6(shared_rep) + (1 - t) * g7(shared_rep)
            loc, scale_diag = tf.split(outcome_dist_params, axis=-1, num_or_size_splits=[y_size, y_size])
            outcome = tfd.MultivariateNormalDiag(loc=loc,
                                                 scale_diag=scale_diag,
                                                 name='y_pred')
            return outcome

        return treatment_pred, outcome_pred

    def latent_prior(self, cf_dim):
        return tfd.Bernoulli(probs=0.5 * tf.ones(cf_dim, dtype=tf.float32), dtype=tf.float32)
    #  return tfd.Normal(loc=tf.zeros(cf_dim, dtype=tf.float32), scale=tf.ones(cf_dim, dtype=tf.float32))

    def get_model(self, input_dim):
        inputs = Input(shape=(input_dim + 2,), name='input')
        x, y, t = tf.split(inputs, axis=-1, num_or_size_splits=[self.xshape, 1, 1])
        prior = self.latent_prior(self.cf_dim)
        encoder = self.make_inference_networks()
        decoder = self.make_decoder_networks()

        # predict t and y just from x
        t_predictor, y_predictor = self.make_prediction_networks(y_size=1, x_size=self.xshape, h=200)

        qz = encoder(x=x, t=t, y=y)

        # within-sample posterior predictive distributions, use observed treatment
        x_dec, t_dec, y_dec, print_ops = decoder(x)
        x_post = x_dec()
        t_post = t_dec()
        y_post = y_dec(t)

        # observational y prediction is not the same as interventional y prediction
        t_predictions, y_predictions = t_predictor(x=x), y_predictor(t=t, x=x)
        y0_predictions = y_predictor(t=tf.zeros_like(t), x=x)
        y1_predictions = y_predictor(t=tf.ones_like(t), x=x)

        pred_log_prob = tf.squeeze(t_predictions.log_prob(t)) + \
                        tf.squeeze(y_predictions.log_prob(y))
        elbo = tf.squeeze(x_post.log_prob(x)) + \
               tf.squeeze(t_post.log_prob(y)) + \
               tf.squeeze(y_post.log_prob(y)) - \
               tf.squeeze(tfd.kl_divergence(qz, prior))

        loss = -elbo - pred_log_prob
        epsilons = EpsilonLayer()(t_predictions.mean(), name='epsilon')
        print('@@@@@@@@@@@@')
        print(loss)
        self.loss = tf.reduce_mean(loss)
        self.elbo = tf.reduce_mean(elbo)
        self.pred_log_prob = tf.reduce_mean(pred_log_prob)
        concat_pred = Concatenate(1)([t_predictions.mean(), y0_predictions.mean(), y1_predictions.mean(), epsilons])
        model = Model(inputs=inputs, outputs=concat_pred)

        return model

    def getloss(self):
        return None
'''
'''
class CEVAE(Layer):
    def __init__(self, datafeed, xshape=25, **kwargs):
        self.is_placeholder = True
        self.xshape = xshape
        self.yshape = 1
        self.weight_decay = 1.0e-3
        self.cf_dim = 20

        super(CEVAE, self).__init__(**kwargs)

    def make_inference_networks(self):
        def encoder(inputs, h=200):
            # CEVAE variational approximation (encoder) \begin

            x_in = inputs
            qt = fc_layer_multi(x_in, [h, h, self.cf_dim])
            qt_x = tfp.distributions.Bernoulli(logits=qt, dtype=tf.float32).sample()  # q(t|x)

            qy_xt_shared = fc_layer_multi(qt_x, [h, self.cf_dim])
            qy_xt0 = fc_layer_multi(qy_xt_shared, [h, h, 1])
            qy_xt1 = fc_layer_multi(qy_xt_shared, [h, h, 1])
            qy_xt = tfp.distributions.Normal(loc=(1.0 - qt_x) * qy_xt0 + qt_x * qy_xt1,
                                             scale=tf.ones_like(qy_xt_shared)).sample()  # q(y|x,t)

            xy_in = tf.concat([x_in, qy_xt], 1)
            qz_xty_shared = fc_layer_multi(xy_in, [h, h])
            qz_xt0 = fc_layer_multi(qz_xty_shared, [h, h])
            qz_xt0_mul = fc_layer_multi(qz_xt0, [h, self.cf_dim], True)
            qz_xt0_sigma = fc_layer_dup(qz_xt0, [self.cf_dim])
            qz_xt1 = fc_layer_multi(qz_xty_shared, [h, h])
            qz_xt1_mul = fc_layer_multi(qz_xt1, [h, self.cf_dim], True)
            qz_xt1_sigma = fc_layer_dup(qz_xt1, [self.cf_dim])
            qz_xty = tfp.distributions.Normal(loc=(1.0 - qt_x) * qz_xt0_mul + qt_x * qz_xt1_mul,
                                              scale=(1.0 - qt_x) * qz_xt0_sigma + qt_x * qz_xt1_sigma)  # q(z|x,t,y)
            # CEVAE variational approximation (encoder) \end
            return qz_xty

        return encoder

    def make_decoder_networks(self, h=200):
        def decoder(x_in):
            """
            :param x_in:
            :return: posterior predictive distributions
            """
            print_ops = []
            EPS = 1e-03
            # CEVAE variational approximation (decoder) \begin
            # z = tf.get_variable('normal', shape=tf.shape(x_in), initializer=tf.random_normal_initializer(stddev=1))
            z = tfp.distributions.Normal(loc=tf.zeros([tf.shape(x_in)[0], self.cf_dim]),
                                         scale=tf.ones([tf.shape(x_in)[0], self.cf_dim]))  # p(z)

            px_z_shared = fc_layer_multi(z.sample(), [h, h])
            px_z_bin = fc_layer_multi(px_z_shared, [self.xshape * 2])
            loc, scale_diag = tf.split(px_z_bin, axis=-1, num_or_size_splits=[self.xshape, self.xshape])
            x = px_z = tfd.MultivariateNormalDiag(loc=loc,
                                                  scale_diag=scale_diag + 1e-03,
                                                  name='x_post')
            # px_z_shared = fc_layer_multi(z, [200, 200, 200])
            # px_z_bin = fc_layer_multi(px_z_shared, [len(self.binfeats)])
            # x1 = bernoulli_px_z = tfp.distributions.Bernoulli(logits=px_z_bin, dtype=tf.float32)
            #
            # px_z = fc_layer_multi(px_z_shared, [200, 200])  # &**********************
            # px_z_mu = fc_layer_multi(px_z, [len(self.contfeats)])
            # px_z_sigma = fc_layer_dup(px_z, [200, 200, 200])  # &**********************
            # x2 = gaussian_px_z = tfp.distributions.Normal(loc=px_z_mu, scale=px_z_sigma)  # p(x|z)

            pt_z = fc_layer_multi(z.sample(), [h, h, 1])
            t = tfp.distributions.Bernoulli(logits=pt_z, dtype=tf.float32)  # p(t|z)
            pt_z = t.sample()
            py_t0z = fc_layer_multi(z.sample(), [h, h, 1])
            py_t1z = fc_layer_multi(z.sample(), [h, h, 1])
            y = tfp.distributions.Normal(loc=(1.0 - pt_z) * py_t0z + pt_z * py_t1z,
                                         scale=tf.ones_like(py_t0z))  # p(y|t,z)

            # CEVAE variational approximation (decoder) \end

            # p(x|z)
            def x_decoder():
                return x

            # p(t|z)
            def t_decoder():
                return t

            # p(y|t, z)
            def y_decoder():
                return y

            return x_decoder, t_decoder, y_decoder, print_ops

        return decoder

    def make_prediction_networks(self, x_size, y_size, h=200):
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        regularizer = l2(0.01)
        dense = functools.partial(tf.keras.layers.Dense, units=h,
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  activation=tf.nn.elu)

        g4 = dense(input_shape=(x_size,), units=1, activation=tf.nn.sigmoid, name='g4')

        g5 = tf.keras.Sequential([
            dense(input_shape=(x_size,)),
            dense(),
            dense(units=self.cf_dim)
        ], name='g5')

        g6 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=2 * y_size)
        ], name='g6')

        g7 = tf.keras.Sequential([
            dense(input_shape=(self.cf_dim,)),
            dense(),
            dense(units=2 * y_size)
        ], name='g7')

        def treatment_pred(x):
            treatment_dist_param = g4(x)
            treatment = tfd.Bernoulli(logits=treatment_dist_param, dtype=tf.float32, name="t_pred")
            return treatment

        def outcome_pred(t, x):
            shared_rep = g5(x)
            outcome_dist_params = t * g6(shared_rep) + (1 - t) * g7(shared_rep)
            loc, scale_diag = tf.split(outcome_dist_params, axis=-1, num_or_size_splits=[y_size, y_size])
            outcome = tfd.MultivariateNormalDiag(loc=loc,
                                                 scale_diag=scale_diag,
                                                 name='y_pred')
            return outcome

        return treatment_pred, outcome_pred

    def latent_prior(self, cf_dim):
        # return tfd.Bernoulli(probs=0.5 * tf.ones(cf_dim, dtype=tf.float32), dtype=tf.float32)
        return tfd.Normal(loc=tf.zeros(cf_dim, dtype=tf.float32), scale=tf.ones(cf_dim, dtype=tf.float32))

    def get_model(self, input_dim):
        """
        train posterior inference (encoder), decoder and prediction networks using training tuples (x, t, y)
        use observed value of t to decode or predict y (teacher forcing)
        :return:
        """
        inputs = Input(shape=(input_dim + 2,), name='input')
        x, y, t = tf.split(inputs, axis=-1, num_or_size_splits=[self.xshape, 1, 1])
        prior = self.latent_prior(self.cf_dim)
        encoder = self.make_inference_networks()
        decoder = self.make_decoder_networks()

        # predict t and y just from x
        t_predictor, y_predictor = self.make_prediction_networks(y_size=1, x_size=self.xshape, h=200)

        qz = encoder(inputs=x)

        # within-sample posterior predictive distributions, use observed treatment
        x_dec, t_dec, y_dec, print_ops = decoder(x)
        x_post = x_dec()
        t_post = t_dec()
        y_post = y_dec()

        # observational y prediction is not the same as interventional y prediction
        t_predictions, y_predictions = t_predictor(x=x), y_predictor(t=t, x=x)
        y0_predictions = y_predictor(t=tf.zeros_like(t), x=x)
        y1_predictions = y_predictor(t=tf.ones_like(t), x=x)

        pred_log_prob = tf.squeeze(t_predictions.log_prob(t)) + \
                        tf.squeeze(y_predictions.log_prob(y))
        elbo = tf.squeeze(x_post.log_prob(x)) + \
               tf.squeeze(t_post.log_prob(y)) + \
               tf.squeeze(y_post.log_prob(y)) - \
               tf.squeeze(tfd.kl_divergence(qz, prior))

        loss = -elbo - pred_log_prob
        epsilons = EpsilonLayer()(t_predictions.mean(), name='epsilon')
        print(loss)
        loss = tf.reduce_mean(loss)
        elbo = tf.reduce_mean(elbo)
        pred_log_prob = tf.reduce_mean(pred_log_prob)
        concat_pred = Concatenate(1, name='output')([t_predictions.mean(), y0_predictions.mean(), y1_predictions.mean(), epsilons])
        model = Model(inputs=inputs, outputs=loss)
        return model

    def getloss(self):
        def loss(y_true, y_pred):
            return y_pred
        return loss
'''
