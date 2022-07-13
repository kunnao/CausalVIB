import math
import os
import numpy as np
import tensorflow as tf


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keepdims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keepdims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


def mmd2_lin(Xc, Xt, p):
    if Xc.shape[0] is None or Xt.shape[0] is None:
        k = 0
    mean_control = tf.reduce_mean(Xc, axis=0)
    mean_treated = tf.reduce_mean(Xt, axis=0)
    mmd = tf.reduce_sum(tf.square(2.0 * p * mean_treated - 2.0 * (1.0 - p) * mean_control))
    return mmd * k


def mmd2_lin_t(X, t, p):
    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]

    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)

    mean_control = tf.reduce_mean(Xc, axis=0)
    mean_treated = tf.reduce_mean(Xt, axis=0)

    mmd = tf.reduce_sum(tf.square(2.0 * p * mean_treated - 2.0 * (1.0 - p) * mean_control))
    return mmd


def mmd2_rbf(Xc, Xt, p, sig):
    """ Computes the l2-RBF MMD for X given t """

    Kcc = tf.exp(-pdist2sq(Xc, Xc) / tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc, Xt) / tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt, Xt) / tf.square(sig))

    m = tf.compat.v1.to_float(tf.shape(Xc)[0])
    n = tf.compat.v1.to_float(tf.shape(Xt)[0])

    mmd = tf.square(1.0 - p) / (m * (m - 1.0)) * (tf.reduce_sum(Kcc) - m)
    mmd = mmd + tf.square(p) / (n * (n - 1.0)) * (tf.reduce_sum(Ktt) - n)
    mmd = mmd - 2.0 * p * (1.0 - p) / (m * n) * tf.reduce_sum(Kct)
    mmd = 4.0 * mmd

    return mmd


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    ret_val = tf.math.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.math.exp(-logvar) + logvar + log2pi),
        axis=raxis
    )
    return tf.reduce_mean(ret_val)


def t_sig(x, t):
    return (tf.math.exp((2 * t - 1) * x) + 1) ** -1


def save_model(model, network_type, save_model_dir):
    # model_path = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_path = save_model_dir

    if not os.path.exists(model_path):
        open(model_path, 'a').close()

    model.save(model_path)


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(features):
    features = tf.convert_to_tensor(features)
    return features * tf.nn.sigmoid(features)


def sigmoid_k(x):
    x = tf.convert_to_tensor(x)
    return tf.nn.sigmoid(x / 6)
