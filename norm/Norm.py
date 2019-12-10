# -*- encoding: utf-8 -*-
"""
@File    : Norm.py
@Time    : 2019/12/10 15:25
@Author  : zwt
@git   : https://github.com/taki0112/Group_Normalization-Tensorflow
@Software: PyCharm
"""
import tensorflow as tf
import tensorflow.contrib as tf_contrib


def batch_norm(x, is_training=False, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=False, data_format='NCHW', zero_debias_moving_mean=True,
                                        is_training=is_training, scope=scope)

    # return tf.layers.batch_normalization(x, momentum=0.99, epsilon=1e-05, center=True, scale=True, renorm=True,
    # training=is_training, name=scope)


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm'):
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def group_norm(x, G=32, eps=1e-5, scope='group_norm'):
    with tf.variable_scope(scope):
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


if __name__ == '__main__':
    x = tf.random_uniform([2, 2, 2], minval=1.0, maxval=5.0, dtype=tf.float32)
    y = layer_norm(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(y))
        print(sess.run(x))