# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.rnn as rnn


class ATRCell(rnn.RNNCell):
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def output_size(self):
        return self.num_units

    @property
    def state_size(self):
        return self.num_units

    def __call__(self, x, h_, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            w = tf.get_variable("w", [x.get_shape()[-1].value, self.num_units])
            u = tf.get_variable("u", [self.num_units, self.num_units])

            p = tf.matmul(x, w)
            q = tf.matmul(h_, u)

            i = tf.sigmoid(p + q)
            f = tf.sigmoid(p - q)

            h = i * p + f * h_

            return h, h
