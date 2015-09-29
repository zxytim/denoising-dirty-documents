#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: model_tools.py
# $Date: Wed Sep 30 04:23:29 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>


from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, \
    Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
    ZeroPadding2D
from keras.optimizers import SGD, Adam

import theano
import theano.tensor as T


def get_inception(input_channel,
                  nr_c0_conv_1x1,
                  nr_c1_conv_1x1,
                  nr_c1_conv_3x3,
                  nr_c2_conv_1x1,
                  nr_c2_conv_5x5,
                  nr_c3_conv_1x1,
                  return_output_channels=False):

    g = Graph()

    c0 = Sequential()
    c0.add(Convolution2D(nr_c0_conv_1x1, input_channel, 1, 1))
    c0.add(Activation('relu'))

    c1 = Sequential()
    c1.add(Convolution2D(nr_c1_conv_1x1, input_channel, 1, 1))
    c1.add(Activation('relu'))
    c1.add(Convolution2D(nr_c1_conv_3x3, nr_c1_conv_1x1, 3, 3,
                         border_mode='same'))
    c1.add(Activation('relu'))

    c2 = Sequential()
    c2.add(Convolution2D(nr_c2_conv_1x1, input_channel, 1, 1))
    c2.add(Activation('relu'))
    c2.add(Convolution2D(nr_c2_conv_5x5, nr_c2_conv_1x1, 5, 5,
                         border_mode='same'))
    c2.add(Activation('relu'))

    c3 = Sequential()
    c3.add(ZeroPadding2D(pad=(1, 1)))
    c3.add(MaxPooling2D(poolsize=(3, 3), stride=(1,1)))
    c3.add(Convolution2D(nr_c3_conv_1x1, input_channel, 1, 1))
    c3.add(Activation('relu'))

    for c in [c0, c1, c2, c3]:
        c.add(Permute((2, 3, 1)))

    g.add_input('input', ndim=4)
    g.add_node(c0, name='c0', input='input')

    g.add_node(c1, name='c1', input='input')
    g.add_node(c2, name='c2', input='input')
    g.add_node(c3, name='c3', input='input')

    g.add_node(Permute((3, 1 ,2)), inputs=['c0', 'c1', 'c2', 'c3'],
               name='last_permute',
               merge_mode='concat')
    g.add_output(name='output', input='last_permute')

    if return_output_channels:
        return g, nr_c0_conv_1x1 + nr_c1_conv_3x3 + nr_c2_conv_5x5 + \
            nr_c3_conv_1x1
    return g


def log_l2_loss(y_truth, y_pred):
    return T.log(T.mean(T.sqr(y_truth - y_pred)))



# vim: foldmethod=marker
