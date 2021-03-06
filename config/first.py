#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: first.py
# $Date: Wed Sep 30 00:05:35 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>


import math

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, \
    Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

import os
import cv2
import numpy as np

import utils

import logging
logger = logging.getLogger(__name__)


MODEL_INPUT_SHAPE = (11, 11)

assert MODEL_INPUT_SHAPE[0] % 2 == 1 and MODEL_INPUT_SHAPE[1] % 2 == 1


def get_model():
    model = Sequential()
    model.add(Convolution2D(8, 1, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 8, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(24, 16, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 24, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 32, 3, 3))
    model.add(Activation('relu'))

    # 1x1 here

    model.add(Convolution2D(1, 48, 1, 1))  # a fully connected layer
    model.add(Activation('sigmoid'))

    return model


def read_images(fpath):
    lines = utils.read_image_list(fpath)

    logger.info('loading data: {}'.format(fpath))
    X_data, y_data = [], []
    for inst_path, truth_path in lines:
        inst, truth = [cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                for p in (inst_path, truth_path)]
        assert inst is not None and truth is not None, (inst_path, truth_path)

        pad_h, pad_w = [x / 2 for x in MODEL_INPUT_SHAPE]
        inst = cv2.copyMakeBorder(inst, pad_h, pad_h, pad_w, pad_w,
                               cv2.BORDER_REFLECT)
        # pad input image
        X_data.append(inst)
        y_data.append(truth)

    return X_data, y_data


def image_list2tensor3d(imgs):
    return [(img.reshape((1, ) + img.shape) / 255.0).astype('float32')
            for img in imgs]


def get_data(list_path):
    return map(image_list2tensor3d, read_images(list_path))


def get():
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    train_list, val_list = [
        os.path.join(DATA_DIR, p) for p in ('train.list', 'val.list')]
    return dict(
        model=get_model(),
        data_train=get_data(train_list),
        data_val=get_data(val_list)
    )

# vim: foldmethod=marker
