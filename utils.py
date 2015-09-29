#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: utils.py
# $Date: Wed Sep 30 02:18:27 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>


import logging

import os
import cPickle as pickle
import logconf
import cv2


def set_logging_file(filename):
    hdl = logging.FileHandler(
        filename=filename, mode='a', encoding='utf-8')
    logging.getLogger('__main__').addHandler(hdl)
    logging.getLogger('theano').addHandler(hdl)
    logging.getLogger('keras').addHandler(hdl)


def mkdir_p(dirname):
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != 17:
            raise e



def serialize(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def deserialize(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(model, history, prefix):
    model.save_weights(prefix + '.model_weights', overwrite=True)
    serialize(history, prefix + '.history')


def load_model(model, prefix):
    model.load_weights(prefix + '.model_weights')
    history = deserialize(prefix + '.history')
    return history


def read_image_list(fpath):
    base_dir = os.path.dirname(fpath)
    with open(fpath) as f:
        lines = [line.rstrip().split() for line in f]
        lines = [[os.path.join(base_dir, p) for p in line] for line in lines]
    return lines


# vim: foldmethod=marker
