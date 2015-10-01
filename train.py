#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: train.py
# $Date: Thu Oct 01 12:02:29 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>


import logconf
import math

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, \
    Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

import os
import cv2
import numpy as np
import argparse

from progressbar import ProgressBar
import imp
import tabulate

from utils import mkdir_p, load_model, save_model, set_logging_file


import logging
logger = logging.getLogger(__name__)


def summarize_history(histories):
    loss = np.mean([h.totals['loss'] for h in histories])
    return dict(
        loss=loss,
        sqrt_loss=-math.sqrt(-loss) if loss < 0 else math.sqrt(loss),
        loss_exp_sqrt=math.sqrt(math.exp(loss))
    )


def train_images_one_by_one(model, X_train, Y_train, indexes):
    '''X_train, Y_train are list of (c, h, w) images'''
    assert len(X_train) == len(Y_train) == len(indexes), (
        len(X_train), len(Y_train), len(indexes))

    pbar = ProgressBar()

    histories = []
    for i in pbar(indexes):
        hist = model.fit(np.array(X_train[i:i+1], dtype='float32'),
                         np.array(Y_train[i:i+1], dtype='float32'),
                         batch_size=1, nb_epoch=1, verbose=False)
        histories.append(hist)
    return summarize_history(histories)


def validate_images_one_by_one(model, X_val, Y_val):
    pbar = ProgressBar()

    loss = 0
    for i in pbar(range(len(X_val))):
        loss += model.evaluate(
            np.array(X_val[i:i+1], dtype='float32'),
            np.array(Y_val[i:i+1], dtype='float32'),
            batch_size=1, verbose=False)
    loss /= len(X_val)

    return dict(loss=loss,
                sqrt_loss=-math.sqrt(-loss) if loss < 0 else math.sqrt(loss),
                loss_exp_sqrt=math.sqrt(math.exp(loss))
                )

def get_history_min_val_loss(history):
    if len(history) == 0:
        return float('inf')
    return min(h['val_summary']['loss'] for h in history)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('--log_dir_base', default='train_log')
    parser.add_argument('--seed', default='42')
    parser.add_argument('--load')
    args = parser.parse_args()

    name, _ = os.path.splitext(os.path.basename(args.config))
    log_dir = os.path.join(args.log_dir_base, name)

    mkdir_p(log_dir)
    set_logging_file(os.path.join(log_dir, 'training.log'))

    model_dump_paths = dict(
        latest=os.path.join(log_dir, 'lastest'),
        min_val_loss=os.path.join(log_dir, 'min_val_loss')
    )

    config = imp.load_source('config', args.config).get()

    model = config['model']
    epoch = 0
    history = []

    if args.load:
        logger.info('loading snapshot: {}'.format(args.load))
        history = load_model(model, args.load)

    # TODO load model

    rng = np.random.RandomState(hash(args.seed))

    optimizer = Adam()
    model.compile(loss=config.get('loss', 'mean_squared_error'),
                                  optimizer=optimizer)

    (X_train, Y_train), (X_val, Y_val) = \
        config['data_train'], config['data_val']


    min_val_loss = get_history_min_val_loss(history)
    epoch = len(history)

    indexes = np.arange(len(X_train))
    while True:
        epoch += 1
        logger.info('starting epoch {}'.format(epoch))

        rng.shuffle(indexes)
        train_summary = train_images_one_by_one(
            model, X_train, Y_train, indexes)
        logger.info('training summary:\n' +
                    tabulate.tabulate(train_summary.items()))

        logger.info('validating ...')
        val_summary = validate_images_one_by_one(model, X_val, Y_val)
        val_loss = val_summary['loss']
        logger.info('validation summary:\n' +
                    tabulate.tabulate(
                        val_summary.items() +
                        [('min_val_loss', min(val_loss, min_val_loss))]
                    ))

        cur_status = dict(
            epoch=epoch,
            train_summary=train_summary,
            val_summary=val_summary,
        )
        history.append(cur_status)

        logger.info('saving model')
        save_model(model, history, model_dump_paths['latest'])
        if  val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model(model, history, model_dump_paths['min_val_loss'])


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
