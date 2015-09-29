#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: predict_images.py
# $Date: Wed Sep 30 00:53:38 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>


import argparse
import imp
from progressbar import ProgressBar
import numpy as np
import cv2
import os

import utils

import logging
logger = logging.getLogger(__name__)

from keras.optimizers import Adam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dump_prefix', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    conf_mod = imp.load_source('config', args.config)
    config = conf_mod.get()

    model = config['model']
    utils.load_model(model, args.dump_prefix)

    X, _ = conf_mod.get_data(args.image_list)

    utils.mkdir_p(args.output_dir)
    image_list = utils.read_image_list(args.image_list)

    logger.info('compiling model ...')
    model.compile(loss='mean_squared_error', optimizer=Adam())

    for x, (input_path, _) in ProgressBar()(zip(X, image_list)):
        y = model.predict(np.array([x], dtype='float32'),
                               batch_size=1, verbose=False)
        img = (y.reshape(y.shape[2:]) * 255.0).astype('uint8')

        # FIXME: we assume that basenames of images are distinct
        fname = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(output_path, img)


if __name__ == '__main__':
    main()


# vim: foldmethod=marker
