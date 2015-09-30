#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: compute_rmse.py
# $Date: Wed Sep 30 12:33:26 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>


import argparse
import glob
import os
import cv2
from sklearn.metrics import mean_squared_error

from collections import namedtuple


class ImageItem(object):
    def __init__(self, fpath):
        self.fpath = fpath
        self.fname = os.path.basename(fpath)
        self.image = cv2.imread(fpath)

        self.id = self.fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='truth_dir')
    parser.add_argument(dest='pred_dir')
    args = parser.parse_args()

    pred_items = map(ImageItem, glob.glob(args.pred_dir + "/*.png"))
    truth_items = map(ImageItem, glob.glob(args.truth_dir + "/*.png"))
    truth_dict = {i.id: i for i in truth_items}


    rmse = 0
    for p in pred_items:
        assert p.id in truth_dict, p.id
        t = truth_dict[p.id]
        rmse = mean_squared_error(p.image / 255.0, t.image / 255.0)**0.5
    rmse /= len(pred_items)

    print 'RMSE: ', rmse


if __name__ == '__main__':
    main()


# vim: foldmethod=marker
