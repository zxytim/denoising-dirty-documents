#!/bin/bash
for i in train_log/*; do [ -d $i ] && echo ./predict_images_all.sh config/$(basename $i).py; done | parallel bash -c {}
