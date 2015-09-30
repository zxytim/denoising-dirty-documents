#!/bin/bash
for i in out/*; do [ -d $i ] && echo ./predict_images_all.sh config/$(basename $i).py; done | parallel bash -c {}
