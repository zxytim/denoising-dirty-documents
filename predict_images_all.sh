#!/bin/bash -e

[ $# != 1 ] && echo "Usage: $0 <config_file>" && exit 1

set  -x

config_file="$1"
bname=$(basename "$config_file")
name=${bname%.*}

for dataset in train val test; do
	echo ./theano gpu4 \
		./predict_images.py \
		--image_list data/${dataset}.list \
		--config $config_file \
		--dump_prefix train_log/${name}/min_val_loss \
		--output_dir out/${name}/${dataset}
done | parallel bash -c {}
