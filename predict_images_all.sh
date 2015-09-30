#!/bin/bash -e

[ $# != 1 ] && echo "Usage: $0 <config_file>" && exit 1

set  -x

make

config_file="$1"
bname=$(basename "$config_file")
name=${bname%.*}


num_files() {
	local datset="$1"
	[ $dataset = "train" ] && echo 115
	[ $dataset = "val" ] && echo 29
	[ $dataset = "test" ] && echo 72
}

for dataset in train val test; do
	out_dir=out/${name}/${dataset}
	dump_prefix=train_log/${name}/min_val_loss

#    [ $(ls "$out_dir" | wc -l) == $(num_files $dataset) ] && continue

	echo "./theano gpu4 \
		./predict_images.py \
		--image_list data/${dataset}.list \
		--config $config_file \
		--dump_prefix $dump_prefix \
		--output_dir $out_dir"
done | parallel bash -c {}


csv_path=out/${name}.test.csv
zip_path=${csv_path}.test.zip
./images2csv out/${name}/test $csv_path && 7z a $zip_path $csv_path && rm $csv_path
