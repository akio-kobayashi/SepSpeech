#!/bin/sh

cwd=`pwd`
for model in ak bk;
do
    for station in ak bk;
    do
	for version in 1 2 3 4;
	do
	    test_csv=csv/${station}_test.csv
	    mkdir -p outputs/csv
	    output_csv=${cwd}/outputs/csv/md-${model}_st-${station}_v-${version}.csv
	    output_root=${cwd}/outputs/estimate/md-${model}/st-${station}/v-${version}/
	    mkdir -p $output_root
	    python3 inference.py --input_csv $test_csv --output_csv $output_csv \
		    --output_root $output_root \
		    --checkpoint unet/${model}/lightning_logs/version_${version}/checkpoints/last.ckpt \
		    --config unet/${model}/lightning_logs/version_1/hparams.yaml
	done
    done
done
