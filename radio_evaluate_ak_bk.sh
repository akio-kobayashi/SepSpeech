#!/bin/sh

model_type=unet
version=1
for station in ak bk;
do
    for testset in ak bk;
    do
	if [ "$testset" == 'ak' ];then
	    input_csv=audio/csv/test.csv
	else
	    input_csv=audio/csv/bk_test_500.csv
	fi
	#echo $station $input_csv
	config=${model_type}/${station}/lightning_logs/version_${version}/hparams.yaml
	checkpoint=${model_type}/${station}/lightning_logs/version_${version}/checkpoints/last.ckpt
    
	output_root=/media/akio/hdd1/20230515/${model_type}/${station}/version_${version}/
	if [ -e $output_root ];then
	    rm -rf $output_root
	fi
	output_root=${output_root}/${testset}
	mkdir -p $output_root
    
	output_csv=${output_root}/output.csv
	output_dir=${output_root}/evaluate
	if [ -e $output_dir ];then
	    rm -rf $output_dir
	fi
	mkdir -p $output_dir
    
	python3 radio_evaluate.py --config $config --checkpoint $checkpoint \
    		--input_csv $input_csv --output_csv $output_csv \
    		--output_dir $output_dir --model_type $model_type
    done
done

