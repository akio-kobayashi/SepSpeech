#!/bin/sh

rootdir=/media/fujie/hdd1/Speech/test
csvdir=audio/csv/deaf/

cofig=config.yaml
#model_type=tasnet
#checkpoint=tasnet_logs/version_2/checkpoints/checkpoint_epoch\=394-step\=308890-valid_loss\=0.047.ckpt
model_type=unet
checkpoint=lightning_logs/version_3/checkpoints/checkpoint_epoch\=371-step\=290904-valid_loss\=0.042.ckpt

for speaker in M023 M024 F013 F014 F018;
do
    for gender in  male female;
    do
	for snr in 0 5 10 20;
	do
	    input_csv=${csvdir}/mix/${speaker}_${gender}_${snr}.csv
	    enroll_csv=${csvdir}/${speaker}C.csv
	    output_csv_dir=${csvdir}/mix/output/${model_type}
	    if [ ! -e $output_csv_dir ];then
		mkdir -p $output_csv_dir
	    fi
	    output_csv=${output_csv_dir}/${speaker}_${gender}_${snr}_output.csv
	    output_dir=${rootdir}/estimate/$model_type/${snr}/${gender}/${speaker}
	    if [ ! -e $output_dir ];then
		mkdir -p $output_dir
	    fi
	    
	    python3 -m bin.evaluate.py --input_csv $input_csv \
		    --enroll_csv $enroll_csv \
		    --output_csv $output_csv \
		    --output_dir $output_dir \
		    --model_type $model_type \
		    --config config.yaml \
		    --checkpoint $checkpoint
	done
    done
done
