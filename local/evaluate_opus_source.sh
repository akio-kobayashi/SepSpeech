#!/bin/sh

rootdir=/export1/Speech/mixture/test_b01-25/estimate
csvdir=audio/csv/test/

cofig=config.yaml
model_type=unet
checkpoint=lightning_logs/version_6/checkpoints/checkpoint_epoch=399-step=359600-valid_loss=0.075.ckpt

# source speech
input_csv=${csvdir}/jnas_b_01-25.csv
for speaker in F067 F068 F069 F070 F108 F109 F110 F111 F121 F122 F123 F124 F143B F144B F145B F146B M005 M006 M040 M041 M042 M043 M081 M082 M083 M084 M139B M140B M141B M142B M148 MP02;
do
    speaker_input_csv=/tmp/input.csv

    python3 -m audio.pick_csv --csv $input_csv --output-csv $speaker_input_csv --speaker $speaker

    output_csv_dir=${csvdir}/output/${model_type}/test_b01-25/source/$speaker
    if [ ! -e $output_csv_dir ];then
	mkdir -p $output_csv_dir
    fi
    output_csv=${output_csv_dir}/output.csv
		    
    python3 -m bin.evaluate_source --input_csv $speaker_input_csv \
	    --output_csv $output_csv \
	    --model_type $model_type 
    rm -f $speaker_input_csv
done
