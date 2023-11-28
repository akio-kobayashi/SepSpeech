#!/bin/sh

rootdir=/export1/Speech/mixture/test_b01-25/estimate
csvdir=audio/csv/test/

config=tasnet/lightning_logs/version_6/hparams.yaml
model_type=tasnet
checkpoint=tasnet/lightning_logs/version_6/checkpoints/last.ckpt

# opus encoded speech
for snr in 60 20 10 0;
#for snr in 0;
do
    #for bps in 6000 12000 16000;
    #for bps in 16000;
    for bps in 6000 12000;
    do
	for packet_loss_rate in 0.00001 0.05 0.1;
	#for packet_loss_rate in 0.1;
	do
	    for gender in male female;
	    do
		input_csv=${csvdir}/test_b_01-25_${gender}_snr${snr}_bps${bps}_plr${packet_loss_rate}.csv
		enroll_csv=${csvdir}/jnas_b_26-50.csv
		for speaker in F067 F068 F069 F070 F108 F109 F110 F111 F121 F122 F123 F124 F143B F144B F145B F146B M005 M006 M040 M041 M042 M043 M081 M082 M083 M084 M139B M140B M141B M142B M148 MP02;
		do
		    speaker_input_csv=/tmp/input.csv
		    speaker_enroll_csv=/tmp/enroll.csv
		    
		    python3 -m audio.pick_csv --csv $input_csv --output-csv $speaker_input_csv --speaker $speaker
		    python3 -m audio.pick_csv --csv $enroll_csv --output-csv $speaker_enroll_csv --speaker $speaker
		    
		    output_csv_dir=${csvdir}/output/${model_type}/test_b01-25/${snr}/${bps}/${packet_loss_rate}/${gender}/${speaker}
		    if [ ! -e $output_csv_dir ];then
			mkdir -p $output_csv_dir
		    fi
		    output_csv=${output_csv_dir}/output.csv
		    output_dir=${rootdir}/${model_type}/test_b01-25/${snr}/${bps}/${packet_loss_rate}/${gender}/${speaker}
		    if [ ! -e $output_dir ];then
			mkdir -p $output_dir
		    fi
		    
		    python3 -m bin.evaluate --input_csv $speaker_input_csv \
			    --enroll_csv $speaker_enroll_csv \
			    --output_csv $output_csv \
			    --output_dir $output_dir \
			    --model_type $model_type \
			    --config $config \
			    --checkpoint $checkpoint
		    rm -f $speaker_input_csv
		    rm -f $speaker_enroll_csv
		    
		done
	    done
	done
    done
done

