#!/bin/sh

rt60=0.6
config=config.yaml

mkdir -p audio/csv/test/
bset=audio/csv/jnas_b.csv
grep -e "B01" -e "B02" -e "B03" -e "B04" -e "B05" -e "B06" -e "B07" -e "B08" -e "B09" -e "B10" \
     -e "B11" -e "B12" -e "B13" -e "B14" -e "B15" -e "B16" -e "B17" -e "B18" -e "B19" -e "B20" \
     -e "B21" -e "B22" -e "B23" -e "B24" -e "B25" -e source $bset \
     > audio/csv/test/jnas_b_01-25.csv

grep -e "B26" -e "B27" -e "B28" -e "B29" -e "B30" -e source \
     -e "B31" -e "B32" -e "B33" -e "B34" -e "B35" -e "B36" -e "B37" -e "B38" -e "B39" -e "B40" \
     -e "B41" -e "B42" -e "B43" -e "B44" -e "B45" -e "B46" -e "B47" -e "B48" -e "B49" -e "B50" $bset \
     > audio/csv/test/jnas_b_26-50.csv

speech_csv=audio/csv/test/jnas_b_01-25.csv
#noise_csv=audio/csv/noise_test.csv

mix_csv=""
for snr in 20 10 0;
#for snr in 20;
do
    for bps in 6000 12000 16000;
    #for bps in 6000;
    do
	for packet_loss_rate in 0.00001 0.05;
	#for packet_loss_rate in 0.00001;
	do
	    for gender in female;
	    do
		noise_csv=audio/csv/noise_test_${gender}.csv
		output_csv=audio/csv/test/test_b_01-25_${gender}_snr${snr}_bps${bps}_plr${packet_loss_rate}.csv
		output_dir=/export1/Speech/mixture/test_b01-25/${gender}/${snr}/${bps}/${packet_loss_rate}/
		mkdir -p $output_dir
		if [ "$mixed" == "" ];then
		    #echo "first time"
		    python3 -m audio.opus_mix_fix \
			    --speech-csv $speech_csv \
			    --noise-csv $noise_csv \
			    --output-csv $output_csv \
			    --output-dir $output_dir \
			    --config $config \
			    --snr $snr \
			    --rt60 $rt60 \
			    --bps $bps \
			    --packet_loss_rate $packet_loss_rate
		else
		    #echo "second time"
		    python3 -m audio.opus_mix_fix \
			    --speech-csv $speech_csv \
			    --noise-csv $noise_csv \
			    --mixed-csv $mixed \
			    --output-csv $output_csv \
			    --output-dir $output_dir \
			    --config $config \
			    --snr $snr \
			    --rt60 $rt60 \
			    --bps $bps \
			    --packet_loss_rate $packet_loss_rate
		fi
		mixed=$output_csv
	    done
	done
    done
done

