#!/bin/sh

rt60=0.6
config=config.yaml

mkdir -p audio/csv/test/
#bset=audio/csv/jnas_b.csv
#grep -e "B01" -e "B02" -e "B03" -e "B04" -e "B05" -e "B06" -e "B07" -e "B08" -e "B09" -e "B10" \
#     -e "B11" -e "B12" -e "B13" -e "B14" -e "B15" -e "B16" -e "B17" -e "B18" -e "B19" -e "B20" \
#     -e "B21" -e "B22" -e "B23" -e "B24" -e "B25" -e source $bset \
#     > audio/csv/test/jnas_b_01-25.csv

#grep -e "B26" -e "B27" -e "B28" -e "B29" -e "B30" -e source \
#     -e "B31" -e "B32" -e "B33" -e "B34" -e "B35" -e "B36" -e "B37" -e "B38" -e "B39" -e "B40" \
#     -e "B41" -e "B42" -e "B43" -e "B44" -e "B45" -e "B46" -e "B47" -e "B48" -e "B49" -e "B50" $bset \
#     > audio/csv/test/jnas_b_26-50.csv

speech_csv=audio/csv/test/jnas_b_01-25.csv
#noise_csv=audio/csv/noise_test.csv
# female w/o packet loss
#gender=female
#mix_csv=audio/csv/test/test_b_01-25_female_snr20_bps6000_plr0.00001.csv
# male w/o packet loss
#gender=male
#mix_csv=audio/csv/test/test_b_01-25_male_snr20_bps6000_plr0.00001.csv

# female w/ packet loss 0.05
#gender=female
#mix_csv=audio/csv/test/test_b_01-25_female_snr20_bps6000_plr0.05.csv
# male w/ packet loss 0.05
#gender=male
#mix_csv=audio/csv/test/test_b_01-25_male_snr20_bps6000_plr0.05.csv

# female w/ packet loss 0.1
#gender=female
#mix_csv=audio/csv/test/test_b_01-25_female_snr20_bps6000_plr0.1.csv
# female w/ packet loss 0.1
#gender=male
#mix_csv=audio/csv/test/test_b_01-25_male_snr20_bps6000_plr0.1.csv

for snr in 60 20 10 5 0;
do
    for bps in 6000 12000 16000;
    do
	for gender in male female;
	do
	    noise_csv=audio/csv/noise_test_${gender}.csv
	    for packet_loss_rate in 0.00001 0.05 0.1;
	    do
		output_csv=audio/csv/test/test_b_01-25_${gender}_snr${snr}_bps${bps}_plr${packet_loss_rate}.csv
		if [ -e $output_csv ];then
		    echo skip $output_csv
		    continue
		fi
		if [ "$gender" == "male" ]; then
		    if [ "$packet_loss_rate" == "0.00001" ];then
			mix_csv=audio/csv/test/test_b_01-25_male_snr20_bps6000_plr0.00001.csv
		    elif [ "$packet_loss_rate" == "0.05" ];then
			mix_csv=audio/csv/test/test_b_01-25_male_snr20_bps6000_plr0.05.csv
		    else
			mix_csv=audio/csv/test/test_b_01-25_male_snr20_bps6000_plr0.1.csv
		    fi
		else
		    if [ "$packet_loss_rate" == "0.00001" ];then
			mix_csv=audio/csv/test/test_b_01-25_female_snr20_bps6000_plr0.00001.csv
		    elif [ "$packet_loss_rate" == "0.05" ];then
			mix_csv=audio/csv/test/test_b_01-25_female_snr20_bps6000_plr0.05.csv
		    else
			mix_csv=audio/csv/test/test_b_01-25_female_snr20_bps6000_plr0.1.csv
		    fi
		fi
		echo $gender $packet_loss_rate $mix_csv
		output_dir=/export1/Speech/mixture/test_b01-25/${gender}/${snr}/${bps}/${packet_loss_rate}/
		mkdir -p $output_dir
		python3 -m audio.opus_mix_fix \
			--speech-csv $speech_csv \
			--noise-csv $noise_csv \
			--mixed-csv $mix_csv \
			--output-csv $output_csv \
			--output-dir $output_dir \
			--config $config \
			--snr $snr \
			--rt60 $rt60 \
			--bps $bps \
			--packet_loss_rate $packet_loss_rate
	    done
	done
    done
done

