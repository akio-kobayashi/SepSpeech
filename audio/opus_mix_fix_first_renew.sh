#!/bin/sh

rt60=0.6
config=config.yaml

mkdir -p audio/csv/test/

speech_csv=audio/csv/test/jnas_b_01-25.csv
#noise_csv=audio/csv/noise_test.csv

# female
gender=female
mix_csv=audio/csv/test/test_b_01-25_female_snr20_bps6000_plr0.00001.csv
# male
#gender=male
#mix_csv=audio/csv/test/test_b_01-25_male_snr20_bps6000_plr0.00001.csv

packet_loss_rate=0.1

for snr in 20;
do
    for bps in 6000;
    do
	noise_csv=audio/csv/noise_test_${gender}.csv
	output_csv=audio/csv/test/test_b_01-25_${gender}_snr${snr}_bps${bps}_plr${packet_loss_rate}.csv
	output_dir=/export1/Speech/mixture/test_b01-25/${gender}/${snr}/${bps}/${packet_loss_rate}/
	mkdir -p $output_dir
	python3 -m audio.opus_mix_fix \
		--speech-csv $speech_csv \
		--noise-csv $noise_csv \
		--output-csv $output_csv \
		--output-dir $output_dir \
		--config $config \
		--snr $snr \
		--rt60 $rt60 \
		--bps $bps \
		--packet_loss_rate $packet_loss_rate \
		--mixed-csv $mix_csv \
		--renew_markov_states
	echo $output_csv
    done
done

