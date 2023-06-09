#!/bin/sh

config=radio.yaml
checkpoint=lightning_logs/version_1/checkpoints/checkpoint_epoch\=290-step\=701310-valid_loss\=0.065.ckpt
input_csv=audio/csv/test.csv
output_csv=/media/akio/hdd1/20230515/test/csv/out.csv
output_dir=/media/akio/hdd1/20230515/test/estimate
model_type=unet

python3 radio_evaluate.py --config $config --checkpoint $checkpoint \
	--input_csv $input_csv --output_csv $output_csv \
	--output_dir $output_dir --model_type $model_type

