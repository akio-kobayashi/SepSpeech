#!/bin/sh

#for version in 1 2 3 4;do
for version in 5;do
    #version=4
    config=lightning_logs/version_${version}/hparams.yaml
    if [[ $version == 1 ]]; then
	checkpoint=lightning_logs/version_${version}/checkpoints/checkpoint_epoch=144-step=174725-valid_loss=0.078.ckpt
    elif [[ $version == 2 ]]; then
	checkpoint=lightning_logs/version_${version}/checkpoints/checkpoint_epoch=176-step=213285-valid_loss=0.077.ckpt
    elif [[ $version == 3 ]]; then
	checkpoint=lightning_logs/version_${version}/checkpoints/checkpoint_epoch=159-step=192800-valid_loss=0.053.ckpt
    elif [[ $version == 4 ]]; then
	checkpoint=lightning_logs/version_${version}/checkpoints/checkpoint_epoch=153-step=185570-valid_loss=0.044.ckpt
    elif [[ $version == 5 ]];then
	checkpoint=lightning_logs/version_${version}/checkpoints/checkpoint_epoch=184-step=222925-valid_loss=2.600.ckpt
    fi
    echo $checkpoint
    #config=radio.yaml
    #checkpoint=lightning_logs/version_1/checkpoints/checkpoint_epoch\=144-step\=174725-valid_loss\=0.078.ckpt
    input_csv=audio/csv/test.csv
    output_root=/media/akio/hdd1/20230515/test/version_${version}/
    if [ ! -e $output_root ]; then
	mkdir -p $output_root
    fi
    output_csv=${output_root}/output.csv
    output_dir=${output_root}/evaluate
    if [ ! -e $output_dir ];then
	mkdir -p $output_dir
    fi
    model_type=unet

    python3 radio_evaluate.py --config $config --checkpoint $checkpoint \
	    --input_csv $input_csv --output_csv $output_csv \
	    --output_dir $output_dir --model_type $model_type
done
