#!/bin/sh

trap 'exit 1' SIGHUP SIGSEGV SIGINT

for station in ak bk;
do
    for condition in lowpass+noise lowpass+fading+noise lowpass+fading+noise+band_noise;
    do
	#for hours in 10 50 100 all;
	for hours in all;
	do
	    save_dir=unet/augment/${station}/${condition}/${hours}
	    if [ ! -e $save_dir ];then
		mkdir -p $save_dir
	    fi

	    version=1
	    finished=unet/augment/${station}/${condition}/${hours}/lightning_logs/${version}/finished
	    if [ -e $finished ];then
		continue
	    fi
	    
	    if [ "$hours" == "all" ];then
		train_csv=csv/augment/train/${station}_${condition}.csv
	    else
		train_csv=csv/augment/train/${station}_${condition}_${hours}.csv
	    fi
	    valid_csv=csv/${station}_valid.csv
	    
	    
	    ckpt=unet/augment/${station}/${condition}/${hours}/lightning_logs/${version}/checkpoints/last.ckpt
	    if [ -e $ckpt ];then
		args="--checkpoint $ckpt"
	    else
		args=""
	    fi
	    python3 rewrite_yaml.py --input radio.yaml --output temp.yaml --save_dir $save_dir \
		    --version 1 --train_csv $train_csv --valid_csv $valid_csv 
	
	    python3 radio_train.py --config temp.yaml --gpus 0 $args
	    echo "finished" > $finished
	done
    done
done




