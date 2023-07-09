#!/bin/sh

python3 decode.py --config config.yaml \
	--checkpoint lightning_logs/version_3/checkpoints/checkpoint_epoch=9-step=24960-valid_loss=17.491.ckpt \
	--output temp.trn
