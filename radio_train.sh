#!/bin/sh

trap 'exit 0' SIGINT
trap 'python3 -m radio_train --config radio.yaml --gpus 0 --checkpoint unet/ak/lightning_logs/version_1/checkpoints/last.ckpt' SIGSEGV

python3 -m radio_train --config radio.yaml --gpus 0 --checkpoint unet/ak/lightning_logs/version_1/checkpoints/last.ckpt
