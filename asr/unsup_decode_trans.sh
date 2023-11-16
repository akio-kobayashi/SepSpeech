#!/bin/sh

python3 unsup_decode_trans.py --config logs/transformer/version_5/hparams.yaml --model logs/transformer/version_5/models/best.pt --input_csv data/unsup_test_text.csv --output_csv logs/transformer/version_5/models/unsup_test_decode.csv
