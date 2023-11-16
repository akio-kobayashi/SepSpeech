#!/bin/sh

python3 unsup_decode.py --config logs/braille/version_2/hparams.yaml --model logs/braille/version_2/models/best.pt --input_csv data/unsup_valid_audio.csv --output_csv logs/braille/version_2/models/unsup_valid_decode.csv
python3 unsup_decode.py --config logs/kanji/version_1/hparams.yaml --model logs/kanji/version_1/models/best.pt --input_csv data/unsup_valid_audio.csv --output_csv logs/kanji/version_1/models/unsup_valid_decode.csv
