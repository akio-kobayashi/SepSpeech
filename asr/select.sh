#!/bin/sh

for posterior in 0.2 0.3 0.5 ;do 
    python3 select_csv.py --input_csv logs/braille/version_2/models/unsup_train_decode.csv --output_csv logs/braille/version_2/models/unsup_train_${posterior}.csv --text_dir /export/Broadcast/unsup/braille/ --posterior ${posterior}
    python3 select_csv.py --input_csv logs/kanji/version_1/models/unsup_train_decode.csv --output_csv logs/kanji/version_1/models/unsup_train_${posterior}.csv --text_dir /export/Broadcast/unsup/kanji/ --posterior ${posterior}
done
