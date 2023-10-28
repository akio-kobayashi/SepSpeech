#!/bin/sh

outdir=/home/akio/data/qnt/jnas/
input_csv=audio/vc/jnas_removed.csv
output_csv=audio/vc/qnt_jnas_removed.csv
python3 -m qnt_vc.rename --input_csv $input_csv --output_csv $output_csv --output_root $outdir

outdir=/home/akio/data/qnt/deaf/
input_csv=audio/vc/deaf.csv
output_csv=audio/vc/qnt_deaf.csv
python3 -m qnt_vc.rename --input_csv $input_csv --output_csv $output_csv --output_root $outdir
