#!/bin/sh

outdir=/home/akio/data/qnt/jnas/
mkdir -p $outdir
input_csv=audio/vc/jnas.csv
output_csv=audio/vc/qnt_jnas.csv
python3 -m qnt_vc.quantizer --input_csv $input_csv --output_csv $output_csv --output_root $outdir

outdir=/home/akio/data/qnt/deaf/
mkdir -p $outdir
input_csv=audio/vc/deaf.csv
output_csv=audio/vc/qnt_deaf.csv
python3 -m qnt_vc.quantizer --input_csv $input_csv --output_csv $output_csv --output_root $outdir
