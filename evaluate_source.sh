#!/bin/sh

csvdir=audio/csv/deaf/

for speaker in M023 M024 F013 F014 F018;
do
    input_csv=${csvdir}/${speaker}B.csv
    output_csv_dir=${csvdir}/mix/output/source
    if [ ! -e $output_csv_dir ];then
	mkdir -p $output_csv_dir
    fi
    output_csv=${output_csv_dir}/${speaker}B_output.csv
    python3 evaluate_source.py --input_csv $input_csv --output_csv $output_csv 
done
