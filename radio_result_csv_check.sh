#!/bin/sh

for version in 1 2 3 4 5;do
    output_root=/media/akio/hdd1/20230515/test/version_${version}/
    output_csv=${output_root}/output.csv
    python3 csv_check.py --csv $output_csv
done
