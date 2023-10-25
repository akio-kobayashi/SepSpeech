#!/bin/sh

ref_csv=/media/akio/hdd1/20230515/test/reference_bk.csv

for version in 1 2 3 4 5 6;
do
    root=/media/akio/hdd1/20230515/test/bk/version_${version}/
    hyp_csv=${root}/output.csv
    python3 -m audio.csv.extract_hyp_ref --ref-csv $ref_csv --hyp-csv $hyp_csv --output-dir $root

    ref_trn=$root/ref.trn

    for rtype in clean noisy denoise;
    do
	hyp_trn=$root/${rtype}.trn
	docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media akiokobayashi/sctk:latest \
	       sclite -r $ref_trn trn -h $hyp_trn trn -i rm -o sum prf dtl > /dev/null
    done
done
