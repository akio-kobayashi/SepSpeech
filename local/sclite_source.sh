#!/bin/sh

reference=audio/b_set_reference.trn
csvdir=audio/csv/test/output/unet/test_b01-25/source/

for speaker in F067 F068 F069 F070 F108 F109 F110 F111 F121 F122 F123 F124 F143B F144B F145B F146B M005 M006 M040 M041 M042 M043 M081 M082 M083 M084 M139B M140B M141B M142B M148 MP02;
do
    output_csv=${csvdir}/${speaker}/output.csv
    
    python3 -m bin.extract_trn --csv $output_csv --ref $reference --output_dir ${csvdir}/${speaker}/
    docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media akiokobayashi/sctk:latest \
	   sclite -r ${csvdir}/${speaker}/ref.trn trn -h ${csvdir}/${speaker}/hyp.trn trn -i rm -o sum prf dtl > /dev/null
done
