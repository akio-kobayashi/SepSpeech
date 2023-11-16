#!/bin/sh

dir=$1
decode=${dir}/decode.trn
#reference=data/reference/braille_test_ref.trn
reference=data/reference/kanji_test_ref.trn

#python3 sort_trn.py --trn $decode --output ${dir}/sorted.trn
#docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media akiokobayashi/sctk:latest \
#sclite -r $reference trn -h ${dir}/sorted.trn trn -i rm -o sum prf dtl > /dev/null

#reference=data/reference/braille_test_ref_split.trn
python3 sort_trn.py --trn $decode --output ${dir}/sorted.trn --split
docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media akiokobayashi/sctk:latest \
sclite -r $reference trn -h ${dir}/sorted.trn trn -i rm -o sum prf dtl > /dev/null
