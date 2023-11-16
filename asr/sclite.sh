#!/bin/sh

dir=$1
decode=${dir}/test_greedy.trn
reference=data/reference/braille_test_ref.trn

# w/o splitting words
python3 sort_trn.py --trn $decode --output ${dir}/sorted.trn
docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media akiokobayashi/sctk:latest \
sclite -r $reference trn -h ${dir}/sorted.trn trn -i rm -o sum prf dtl > /dev/null

# w/ splitting words including spaces
python3 split_include_space.py --trn $reference --output $dir/ref.trn
python3 split_include_space.py --trn $decode --output $dir/sorted_space.trn
docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media akiokobayashi/sctk:latest \
sclite -r ${dir}/ref.trn trn -h ${dir}/sorted_space.trn trn -i rm -o sum prf dtl > /dev/null

# w/ splitting words
reference=data/reference/braille_test_ref_split.trn
python3 sort_trn.py --trn $decode --output ${dir}/sorted_split.trn --split
docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media akiokobayashi/sctk:latest \
sclite -r $reference trn -h ${dir}/sorted_split.trn trn -i rm -o sum prf dtl > /dev/null
