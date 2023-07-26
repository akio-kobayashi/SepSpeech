#!/bin/sh

#decode=$1
hyp=$1
reference=$2

#dir=`dirname $decode`
#python3 -m bin.sort_trn --trn $decode --output $dir/sorted.trn --split
docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media akiokobayashi/sctk:latest \
sclite -r $reference trn -h $hyp trn -i rm -o sum prf dtl > /dev/null
