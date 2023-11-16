#!/bin/sh

dir=logs/braille/version_2
config=${dir}/hparams.yaml
model=${dir}/models/best.pt
output=${dir}/models/test_greedy.trn
python3 decode.py --config $config --model $model --output $output --greedy

dir=logs/kanji/version_1/
config=${dir}/hparams.yaml
model=${dir}/models/best.pt
output=${dir}/models/test_greedy.trn
python3 decode.py --config $config --model $model --output $output --greedy

dir=logs/transformer/version_5/
config=${dir}/hparams.yaml
model=${dir}/models/best.pt
output=${dir}/models/test_greedy.trn
python3 decode_trans.py --config $config --model $model  --output $output --greedy

dir=logs/mtl/version_2/
config=${dir}/hparams.yaml
model=${dir}/models/best.pt
output=${dir}/models/test_greedy.trn
python3 decode_ctc.py --config $config --model $model --output $output --greedy

dir=logs/mtl/version_4/
config=${dir}/hparams.yaml
model=${dir}/models/best.pt
output=${dir}/models/test_greedy.trn
python3 decode_ctc.py --config $config --model $model --output $output --greedy
