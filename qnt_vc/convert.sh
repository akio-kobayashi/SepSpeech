#!/bin/sh

mkdir -p audio/vc/

# /export1/Speech/jnas/JNAS_1/WAVES_DT/F001/NP/NF001001_DT.wav,108176,F001,70,001
cat audio/csv/jnas.csv | sed -e "s/\/export1\/Speech/\/home\/akio\/data/g" > audio/vc/jnas.csv
cat audio/csv/jnas_removed.csv | sed -e "s/\/export1\/Speech/\/home\/akio\/data/g" > audio/vc/jnas_removed.csv

# /export1/DeafSpeech/16k/F001/B/F001B01.wav,276978,DF001,326,B01
cat audio/csv/deaf/deaf.csv | sed -e "s/\/export1/\/home\/akio\/data/g" > audio/vc/deaf.csv
cat audio/csv/deaf/deaf_remove_b_c.csv | sed -e "s/\/export1/\/home\/akio\/data/g" > audio/vc/deaf_removed.csv
