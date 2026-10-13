#!/bin/bash
# $1 - model name, $2 - source file, $3 source lang, $4 target lang
pythonCommand="../v/bin/python3.11"
modelName=$1
cd /storage/praha1/home/tondaotm/npfl087/npfl087-otmara
$pythonCommand ./python/generateBadTranslations.py $modelName $2 $3 $4