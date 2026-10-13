#!/bin/bash
# $1 - model name
# Builds the candidateCorrupt class for one model: the model LLM-corrupts its
# own good-class answers, every prompt block kept verbatim. Unlike the corrupt
# class this writes resources/output30/candidateCorrupt/{model}.txt straight
# away -- the blocks are already in class format, so there is no separateAnswers
# step -- and then applies the usual post-processing.
pythonCommand="../v/bin/python3.11"
modelName=$1
cd /storage/praha1/home/tondaotm/npfl087/npfl087-otmara

inFile="./resources/output30/good/${modelName}.txt"
outDir="./resources/output30/candidateCorrupt"
outFile="${outDir}/${modelName}.txt"

if [ ! -s "$inFile" ]; then
    echo "no good-class input for ${modelName}: $inFile" >&2
    exit 1
fi

mkdir -p "$outDir"

$pythonCommand ./python/corruptSentencesLLMAnswerOnly.py $modelName < "$inFile" > "$outFile"

$pythonCommand ./python/removeSpecialTokens.py "$outFile"
bash ./bash/deleteEmptyLines.sh "$outFile"
