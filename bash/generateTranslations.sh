#!/bin/bash
pythonCommand="../../v/bin/python3"
modelName=$1
$pythonCommand ../python/generateTranslations.py $modelName ../resources/good/cs.txt ../resources/good/en.txt $2 $3 $4
