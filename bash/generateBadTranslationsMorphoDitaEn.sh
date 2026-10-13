#!/usr/bin/bash
outputfile=$1
infile=$2
morphoditaCommand="./morphodita-1.11.3-bin/bin-linux64/"
pythonCommand="./v/bin/python3.11"
morphologyModel="english-morphium-wsj-140407/english-morphium-14047-no_negation.dict"

${morphoditaCommand}run_morpho_analyze $morphologyModel 0 $infile --output=vertical | $pythonCommand ./npfl087-otmara/python/breakTagsEn.py | ${morphoditaCommand}run_morpho_generate $morphologyModel 0 | $pythonCommand ./npfl087-otmara/python/toSentences.py > $outputfile