#!/usr/bin/bash
outputfile=$1
infile=$2
morphoditaCommand="./morphodita-1.11.3-bin/bin-linux64/"
pythonCommand="./v/bin/python3"
morphologyModel="czech-morfflex2.1-pdtc2.0-250909/czech-morfflex2.1-250909.dict"

${morphoditaCommand}run_morpho_analyze $morphologyModel 0 $infile --output=vertical | $pythonCommand ./npfl087-otmara/python/breakTags.py | ${morphoditaCommand}run_morpho_generate $morphologyModel 0 | $pythonCommand ./npfl087-otmara/python/toSentences.py > $outputfile