#!/usr/bin/bash
morphologyModel="czech-morfflex2.1-pdtc2.0-250909/czech-morfflex2.1-250909.dict"
cd /storage/praha1/home/tondaotm/npfl087
./morphodita-1.11.3-bin/bin-linux64/run_morpho_generate $morphologyModel 0 < /dev/stdin 