#!/bin/bash
for c in "corrupt" "broken" "corruptTriple"; do
  for m in "ayaexp" "commandr" "gemmabig" "llama" "mistral7b" "tower"; do
    echo check ./output30/perps/by$1/$c/of$m.txt
    if [ -f ./output30/perps/by$1/$c/of$m.txt ];then
      rm ./output30/perps/by$1/$c/of$m.txt
      echo deleted ./output30/perps/by$1/$c/of$m.txt
    fi
  done
done
