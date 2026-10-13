#!/bin/bash
  for m in "ayaexp" "commandr" "gemmabig" "llama" "mistral7b" "tower" "ibm" "qwen" "gemma" "gptoss" "darwin" "dolphin" "eu" "aya" "ayabig"; do
    echo check ./output30/perps/by$m/$2/of$1.txt
    if [ -f ./output30/perps/by$m/$2/of$1.txt ];then
      rm ./output30/perps/by$m/$2/of$1.txt
      echo deleted ./output30/perps/by$m/$2/of$1.txt
    fi
done
