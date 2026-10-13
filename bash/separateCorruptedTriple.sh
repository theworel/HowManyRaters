#!/bin/bash
# Runs separateAnswersCorrupted.py (or the Thinking variant for thinking models)
# on every resources/out/{model}corruptedTriple_cs.txt file, writing the result
# to resources/output30/corruptTriple/{model}.txt.
# Afterwards it strips special tokens and deletes empty lines from all outputs.

pythonCommand="../v/bin/python3.11"
cd /storage/praha1/home/tondaotm/npfl087/npfl087-otmara

prompt="./resources/prompts/en_to_cs_single.txt"
inDir="./resources/out"
outDir="./resources/output30/corruptTriple"
thinkTag="</think>"

# Models whose output contains a thinking block -> use the Thinking separator.
thinkingModels=("darwin")

mkdir -p "$outDir"

isThinking() {
    local name="$1"
    for m in "${thinkingModels[@]}"; do
        if [ "$m" == "$name" ]; then
            return 0
        fi
    done
    return 1
}

createdFiles=()

for inFile in "$inDir"/*corruptedTriple_cs.txt; do
    [ -e "$inFile" ] || continue
    base=$(basename "$inFile")
    modelName="${base%corruptedTriple_cs.txt}"
    outFile="$outDir/$modelName.txt"

    if isThinking "$modelName"; then
        echo "[thinking] $modelName"
        $pythonCommand ./python/separateAnswersCorruptedThinking.py "$inFile" "$outFile" "$prompt" "$thinkTag"
    else
        echo "[normal]   $modelName"
        $pythonCommand ./python/separateAnswersCorrupted.py "$inFile" "$outFile" "$prompt"
    fi

    createdFiles+=("$outFile")
done

# Post-process every created file.
for f in "${createdFiles[@]}"; do
    $pythonCommand ./python/removeSpecialTokens.py "$f"
    bash ./bash/deleteEmptyLines.sh "$f"
done
