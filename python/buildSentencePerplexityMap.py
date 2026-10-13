"""Cache raw per-sentence perplexities for every (class, generator, measurer)
triple in one JSON file, organized so index i is always source sentence i.

Every generator translates the same source file in order, so position i in
*any* generator's per-class perplexity file already corresponds to the same
source sentence i --- there is no separate "sentence id" to compute. What is
expensive is re-globbing and re-parsing PERPS_DIR/by<measurer>/<cls>/*.txt
(13 measurers x ~13 generators x several classes) on every single script
invocation. This script does that scan once and writes the result as:

{
  "<cls>": {
    "<generator_stem>": {
      "<measurer>": [perp_0, perp_1, ...],   # index i == source sentence i
      ...
    },
    ...
  },
  ...
}

so a "source sentence -> perplexities of its candidate translations" lookup
for sentence i of class cls is just: {stem: byMeasurer[stem][i] for stem in
data[cls] for measurer in data[cls][stem] if i < len(data[cls][stem][measurer])}.

scoreSeparationSampledEnsembleNormalizedLogSplitBySentence.py reads this same
file (and will build/extend it itself if missing), but on a shared cluster it
is safer to pre-build it once with this script before launching several
ensemble-scoring jobs in parallel, so they don't race to write it concurrently.

Usage:
    ../v/bin/python3.11 ./python/buildSentencePerplexityMap.py [CLASSES] [PERPS_DIR] [OUT_PATH]
"""

import json
import os
import sys
from pathlib import Path

SELECTED_CLASSES = [c.strip() for c in sys.argv[1].split(",") if c.strip()] \
    if len(sys.argv) > 1 else ["good", "bad", "broken", "corrupt", "corruptTriple",
                               "candidateCorrupt", "deprefGood", "deprefBroken"]
PERPS_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("resources/output30/perps")
OUT_PATH = Path(sys.argv[3]) if len(sys.argv) > 3 else PERPS_DIR / "sentenceMap.json"

MEASURERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]
EXCLUDED_GENERATORS = ["arnir", "liquidai"]


def readPerpsRaw(path):
    """Return the raw perplexities from the third line (nan/inf kept in place)."""
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        return None
    line = lines[2].strip()
    if not (line.startswith("[") and line.endswith("]")):
        return None
    try:
        return [float(v) for v in line[1:-1].split(",") if v.strip()]
    except ValueError:
        return None


def buildClass(cls):
    """{generator_stem: {measurer: [perp, ...]}} for one class."""
    byGenerator = {}
    for measurer in MEASURERS:
        d = PERPS_DIR / f"by{measurer}" / cls
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.txt")):
            stem = f.stem
            if stem.removeprefix("of") in EXCLUDED_GENERATORS:
                continue
            values = readPerpsRaw(f)
            if values is None:
                print(f"skipping invalid file: {f}", file=sys.stderr)
                continue
            byGenerator.setdefault(stem, {})[measurer] = values
    return byGenerator


def writeAtomic(data, path):
    """Write JSON via a same-directory temp file + rename, so a job reading
    the cache never sees a half-written file from a concurrent build."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def main():
    # Extend rather than replace: the class list is a request to REFRESH those
    # classes, not a statement that they are the only ones anyone cares about.
    # Rebuilding with a short list used to drop every class not named, which
    # is silent -- the scoring scripts treat a missing class as one to rebuild
    # on the fly, so the run still works and only the wasted rescan (and, on a
    # class the builder's list never had, an empty source-id map behind it)
    # shows that anything happened.
    data = {}
    if OUT_PATH.exists():
        with OUT_PATH.open() as fh:
            data = json.load(fh)
        print(f"extending {OUT_PATH} ({', '.join(sorted(data)) or 'empty'})")

    for cls in SELECTED_CLASSES:
        byGenerator = buildClass(cls)
        data[cls] = byGenerator
        n_pairs = sum(len(v) for v in byGenerator.values())
        print(f"{cls}: {len(byGenerator)} generators, {n_pairs} (generator,measurer) arrays")

    writeAtomic(data, OUT_PATH)
    print(f"wrote {OUT_PATH} ({', '.join(sorted(data))})")


if __name__ == "__main__":
    main()
