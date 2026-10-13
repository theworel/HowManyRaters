"""Count surviving sentences per ensemble subset and class.

Companion to `scoreSeparationSampledEnsemble.py`: it reads *exactly the same
data* (same measurer universe, same perps directory, same generator exclusions,
same per-sentence alignment/filtering), but instead of scoring separation it
just reports, for every non-empty subset of the fixed measurer set, HOW MANY
sentences survive per class.

A sentence survives for a subset/class only if it is present and finite for
*every* measurer in the subset -- i.e. it is the count that
`scoreSeparationSampledEnsemble.ensemblePool` would return after discarding:
  * generator files missing for some measurer in the subset,
  * positions past the shortest measurer's length (index misalignment),
  * any position where at least one measurer returned nan/inf.

So this answers "how many did each ensemble NOT have to discard because a rater
returned nan or nothing at all". `good` is shown alongside the selected classes,
plus a `combined` column pooling exactly the selected (bad-ish) classes.

Usage:
    ../v/bin/python3.11 ./python/ensembleCountsSampledEnsemble.py [CLASSES] [PERPS_DIR] [CSV_PATH]
"""

import csv
import sys
from itertools import combinations
from pathlib import Path
import datetime

import numpy as np

SELECTED_CLASSES = [c.strip() for c in sys.argv[1].split(",") if c.strip()] \
    if len(sys.argv) > 1 else ["bad", "broken", "corrupt", "corruptTriple", "candidateCorrupt", "deprefBroken"]
PERPS_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("resources/output30/perps")
CSV_PATH = Path(sys.argv[3]) if len(sys.argv) > 3 \
    else PERPS_DIR.parent / f"ensemble_counts_{str(datetime.datetime.now())}.csv"

# the fixed universe of measurers; every non-empty subset gets one output row
MEASURERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]

# classes whose files we actually need to read: good (negative) + selected
READ_CLASSES = ["good"] + [c for c in SELECTED_CLASSES if c != "good"]
EXCLUDED_GENERATORS = ["arnir", "liquidai"]

# (measurer, cls) -> {generator_stem: np.array}; each file read at most once
_rawCache = {}


def readPerpsRaw(path):
    """Return the raw perplexities from the third line (nan/inf kept in place).

    Positions must be preserved for cross-measurer alignment, so we do NOT drop
    non-finite values here; they are filtered later, after alignment by index."""
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


def rawByStem(measurer, cls):
    """{generator_stem: np.array} for one measurer/class (positions preserved)."""
    key = (measurer, cls)
    if key not in _rawCache:
        out = {}
        for f in sorted((PERPS_DIR / f"by{measurer}" / cls).glob("*.txt")):
            if f.stem.removeprefix("of") in EXCLUDED_GENERATORS:
                continue
            values = readPerpsRaw(f)
            if values is None:
                print(f"skipping invalid file: {f}", file=sys.stderr)
                continue
            out[f.stem] = np.asarray(values, dtype=float)
        _rawCache[key] = out
    return _rawCache[key]


def ensembleCount(subset, cls):
    """Number of sentences that survive for `subset`/`cls`.

    Mirrors `scoreSeparationSampledEnsemble.ensemblePool` but counts instead of
    building the product pool: only generator files present for every measurer
    are used, measurers are aligned by index (truncated to the shortest), and a
    position is counted only when finite for all measurers."""
    perMeasurer = [rawByStem(m, cls) for m in subset]
    common = set(perMeasurer[0])
    for d in perMeasurer[1:]:
        common &= set(d)

    total = 0
    for stem in sorted(common):
        arrays = [d[stem] for d in perMeasurer]
        length = min(len(a) for a in arrays)
        stacked = np.vstack([a[:length] for a in arrays])  # (measurers, length)
        finite = np.isfinite(stacked).all(axis=0)
        total += int(finite.sum())
    return total


def subsetCounts(subset, cols):
    """Surviving-sentence count per column for one subset."""
    data = {cls: ensembleCount(subset, cls) for cls in READ_CLASSES}
    counts = []
    for cls in cols:
        if cls == "combined":
            counts.append(sum(data.get(c, 0) for c in SELECTED_CLASSES))
        else:
            counts.append(data.get(cls, 0))
    return counts


def writeCsv(labels, cols, rows):
    """Write one row per subset; one count column per class."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["ensemble"] + list(cols)
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for label, counts in zip(labels, rows):
            writer.writerow([label] + list(counts))
    print(f"wrote {CSV_PATH}")


def main():
    # every non-empty subset, grouped by size (singletons first, full set last)
    subsets = [s for r in range(1, len(MEASURERS) + 1)
               for s in combinations(MEASURERS, r)]
    labels = ["*".join(s) for s in subsets]

    # good is always shown; then each selected class, plus a "combined" column
    # pooling exactly the selected classes (omitted when a single class selected)
    cols = ["good"] + SELECTED_CLASSES \
        + (["combined"] if len(SELECTED_CLASSES) > 1 else [])

    print(f"ensemble subsets of {{{','.join(MEASURERS)}}}  "
          f"classes={','.join(SELECTED_CLASSES)}  (surviving-sentence counts)")
    width = max(16, max(len(l) for l in labels) + 2)
    header = f"{'ensemble':<{width}}" + "".join(f"{c:>16}" for c in cols)
    print(header)
    print("-" * len(header))

    rows = []
    for subset, label in zip(subsets, labels):
        counts = subsetCounts(subset, cols)
        rows.append(counts)
        print(f"{label:<{width}}" + "".join(f"{n:>16}" for n in counts))

    writeCsv(labels, cols, rows)


if __name__ == "__main__":
    main()
