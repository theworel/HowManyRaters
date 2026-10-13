"""Find example sentences where one rater disagrees with the other judges.

For every class, all sentences from the generator files common to every
measurer are pooled (aligned by index, positions finite for all measurers,
exactly like the ensemble scripts). Each measurer's perplexities are converted
to within-class percentile ranks in [0, 1] --- rank-based, so raw, log, and
z-scored perplexities all give the same result. A sentence's disagreement
score is

    d = target's percentile - mean percentile of the other measurers

so d > 0 means the target finds the sentence far less probable than the
consensus does, and d < 0 far more probable. The TOP_N sentences in each
direction are printed per class, with the answer text pulled from the
corresponding data file (`<cls>/<generator>.txt`, unit i of the
"-----------------------------"-separated format).

Usage:
    ../v/bin/python3.11 ./python/raterDisagreementExamples.py [TARGET] [CLASSES] [PERPS_DIR] [TOP_N]
"""

import sys
from pathlib import Path

import numpy as np

TARGET = sys.argv[1] if len(sys.argv) > 1 else "gptoss"
SELECTED_CLASSES = [c.strip() for c in sys.argv[2].split(",") if c.strip()] \
    if len(sys.argv) > 2 else ["bad", "broken", "corrupt", "corruptTriple", "candidateCorrupt", "deprefBroken"]
PERPS_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("resources/output30/perps")
TOP_N = int(sys.argv[4]) if len(sys.argv) > 4 else 8

# sentence texts live next to the perps: <DATA_DIR>/<cls>/<generator>.txt
DATA_DIR = PERPS_DIR.parent

MEASURERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]

CLASSES = ["good"] + [c for c in SELECTED_CLASSES if c != "good"]
EXCLUDED_GENERATORS = ["arnir", "liquidai"]

if TARGET not in MEASURERS:
    sys.exit(f"unknown target rater {TARGET!r}; pick one of {','.join(MEASURERS)}")
OTHERS = [m for m in MEASURERS if m != TARGET]

_rawCache = {}
_unitsCache = {}


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


def getUnits(path):
    """The '-----------------------------'-separated units of a data file,
    parsed exactly like calculatePerplexity.py's getUnits()."""
    if path not in _unitsCache:
        units, unit = [], ""
        for line in path.read_text().splitlines(keepends=True):
            if line.strip() == "-----------------------------":
                units.append(unit)
                unit = ""
                continue
            unit += line
        units.append(unit)
        _unitsCache[path] = units
    return _unitsCache[path]


def extractAnswer(unit):
    """The model output after the last 'answer:' line, joined to one line."""
    lines = unit.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "answer:":
            return " ".join(l.strip() for l in lines[i + 1:] if l.strip())
    return " ".join(unit.split())


def sentenceText(cls, stem, index):
    """Answer text of unit `index` in <DATA_DIR>/<cls>/<generator>.txt."""
    path = DATA_DIR / cls / f"{stem.removeprefix('of')}.txt"
    if not path.exists():
        return f"<{path} not found>"
    units = getUnits(path)
    if index >= len(units):
        return f"<unit {index} out of range in {path}>"
    return extractAnswer(units[index])


def rankPercentiles(values):
    """Average-tie ranks of `values` rescaled to percentiles in [0, 1]."""
    order = np.argsort(values, kind="mergesort")
    sorted_ = values[order]
    ranks = np.empty(len(values))
    i = 0
    while i < len(sorted_):
        j = i
        while j + 1 < len(sorted_) and sorted_[j + 1] == sorted_[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks / max(len(values) - 1, 1)


def pooledMatrix(cls):
    """(matrix, origins) for one class: perplexities of every measurer over
    the pooled common sentences, plus each column's (stem, original index).

    Matrix rows follow MEASURERS; only positions finite for all are kept."""
    perMeasurer = [rawByStem(m, cls) for m in MEASURERS]
    common = set(perMeasurer[0])
    for d in perMeasurer[1:]:
        common &= set(d)

    columns, origins = [], []
    for stem in sorted(common):
        arrays = [d[stem] for d in perMeasurer]
        length = min(len(a) for a in arrays)
        stacked = np.vstack([a[:length] for a in arrays])
        finite = np.isfinite(stacked).all(axis=0)
        columns.append(stacked[:, finite])
        origins.extend((stem, int(i)) for i in np.flatnonzero(finite))
    if not columns:
        return np.empty((len(MEASURERS), 0)), []
    return np.hstack(columns), origins


def printExamples(cls, matrix, origins, picks, pct, direction):
    ti = MEASURERS.index(TARGET)
    oi = [MEASURERS.index(m) for m in OTHERS]
    print(f"\n--- {cls}: {direction} ---")
    for rank, s in enumerate(picks, 1):
        stem, idx = origins[s]
        d = pct[ti, s] - pct[oi, s].mean()
        print(f"{rank:>3}. d={d:+.2f}  {TARGET} ppl={matrix[ti, s]:.3g} pct={pct[ti, s]:.2f}"
              f" | others pct mean={pct[oi, s].mean():.2f}"
              f" [{pct[oi, s].min():.2f}..{pct[oi, s].max():.2f}]"
              f" | {stem.removeprefix('of')}.txt #{idx}")
        print(f"     {sentenceText(cls, stem, idx)}")


def main():
    print(f"sentences where {TARGET} disagrees with the other {len(OTHERS)} raters  "
          f"classes={','.join(CLASSES)}  top_n={TOP_N}")
    print("percentiles are within-class ranks over the pooled sentences: "
          "0 = most probable, 1 = least probable")

    ti = MEASURERS.index(TARGET)
    oi = [MEASURERS.index(m) for m in OTHERS]
    for cls in CLASSES:
        matrix, origins = pooledMatrix(cls)
        if not origins:
            print(f"\n--- {cls}: no data shared by all measurers ---")
            continue
        pct = np.vstack([rankPercentiles(row) for row in matrix])
        d = pct[ti] - pct[oi].mean(axis=0)

        print(f"\n=== {cls}  ({len(origins)} pooled sentences, "
              f"mean |d|={np.abs(d).mean():.2f}) ===")
        highs = np.argsort(d)[::-1][:TOP_N]
        lows = np.argsort(d)[:TOP_N]
        printExamples(cls, matrix, origins, highs, pct,
                      f"{TARGET} finds these far LESS probable than the others do (d>0)")
        printExamples(cls, matrix, origins, lows, pct,
                      f"{TARGET} finds these far MORE probable than the others do (d<0)")


if __name__ == "__main__":
    main()
