"""Pairwise Spearman rank correlation between all raters, one matrix per class.

For every pair of measurers this aligns their per-sentence perplexities the
same way the ensemble scripts do — only generator files present for both
measurers, aligned by index (truncated to the shortest), keeping a position
only when it is finite for both — and computes Spearman's rank correlation
over the pooled aligned sentences.

The correlation is computed **within each class separately** (one matrix for
"good", one per selected bad-ish class). Pooling classes would inject the
shared good/bad signal every competent rater carries and inflate all pairwise
correlations toward 1; within-class correlation instead measures whether two
raters make correlated errors on the same sentences, which is what predicts
whether they ensemble well. An "overall" matrix averages the per-class
correlations weighted by each pair's aligned sentence count per class.

Spearman is rank-based, so the matrices are identical for raw, log, or
z-scored perplexities.

The CSV holds one block of 13 rows per class (plus "overall"), tagged by a
`matrix` column: "rho" blocks hold the correlations, "n" blocks the number of
aligned sentences behind each cell.

Usage:
    ../v/bin/python3.11 ./python/raterCorrelationMatrix.py [CLASSES] [PERPS_DIR] [CSV_PATH]
"""

import csv
import sys
from pathlib import Path
import datetime

import numpy as np
from scipy.stats import spearmanr

SELECTED_CLASSES = [c.strip() for c in sys.argv[1].split(",") if c.strip()] \
    if len(sys.argv) > 1 else ["broken", "corruptTriple", "candidateCorrupt", "deprefBroken"]
PERPS_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("resources/output30/perps")
CSV_PATH = Path(sys.argv[3]) if len(sys.argv) > 3 \
    else PERPS_DIR.parent / f"rater_spearman_{str(datetime.datetime.now())}.csv"

# the fixed universe of raters; every matrix is len(MEASURERS) square
MEASURERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]

# good gets its own matrix alongside every selected bad-ish class
CLASSES = ["good"] + [c for c in SELECTED_CLASSES if c != "good"]
EXCLUDED_GENERATORS = ["arnir", "liquidai"]

# (measurer, cls) -> {generator_stem: np.array}; each file read at most once
_rawCache = {}


def readPerpsRaw(path):
    """Return the raw perplexities from the third line (nan/inf kept in place).

    Positions must be preserved for cross-measurer alignment, so non-finite
    values are NOT dropped here; they are filtered after the pair is aligned."""
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


def alignedPair(m1, m2, cls):
    """Two aligned per-sentence arrays for a rater pair, finite in both.

    Pools generator files common to both raters, index-aligned per file."""
    d1, d2 = rawByStem(m1, cls), rawByStem(m2, cls)
    xs, ys = [], []
    for stem in sorted(set(d1) & set(d2)):
        a, b = d1[stem], d2[stem]
        length = min(len(a), len(b))
        a, b = a[:length], b[:length]
        keep = np.isfinite(a) & np.isfinite(b)
        xs.append(a[keep])
        ys.append(b[keep])
    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


def spearman(x, y):
    """Spearman's rho, nan when undefined (too few points or constant input)."""
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return spearmanr(x, y).correlation


def classMatrices(cls):
    """(rho, n) square matrices for one class.

    The diagonal is 1 wherever the rater has any data (n = its finite count)."""
    size = len(MEASURERS)
    rho = np.full((size, size), float("nan"))
    n = np.zeros((size, size), dtype=int)
    for i, m1 in enumerate(MEASURERS):
        for j in range(i, size):
            x, y = alignedPair(m1, MEASURERS[j], cls)
            n[i, j] = n[j, i] = len(x)
            if i == j:
                r = 1.0 if len(x) else float("nan")
            else:
                r = spearman(x, y)
            rho[i, j] = rho[j, i] = r
    return rho, n


def overallMatrices(perClass):
    """Average of per-class rho weighted by each cell's n; n sums the classes."""
    size = len(MEASURERS)
    num = np.zeros((size, size))
    den = np.zeros((size, size))
    total = np.zeros((size, size), dtype=int)
    for rho, n in perClass.values():
        valid = np.isfinite(rho)
        num[valid] += (rho * n)[valid]
        den[valid] += n[valid]
        total += n
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, float("nan"))
    return rho, total


def printMatrix(title, rho, n):
    """Render one matrix as a labeled console table."""
    width = max(len(m) for m in MEASURERS) + 2
    off = n[~np.eye(len(MEASURERS), dtype=bool)]
    print(f"\n{title}  (aligned sentences per pair: "
          f"min={off.min()}, max={off.max()})")
    print(" " * width + "".join(f"{m:>{width}}" for m in MEASURERS))
    for i, m in enumerate(MEASURERS):
        cells = "".join(f"{v:>{width}.3f}" if np.isfinite(v) else f"{'--':>{width}}"
                        for v in rho[i])
        print(f"{m:<{width}}{cells}")


def writeCsv(matrices):
    """One 13-row block per (class, rho|n), tagged by class and matrix columns."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "matrix", "rater"] + MEASURERS)
        for cls, (rho, n) in matrices.items():
            for i, m in enumerate(MEASURERS):
                writer.writerow([cls, "rho", m] +
                                ["" if not np.isfinite(v) else f"{v:.6f}" for v in rho[i]])
            for i, m in enumerate(MEASURERS):
                writer.writerow([cls, "n", m] + [str(v) for v in n[i]])
    print(f"\nwrote {CSV_PATH}")


def main():
    print(f"pairwise Spearman correlation of raters {{{','.join(MEASURERS)}}}  "
          f"classes={','.join(CLASSES)}")

    perClass = {cls: classMatrices(cls) for cls in CLASSES}
    matrices = dict(perClass)
    matrices["overall"] = overallMatrices(perClass)

    for cls, (rho, n) in matrices.items():
        printMatrix(cls, rho, n)

    writeCsv(matrices)


if __name__ == "__main__":
    main()
