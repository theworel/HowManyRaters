"""Turn a merged cross-validation CSV into the pgfplots coordinate blocks of
the AUROC-versus-ensemble-size figures, each point carrying its fold spread.

Input is one merged run of
scoreSeparationSampledEnsembleNormalizedLogCVBySentence.py --- either the CSV
an undivided run wrote or the one mergeCVShards.py concatenated, both of which
have a row per subset with, per class column and split, the pooled mean, the
`CI` interval, the spread of the per-fold means (`_foldstd`) and every fold's
own mean (`_fold0`..`_foldN`).

For each class column and each ensemble size k it derives the three curves the
paper plots:

    best     the size-k subset with the highest TRAIN mean, scored on dev
    worst    the same by lowest train mean
    median   the median dev mean over all size-k subsets, with the quartiles
             that make the shaded band

and, for each of them, a fold spread to draw as an error bar. The rule is the
same for all three: the interval is the standard deviation, over the ten
folds, OF THE QUANTITY ACTUALLY PLOTTED. For best and worst that is one fixed
subset's ten per-fold dev means, so the number is its `_dev_foldstd` column
unchanged --- the same figure the best-per-size table prints as `sd`. For the
median it is the median RECOMPUTED WITHIN EACH FOLD and then varied over
folds: the subset sitting at the median may differ from fold to fold, but the
statistic on the page is the median, so the honest interval is how much that
statistic moves when the held-out sentences change.

Note what this interval is not. It is one standard deviation of a fold-level
spread, not a standard error of the pooled mean (which would be smaller by
sqrt(10)) and not the `_lo`/`_hi` subsampling interval already in the CSV
(which is narrower still, because resampling scores does not resample
sentences). The fold spread is the honest one to show, because it is the
corpus --- 50 source sentences --- and not the subsampling that limits what
these curves can distinguish.

Each curve is emitted twice: once as a plain coordinate list, which is the
line the reader follows, and once as a whisker-only overlay carrying the
`+- (0,sd)` errors. They are separate plots because the overlay is SHIFTED
ALONG X. Two intervals that overlap in y and sit at the same x are drawn as
one continuous line and the reader cannot tell where either ends --- which is
exactly what happens to the best and worst curves past the middle sizes --- so
each overlay is nudged a fraction of a rater sideways. The lines and marks
stay on the integer ticks; only the whiskers move.

Usage:
    ../v/bin/python3.11 ./python/buildSizeCurveData.py [--dodge=CURVE:DX,...]
        MERGED_CSV [CLASS,...]

With no class list every column in the header is emitted. The default dodge
suits a three-curve figure (best left, median centred, worst right); a figure
that plots only best and median wants something like
`--dodge=best:-0.15,median:0.15`.
"""

import csv
import re
import sys
from pathlib import Path

ARGV = [a for a in sys.argv if not a.startswith("--")]
OPTIONS = [a for a in sys.argv if a.startswith("--")]
if len(ARGV) < 2:
    sys.exit("usage: buildSizeCurveData.py [--dodge=CURVE:DX,...] MERGED_CSV [CLASS,...]")

DODGE = {"best": -0.22, "median": 0.0, "worst": 0.22}
for option in OPTIONS:
    if not option.startswith("--dodge="):
        sys.exit(f"unknown option {option}")
    DODGE = {}
    for pair in option[len("--dodge="):].split(","):
        curve, _, dx = pair.partition(":")
        DODGE[curve.strip()] = float(dx)

CSV_PATH = Path(ARGV[1])
SELECTED = [c.strip() for c in ARGV[2].split(",") if c.strip()] if len(ARGV) > 2 else None


def sd(values):
    """Population sd, matching the scoring script's foldStd."""
    values = [v for v in values if v == v]
    if not values:
        return float("nan")
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def quantile(values, frac):
    """Linear-interpolation quantile of an unsorted list (numpy's default)."""
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = frac * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] * (1 - (pos - lo)) + v[hi] * (pos - lo)


def readRows(path):
    """(rows, columns, nFolds); columns and fold count come from the header so
    this stays in step with whatever classes the run scored."""
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} has no rows")
    header = list(rows[0].keys())
    cols = [n[: -len("_train_mean")] for n in header if n.endswith("_train_mean")]
    # `_train_foldstd` shares the prefix of `_train_fold0`, so match the digits
    nFolds = sum(1 for n in header
                 if re.match(rf"^{re.escape(cols[0])}_train_fold\d+$", n))
    if not cols or not nFolds:
        sys.exit("header has no <class>_train_mean / _train_foldN columns")
    return rows, cols, nFolds


def cell(row, name):
    text = row[name]
    return float(text) if text else float("nan")


def curves(rows, cls, nFolds):
    """[(k, point-per-curve)] for one class column, smallest size first.

    Each curve entry is (value, foldSd); `q25`/`q75` are the band and have no
    interval of their own.
    """
    bySize = {}
    for row in rows:
        bySize.setdefault(row["ensemble"].count("*") + 1, []).append(row)

    out = []
    for k in sorted(bySize):
        group = bySize[k]
        devs = [cell(r, f"{cls}_dev_mean") for r in group]

        best = max(group, key=lambda r: cell(r, f"{cls}_train_mean"))
        worst = min(group, key=lambda r: cell(r, f"{cls}_train_mean"))

        # the median as each fold saw it, so its spread is the spread of the
        # plotted statistic rather than of whichever subset happens to sit at
        # the median of the pooled means
        foldMedians = [quantile([cell(r, f"{cls}_dev_fold{f}") for r in group], 0.5)
                       for f in range(nFolds)]

        out.append((k, {
            "median": (quantile(devs, 0.5), sd(foldMedians)),
            "q25": (quantile(devs, 0.25), float("nan")),
            "q75": (quantile(devs, 0.75), float("nan")),
            "best": (cell(best, f"{cls}_dev_mean"), cell(best, f"{cls}_dev_foldstd")),
            "worst": (cell(worst, f"{cls}_dev_mean"), cell(worst, f"{cls}_dev_foldstd")),
            "bestLabel": best["ensemble"],
            "worstLabel": worst["ensemble"],
        }))
    return out


def printTable(cls, points):
    print(f"\n=== {cls} ===")
    print(f"{'k':>3}{'median':>9}{'+-':>7}{'q25':>9}{'q75':>9}"
          f"{'best':>9}{'+-':>7}{'worst':>9}{'+-':>7}  best-on-train ensemble")
    for k, p in points:
        print(f"{k:>3}{p['median'][0]:>9.4f}{p['median'][1]:>7.4f}"
              f"{p['q25'][0]:>9.4f}{p['q75'][0]:>9.4f}"
              f"{p['best'][0]:>9.4f}{p['best'][1]:>7.4f}"
              f"{p['worst'][0]:>9.4f}{p['worst'][1]:>7.4f}  {p['bestLabel']}")


def coordinates(points, curve, withError, perLine=5, dx=0.0):
    """A pgfplots `coordinates {...}` body, wrapped like the ones in the paper.

    `+- (0,e)` is the explicit y error of a point and needs the plot to be in
    `error bars/y explicit` mode; without `withError` the block is a plain
    coordinate list, which is what the curve itself and the quartile paths
    bounding the shaded band want. `dx` shifts every x, for the whisker
    overlays.
    """
    parts = []
    for k, p in points:
        value, error = p[curve]
        x = f"{k + dx:g}"
        text = f"({x},{value:.4f})"
        if withError:
            text += f" +- (0,{0.0 if error != error else error:.4f})"
        parts.append(text)
    lines = [" ".join(parts[i:i + perLine]) for i in range(0, len(parts), perLine)]
    body = ("\n" + " " * 15).join(lines)
    return "  coordinates {" + body + "};"


def printBlocks(cls, points):
    print(f"\n--- {cls}: pgfplots blocks ---")
    for curve in ("q25", "q75", "median", "best", "worst"):
        print(f"% {curve}")
        print(coordinates(points, curve, False))
    for curve, dx in DODGE.items():
        print(f"% {curve} whiskers, x{dx:+g}")
        # a point with its error is about twice as wide, so half as many fit
        print(coordinates(points, curve, True, 3, dx))


def main():
    rows, cols, nFolds = readRows(CSV_PATH)
    wanted = SELECTED if SELECTED is not None else cols
    missing = [c for c in wanted if c not in cols]
    if missing:
        sys.exit(f"{CSV_PATH} has no column(s) {','.join(missing)}; it has {','.join(cols)}")
    print(f"{CSV_PATH}: {len(rows)} subsets, {nFolds} folds, columns {','.join(cols)}")
    for cls in wanted:
        points = curves(rows, cls, nFolds)
        printTable(cls, points)
        printBlocks(cls, points)


if __name__ == "__main__":
    main()
