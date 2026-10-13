"""Score every *ensemble subset* of a fixed measurer set via subsampling,
with per-measurer z-score normalization.

Like `scoreSeparationSampledEnsemble.py`, but each measurer's perplexities are
first z-score normalized: for every measurer we compute the mean and standard
deviation over *all* of its finite perplexities (all read classes and all
generator files pooled together — good and bad alike, so no label information
leaks into the transform) and map each value to (v - mean) / std. Because
z-scores are signed, the per-sentence ensemble score of a subset is the **sum**
of that subset's measurers' z-scores (the signed analog of the product used in
the unnormalized script) rather than their product.

That combined score is then treated exactly as before: for each of
`sample_times` repetitions we draw `sample_size` good and `sample_size` bad
combined scores and compute AUROC on the draw, then report the mean AUROC and a
90% confidence interval (5th/95th percentiles of the subsample distribution).

Because the sum is per sentence, the measurers must be aligned position by
position. We therefore only pool generator files present for *every* measurer in
the subset, align them by index, and keep a position only when it is finite for
every measurer (a nan/inf in any measurer drops that sentence). Subsets whose
measurers share no data (e.g. any subset containing a measurer with no perp
files) come out all "--".

`classes` is a comma-separated list of bad-ish classes to evaluate (e.g.
"bad,broken"); the output has one column per listed class plus a "combined"
column pooling exactly those listed classes. `good` is always the negative class.

AUROC is the probability that a random bad sentence scores higher than a random
good one:

    0.5 -> no distinguishing power
    1.0 -> perfect separation
   <0.5 -> the measurer ranks good as *worse* than bad (inverted)

Usage:
    ../v/bin/python3.11 ./python/scoreSeparationSampledEnsembleNormalized.py SAMPLE_SIZE SAMPLE_TIMES [CLASSES] [PERPS_DIR] [SEED]
"""

import csv
import sys
from itertools import combinations
from pathlib import Path
import datetime

import numpy as np
from scipy.stats import mannwhitneyu

SAMPLE_SIZE = int(sys.argv[1])
SAMPLE_TIMES = int(sys.argv[2])
SELECTED_CLASSES = [c.strip() for c in sys.argv[3].split(",") if c.strip()] \
    if len(sys.argv) > 3 else ["bad", "broken", "corrupt", "corruptTriple", "candidateCorrupt", "deprefBroken"]
PERPS_DIR = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("resources/output30/perps")
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else None
CSV_PATH = Path(sys.argv[6]) if len(sys.argv) > 6 \
    else PERPS_DIR.parent / f"ensemble_auroc_norm_ss{SAMPLE_SIZE}_st{SAMPLE_TIMES}_{str(datetime.datetime.now())}.csv"

# the fixed universe of measurers; every non-empty subset gets one output row
MEASURERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]

# classes whose files we actually need to read: good (negative) + selected
READ_CLASSES = ["good"] + [c for c in SELECTED_CLASSES if c != "good"]
EXCLUDED_GENERATORS = ["arnir", "liquidai"]
CI = 90  # confidence interval width in percent

# (measurer, cls) -> {generator_stem: np.array}; each file read at most once
_rawCache = {}
# measurer -> (mean, std) over all its finite perplexities across READ_CLASSES
_statsCache = {}


def readPerpsRaw(path):
    """Return the raw perplexities from the third line (nan/inf kept in place).

    Positions must be preserved for cross-measurer alignment, so unlike
    `scoreSeparationSampled.py` we do NOT drop non-finite values here; they are
    filtered later, after the measurers are aligned by index."""
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


def measurerStats(measurer):
    """(mean, std) over all the measurer's finite perplexities, all classes.

    The transform must not depend on the subset being scored or on the class
    label, so the statistics pool every finite value the measurer produced
    across READ_CLASSES. A degenerate std (0 or no data) falls back to 1 so the
    transform stays defined; such measurers carry no signal anyway."""
    if measurer not in _statsCache:
        values = np.concatenate(
            [a for cls in READ_CLASSES for a in rawByStem(measurer, cls).values()]
            or [np.empty(0)])
        values = values[np.isfinite(values)]
        if len(values) == 0:
            _statsCache[measurer] = (0.0, 1.0)
        else:
            std = values.std()
            _statsCache[measurer] = (values.mean(), std if std > 0 else 1.0)
    return _statsCache[measurer]


def ensemblePool(subset, cls):
    """Pooled per-sentence sum of the subset's z-scored perplexities for `cls`.

    Only generator files present for every measurer in the subset are used;
    within each file the measurers are aligned by index (truncated to the
    shortest), and a position contributes its sum only if finite for all."""
    perMeasurer = [rawByStem(m, cls) for m in subset]
    common = set(perMeasurer[0])
    for d in perMeasurer[1:]:
        common &= set(d)

    stats = [measurerStats(m) for m in subset]
    means = np.array([s[0] for s in stats])[:, None]
    stds = np.array([s[1] for s in stats])[:, None]

    pooled = []
    for stem in sorted(common):
        arrays = [d[stem] for d in perMeasurer]
        length = min(len(a) for a in arrays)
        stacked = np.vstack([a[:length] for a in arrays])  # (measurers, length)
        finite = np.isfinite(stacked).all(axis=0)
        z = (stacked[:, finite] - means) / stds
        pooled.extend(z.sum(axis=0).tolist())
    return pooled


def auroc(good, bad):
    """Probability a random `bad` scores higher than a random `good`.

    Equivalent to the ROC AUC with good=negative, bad=positive. Ties count 0.5
    (mannwhitneyu's default tie handling gives exactly that)."""
    if not len(good) or not len(bad):
        return float("nan")
    u = mannwhitneyu(bad, good, alternative="two-sided").statistic
    return u / (len(bad) * len(good))


def sample(rng, pool):
    """Draw SAMPLE_SIZE values from pool (without replacement when possible)."""
    pool = np.asarray(pool)
    replace = True
    return rng.choice(pool, size=SAMPLE_SIZE, replace=replace)


def sampledAuroc(rng, good, bad):
    """Mean AUROC and 90% CI bounds over SAMPLE_TIMES subsamples.

    Returns (mean, lo, hi), all nan if either pool is empty."""
    if not good or not bad:
        return float("nan"), float("nan"), float("nan")
    scores = np.array([auroc(sample(rng, good), sample(rng, bad))
                       for _ in range(SAMPLE_TIMES)])
    tail = (100 - CI) / 2
    return scores.mean(), np.percentile(scores, tail), np.percentile(scores, 100 - tail)


def subsetStats(rng, subset, cols):
    """(mean, lo, hi) per column for one subset (nan-filled where no data)."""
    data = {cls: ensemblePool(subset, cls) for cls in READ_CLASSES}
    good = data.get("good", [])
    stats = []
    for cls in cols:
        if cls == "combined":
            bad = [v for c in SELECTED_CLASSES for v in data.get(c, [])]
        else:
            bad = data.get(cls, [])
        stats.append(sampledAuroc(rng, good, bad))
    return stats


def formatCells(stats):
    """Render one subset's per-column stats as fixed-width console cells."""
    cells = ""
    for mean, lo, hi in stats:
        if np.isfinite(mean):
            cells += f"{mean:>10.3f} [{lo:.3f},{hi:.3f}]"
        else:
            cells += f"{'--':>26}"
    return cells


def writeCsv(labels, cols, rows):
    """Write one row per subset; three columns (mean/lo/hi) per class column."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["ensemble"]
    for c in cols:
        header += [f"{c}_mean", f"{c}_lo", f"{c}_hi"]
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for label, stats in zip(labels, rows):
            row = [label]
            for mean, lo, hi in stats:
                # leave blank rather than 'nan' where a subset lacks data
                row += ["" if not np.isfinite(v) else f"{v:.6f}" for v in (mean, lo, hi)]
            writer.writerow(row)
    print(f"wrote {CSV_PATH}")


def main():
    rng = np.random.default_rng(SEED)
    # every non-empty subset, grouped by size (singletons first, full set last)
    subsets = [s for r in range(1, len(MEASURERS) + 1)
               for s in combinations(MEASURERS, r)]
    labels = ["*".join(s) for s in subsets]

    # one column per selected class, plus a "combined" column pooling them all
    # (omit "combined" when only a single class is selected — it would duplicate)
    cols = SELECTED_CLASSES + (["combined"] if len(SELECTED_CLASSES) > 1 else [])

    print(f"z-normalized ensemble subsets of {{{','.join(MEASURERS)}}}  "
          f"sample_size={SAMPLE_SIZE}  sample_times={SAMPLE_TIMES}  "
          f"classes={','.join(SELECTED_CLASSES)}  "
          f"CI={CI}%" + (f"  seed={SEED}" if SEED is not None else ""))
    width = max(16, max(len(l) for l in labels) + 2)
    header = f"{'ensemble':<{width}}" + "".join(f"{c + ' (mean [lo, hi])':>26}" for c in cols)
    print(header)
    print("-" * len(header))

    rows = []
    for subset, label in zip(subsets, labels):
        stats = subsetStats(rng, subset, cols)
        rows.append(stats)
        print(f"{label:<{width}}{formatCells(stats)}")

    writeCsv(labels, cols, rows)


if __name__ == "__main__":
    main()
