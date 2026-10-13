"""Ensemble scoring with z-scored log-perplexities and an 80:20 train/dev split.

Like `scoreSeparationSampledEnsembleNormalizedLog.py`, but every sentence is
deterministically assigned to a TRAIN (80%) or DEV (20%) split, and each
ensemble subset is scored on both. The point is a generalization check: the
train split picks the best ensemble (per class column and per size), and the
dev split then shows whether that choice holds up on sentences that played no
part in selecting it.

The split is a pure function of (class, generator file, sentence index) and the
seed, so all measurers agree on which sentences are train and which are dev, no
matter which subset is being scored. Each measurer's z-score statistics (mean
and std of its finite log-perplexities pooled over all read classes) are
estimated on the TRAIN sentences only and applied unchanged to both splits, so
no information from dev leaks into the transform.

Everything else matches the parent script: the per-sentence ensemble score of a
subset is the sum of its measurers' z-scored log-perplexities; measurers are
aligned position by position over generator files present for every measurer in
the subset, keeping a position only when finite for all; AUROC is estimated by
drawing `sample_size` good and `sample_size` bad scores `sample_times` times
(mean and 90% CI over the draws), within each split separately.

The console prints one compact `train->dev` mean per column for every subset,
followed by a summary of the best-on-train subset per size with its dev score.
The CSV has the full mean/lo/hi for both splits and every subset.

Usage:
    ../v/bin/python3.11 ./python/scoreSeparationSampledEnsembleNormalizedLogSplit.py SAMPLE_SIZE SAMPLE_TIMES [CLASSES] [PERPS_DIR] [SEED]
"""

import csv
import hashlib
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
    else PERPS_DIR.parent / f"ensemble_auroc_lognorm_split_ss{SAMPLE_SIZE}_st{SAMPLE_TIMES}_{str(datetime.datetime.now())}.csv"

MEASURERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]

READ_CLASSES = ["good"] + [c for c in SELECTED_CLASSES if c != "good"]
EXCLUDED_GENERATORS = ["arnir", "liquidai"]
CI = 90  # confidence interval width in percent
TRAIN_FRAC = 0.8

# the split must be reproducible when SEED is given but still consistent
# within a run when it is not
SPLIT_SEED = SEED if SEED is not None else int(np.random.SeedSequence().entropy) % 2**32

_rawCache = {}
_statsCache = {}
_maskCache = {}


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
    """{generator_stem: np.array of log-perps} for one measurer/class.

    Positions are preserved; the log of a non-positive or non-finite perplexity
    is non-finite and gets dropped by the downstream finite filter."""
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
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f.stem] = np.log(np.asarray(values, dtype=float))
        _rawCache[key] = out
    return _rawCache[key]


def splitMask(cls, stem, length):
    """Boolean train mask for the first `length` positions of (cls, stem).

    A pure function of (SPLIT_SEED, cls, stem, index): the mask is drawn from
    an rng seeded by a hash of those, and prefixes are consistent, so
    measurers whose files differ in length still agree on shared positions."""
    key = (cls, stem)
    if key not in _maskCache or len(_maskCache[key]) < length:
        digest = hashlib.sha256(f"{SPLIT_SEED}:{cls}:{stem}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        _maskCache[key] = rng.random(length) < TRAIN_FRAC
    return _maskCache[key][:length]


def measurerStats(measurer):
    """(mean, std) of the measurer's finite log-perps over TRAIN sentences only.

    Estimated on the train split across all read classes and applied unchanged
    to both splits, so the transform never sees dev data. A degenerate std
    (0 or no data) falls back to 1 so the transform stays defined."""
    if measurer not in _statsCache:
        chunks = []
        for cls in READ_CLASSES:
            for stem, a in rawByStem(measurer, cls).items():
                chunks.append(a[splitMask(cls, stem, len(a))])
        values = np.concatenate(chunks or [np.empty(0)])
        values = values[np.isfinite(values)]
        if len(values) == 0:
            _statsCache[measurer] = (0.0, 1.0)
        else:
            std = values.std()
            _statsCache[measurer] = (values.mean(), std if std > 0 else 1.0)
    return _statsCache[measurer]


def ensemblePool(subset, cls, train):
    """Pooled per-sentence sums of z-scored log-perps for one split of `cls`.

    Only generator files present for every measurer in the subset are used;
    within each file the measurers are aligned by index (truncated to the
    shortest), and a position contributes only if it is finite for all
    measurers AND belongs to the requested split."""
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
        mask = splitMask(cls, stem, length)
        keep = np.isfinite(stacked).all(axis=0) & (mask if train else ~mask)
        z = (stacked[:, keep] - means) / stds
        pooled.extend(z.sum(axis=0).tolist())
    return pooled


def auroc(good, bad):
    """Probability a random `bad` scores higher than a random `good`."""
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
    """Mean AUROC and 90% CI bounds over SAMPLE_TIMES subsamples."""
    if not good or not bad:
        return float("nan"), float("nan"), float("nan")
    scores = np.array([auroc(sample(rng, good), sample(rng, bad))
                       for _ in range(SAMPLE_TIMES)])
    tail = (100 - CI) / 2
    return scores.mean(), np.percentile(scores, tail), np.percentile(scores, 100 - tail)


def subsetStats(rng, subset, cols):
    """{"train": [(mean, lo, hi) per column], "dev": [...]} for one subset."""
    out = {}
    for part, isTrain in (("train", True), ("dev", False)):
        data = {cls: ensemblePool(subset, cls, isTrain) for cls in READ_CLASSES}
        good = data.get("good", [])
        stats = []
        for cls in cols:
            if cls == "combined":
                bad = [v for c in SELECTED_CLASSES for v in data.get(c, [])]
            else:
                bad = data.get(cls, [])
            stats.append(sampledAuroc(rng, good, bad))
        out[part] = stats
    return out


def formatCells(stats):
    """Console cells: per column the train and dev means as 'train->dev'."""
    cells = ""
    for (tm, _, _), (dm, _, _) in zip(stats["train"], stats["dev"]):
        if np.isfinite(tm) or np.isfinite(dm):
            t = f"{tm:.3f}" if np.isfinite(tm) else "--"
            d = f"{dm:.3f}" if np.isfinite(dm) else "--"
            cells += f"{t + '->' + d:>16}"
        else:
            cells += f"{'--':>16}"
    return cells


def writeCsv(labels, cols, rows):
    """One row per subset; per class column, mean/lo/hi for both splits."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["ensemble"]
    for c in cols:
        for part in ("train", "dev"):
            header += [f"{c}_{part}_mean", f"{c}_{part}_lo", f"{c}_{part}_hi"]
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for label, stats in zip(labels, rows):
            row = [label]
            for ti, di in zip(stats["train"], stats["dev"]):
                for mean, lo, hi in (ti, di):
                    row += ["" if not np.isfinite(v) else f"{v:.6f}" for v in (mean, lo, hi)]
            writer.writerow(row)
    print(f"wrote {CSV_PATH}")


def fmtStat(stat):
    mean, lo, hi = stat
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]" if np.isfinite(mean) else "--"


def printBestPerSize(subsets, labels, rows, cols):
    """Per column: the subset with the best TRAIN mean at each size, its dev
    score, and the global train winner --- the generalization check."""
    width = max(16, max(len(l) for l in labels) + 2)
    for ci, cls in enumerate(cols):
        print(f"\nbest-on-train ensembles per size --- {cls}")
        print(f"{'k':>3} {'ensemble':<{width}}{'train (mean [lo,hi])':>24}{'dev (mean [lo,hi])':>24}")
        bestGlobal = None
        for k in range(1, len(MEASURERS) + 1):
            best = None
            for subset, label, stats in zip(subsets, labels, rows):
                if len(subset) != k or not np.isfinite(stats["train"][ci][0]):
                    continue
                if best is None or stats["train"][ci][0] > best[1]["train"][ci][0]:
                    best = (label, stats)
            if best is None:
                continue
            label, stats = best
            print(f"{k:>3} {label:<{width}}{fmtStat(stats['train'][ci]):>24}{fmtStat(stats['dev'][ci]):>24}")
            if bestGlobal is None or stats["train"][ci][0] > bestGlobal[1]["train"][ci][0]:
                bestGlobal = (label, stats)
        if bestGlobal is not None:
            label, stats = bestGlobal
            print(f"  best on train: {label}  "
                  f"train {fmtStat(stats['train'][ci])}  dev {fmtStat(stats['dev'][ci])}")


def main():
    rng = np.random.default_rng(SEED)
    subsets = [s for r in range(1, len(MEASURERS) + 1)
               for s in combinations(MEASURERS, r)]
    labels = ["*".join(s) for s in subsets]

    cols = SELECTED_CLASSES + (["combined"] if len(SELECTED_CLASSES) > 1 else [])

    print(f"log-z-normalized ensemble subsets of {{{','.join(MEASURERS)}}} "
          f"with a {int(TRAIN_FRAC * 100)}:{int(round((1 - TRAIN_FRAC) * 100))} train/dev split  "
          f"sample_size={SAMPLE_SIZE}  sample_times={SAMPLE_TIMES}  "
          f"classes={','.join(SELECTED_CLASSES)}  "
          f"CI={CI}%" + (f"  seed={SEED}" if SEED is not None else f"  split_seed={SPLIT_SEED}"))
    width = max(16, max(len(l) for l in labels) + 2)
    header = f"{'ensemble':<{width}}" + "".join(f"{c + ' (train->dev)':>16}" for c in cols)
    print(header)
    print("-" * len(header))

    rows = []
    for subset, label in zip(subsets, labels):
        stats = subsetStats(rng, subset, cols)
        rows.append(stats)
        print(f"{label:<{width}}{formatCells(stats)}")

    printBestPerSize(subsets, labels, rows, cols)
    writeCsv(labels, cols, rows)


if __name__ == "__main__":
    main()
