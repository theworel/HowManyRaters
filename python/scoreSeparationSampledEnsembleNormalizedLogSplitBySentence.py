"""Ensemble scoring with z-scored log-perplexities, train/dev split over the
SOURCE SENTENCES rather than over positions in the perplexity arrays.

`scoreSeparationSampledEnsembleNormalizedLogSplit.py` assigns each (class,
generator file, index) independently to train or dev, keyed by a hash of
(cls, generator stem). An earlier revision of this script keyed on (cls,
index) instead, on the assumption that index i is source sentence i in every
generator's file. That assumption is false. The pipeline generated several
candidate translations per source sentence and then dropped some of them, so
the class files run from 50 to 191 units and position i means a different
source sentence in each. With only 50 source sentences behind ~2,150 units
per class, splitting on positions puts every source sentence on both sides:
train and dev differ in which candidate translations they hold, not in which
content they are about.

This script splits on content. Every unit carries its source sentence in its
own prompt, so sourceSentenceIds.py labels each unit with a source id in
0..49 (see that module for how, and for the cache it writes); `N_DEV_SOURCES`
of the 50 sentences are drawn once and held out whole. A held-out sentence is
held out everywhere at once --- every generator's translations of it, all 13
measurers' scores of those, and everything derived from it in the other
classes --- so nothing in dev shares source content with anything in train.
Because the draw is over 50 items rather than over thousands, the split is
coarse, and dev estimates come out noisier than the position-level split's ---
that noise is the honest price of the leak it removes, not a regression.

Raw per-sentence perplexities are read from the cache built by
buildSentencePerplexityMap.py (`{cls: {generator_stem: {measurer: [perp,
...]}}}`) instead of re-globbing and re-parsing
PERPS_DIR/by<measurer>/<cls>/*.txt on every invocation; if the cache is
missing or missing a needed class, this script builds that part itself and
extends the cache file on disk, so repeat runs (e.g. sweeping sample sizes)
don't pay the parsing cost again. On a shared cluster, prefer pre-building the
caches once with buildSentencePerplexityMap.py and sourceSentenceIds.py before
launching several jobs in parallel, so they don't race to write them.

Everything else matches the parent split script: measurer z-score statistics
(mean/std of finite log-perplexities, pooled over all read classes) are
estimated on TRAIN sentences only and applied unchanged to both splits; the
per-sentence ensemble score of a subset is the sum of its measurers' z-scored
log-perplexities, aligned position by position over generator files common to
every measurer in the subset; AUROC is estimated by drawing `sample_size` good
and `sample_size` bad scores `sample_times` times (mean and 90% CI), within
each split separately.

The console prints one compact `train->dev` mean per column for every subset,
followed by a summary of the best-on-train subset per size with its dev
score. The CSV has the full mean/lo/hi for both splits and every subset.

Usage:
    ../v/bin/python3.11 ./python/scoreSeparationSampledEnsembleNormalizedLogSplitBySentence.py SAMPLE_SIZE SAMPLE_TIMES [CLASSES] [PERPS_DIR] [SEED] [MAP_PATH]
"""

import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path
import datetime

import numpy as np
from scipy.stats import mannwhitneyu

from sourceSentenceIds import devSourceIds, loadSourceIdMap, loadSources, unitCounts

SAMPLE_SIZE = int(sys.argv[1])
SAMPLE_TIMES = int(sys.argv[2])
SELECTED_CLASSES = [c.strip() for c in sys.argv[3].split(",") if c.strip()] \
    if len(sys.argv) > 3 else ["bad", "broken", "corrupt", "corruptTriple", "candidateCorrupt", "deprefBroken"]
PERPS_DIR = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("resources/output30/perps")
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else None
MAP_PATH = Path(sys.argv[6]) if len(sys.argv) > 6 else PERPS_DIR / "sentenceMap.json"
CSV_PATH = Path(sys.argv[7]) if len(sys.argv) > 7 \
    else PERPS_DIR.parent / f"ensemble_auroc_lognorm_splitbysource_ss{SAMPLE_SIZE}_st{SAMPLE_TIMES}_{str(datetime.datetime.now())}.csv"

MEASURERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]

READ_CLASSES = ["good"] + [c for c in SELECTED_CLASSES if c != "good"]
EXCLUDED_GENERATORS = ["arnir", "liquidai"]
CI = 90  # confidence interval width in percent

# the split: this many of the source sentences are held out whole, the rest
# are training content
N_DEV_SOURCES = 10
SOURCE_FILE = Path("resources/en.txt_unchanged")
SOURCE_MAP_PATH = PERPS_DIR / "sourceIdMap.json"
# where to look for the class files the units (and hence the source ids) come
# from; output20 is the fallback for stems measured in output30 from an
# earlier round's generations
UNIT_DIRS = [PERPS_DIR.parent, PERPS_DIR.parent.parent / "output20"]

SOURCES = loadSources(SOURCE_FILE)

# the split must be reproducible when SEED is given but still consistent
# within a run when it is not
SPLIT_SEED = SEED if SEED is not None else int(np.random.SeedSequence().entropy) % 2**32
DEV_SOURCES = devSourceIds(SPLIT_SEED, len(SOURCES), N_DEV_SOURCES)
DEV_SOURCE_ARRAY = np.array(sorted(DEV_SOURCES))

_rawCache = {}
_statsCache = {}
_assignCache = {}
_sentenceMap = None
_sourceIdMap = None


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


def buildSentenceMapClasses(classes):
    """{cls: {generator_stem: {measurer: [perp, ...]}}} for the given classes,
    read straight from PERPS_DIR (the same scan buildSentencePerplexityMap.py
    does)."""
    out = {}
    for cls in classes:
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
        out[cls] = byGenerator
    return out


def loadSentenceMap():
    """The cached {cls: {stem: {measurer: [perp,...]}}} map, extended on disk
    with any of READ_CLASSES it doesn't already cover.

    Written via a same-directory temp file + atomic rename so a concurrent
    reader never sees a half-written file."""
    global _sentenceMap
    if _sentenceMap is None:
        if MAP_PATH.exists():
            with MAP_PATH.open() as fh:
                _sentenceMap = json.load(fh)
        else:
            _sentenceMap = {}

    missing = [c for c in READ_CLASSES if not _sentenceMap.get(c)]
    if missing:
        print(f"sentence map cache missing {missing}, building from {PERPS_DIR} ...", file=sys.stderr)
        _sentenceMap.update(buildSentenceMapClasses(missing))
        MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MAP_PATH.with_suffix(MAP_PATH.suffix + f".tmp{os.getpid()}")
        with tmp.open("w") as fh:
            json.dump(_sentenceMap, fh)
        os.replace(tmp, MAP_PATH)
        print(f"wrote {MAP_PATH}", file=sys.stderr)
    return _sentenceMap


def rawByStem(measurer, cls):
    """{generator_stem: np.array of log-perps} for one measurer/class, from
    the cached sentence map. Position i is the i-th surviving unit of that
    generator's file, whose source sentence splitAssignment looks up."""
    key = (measurer, cls)
    if key not in _rawCache:
        out = {}
        for stem, byMeasurer in loadSentenceMap().get(cls, {}).items():
            values = byMeasurer.get(measurer)
            if values is None:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                out[stem] = np.log(np.asarray(values, dtype=float))
        _rawCache[key] = out
    return _rawCache[key]


def sourceIdMap():
    """{cls: {generator_stem: [source id, ...]}}, built and cached on first use."""
    global _sourceIdMap
    if _sourceIdMap is None:
        _sourceIdMap = loadSourceIdMap(unitCounts(loadSentenceMap(), READ_CLASSES),
                                       SOURCE_MAP_PATH, SOURCE_FILE, UNIT_DIRS)
    return _sourceIdMap


def splitAssignment(cls, stem, length):
    """Per-position split assignment for one generator file: 1 train, 0 dev,
    -1 unusable.

    A position's side is decided entirely by WHICH SOURCE SENTENCE it was
    generated from, so every candidate translation a generator produced for a
    held-out sentence, every measurer's score of them, and everything the
    other classes derived from that sentence all land in dev together.
    Positions whose source sentence could not be identified --- and whole
    files with no source ids at all, which is what happens when a stem has
    perplexities but no unit file to read them off --- are dropped from both
    splits rather than guessed at."""
    key = (cls, stem)
    if key not in _assignCache:
        ids = sourceIdMap().get(cls, {}).get(stem)
        if ids is None:
            _assignCache[key] = None
        else:
            ids = np.asarray(ids)
            _assignCache[key] = np.where(
                ids < 0, -1, np.where(np.isin(ids, DEV_SOURCE_ARRAY), 0, 1)).astype(np.int8)
    assign = _assignCache[key]
    if assign is None:
        return np.full(length, -1, dtype=np.int8)
    if len(assign) < length:
        # the map is built against these very arrays, so this means the cache
        # is stale; drop the unlabelled tail rather than mislabel it
        return np.concatenate([assign, np.full(length - len(assign), -1, dtype=np.int8)])
    return assign[:length]


def measurerStats(measurer):
    """(mean, std) of the measurer's finite log-perps over TRAIN sentences only.

    Estimated on the train split across all read classes and applied unchanged
    to both splits, so the transform never sees dev data. A degenerate std
    (0 or no data) falls back to 1 so the transform stays defined."""
    if measurer not in _statsCache:
        chunks = []
        for cls in READ_CLASSES:
            for stem, a in rawByStem(measurer, cls).items():
                chunks.append(a[splitAssignment(cls, stem, len(a)) == 1])
        values = np.concatenate(chunks or [np.empty(0)])
        values = values[np.isfinite(values)]
        if len(values) == 0:
            _statsCache[measurer] = (0.0, 1.0)
        else:
            std = values.std()
            _statsCache[measurer] = (values.mean(), std if std > 0 else 1.0)
    return _statsCache[measurer]


def ensemblePool(subset, cls, train):
    """Pooled per-unit sums of z-scored log-perps for one split of `cls`.

    Only generator files present for every measurer in the subset are used;
    within each file the measurers are aligned by index (truncated to the
    shortest), and a position contributes only if it is finite for all
    measurers AND its source sentence belongs to the requested split."""
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
        assign = splitAssignment(cls, stem, length)
        keep = np.isfinite(stacked).all(axis=0) & (assign == (1 if train else 0))
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


def printSplit():
    """The split itself: which source sentences are held out, and how many
    units of each class that leaves on each side."""
    nTrain = len(SOURCES) - N_DEV_SOURCES
    print(f"held-out source sentences ({N_DEV_SOURCES} of {len(SOURCES)}): "
          f"{sorted(DEV_SOURCES)}")
    print(f"{'class':<18}{'train':>10}{'dev':>10}{'dropped':>10}{'generators':>12}")
    for cls in READ_CLASSES:
        counts = [0, 0, 0]
        byStem = rawByStem(MEASURERS[0], cls)
        for stem, a in byStem.items():
            assign = splitAssignment(cls, stem, len(a))
            counts[0] += int((assign == 1).sum())
            counts[1] += int((assign == 0).sum())
            counts[2] += int((assign == -1).sum())
        print(f"{cls:<18}{counts[0]:>10}{counts[1]:>10}{counts[2]:>10}{len(byStem):>12}")
    print(f"(train content = {nTrain} source sentences, dev content = "
          f"{N_DEV_SOURCES}, disjoint)")


def main():
    rng = np.random.default_rng(SEED)
    subsets = [s for r in range(1, len(MEASURERS) + 1)
               for s in combinations(MEASURERS, r)]
    labels = ["*".join(s) for s in subsets]

    cols = SELECTED_CLASSES + (["combined"] if len(SELECTED_CLASSES) > 1 else [])

    print(f"log-z-normalized ensemble subsets of {{{','.join(MEASURERS)}}} "
          f"with a {len(SOURCES) - N_DEV_SOURCES}:{N_DEV_SOURCES} train/dev split "
          f"over the {len(SOURCES)} SOURCE SENTENCES  "
          f"sample_size={SAMPLE_SIZE}  sample_times={SAMPLE_TIMES}  "
          f"classes={','.join(SELECTED_CLASSES)}  "
          f"CI={CI}%" + (f"  seed={SEED}" if SEED is not None else f"  split_seed={SPLIT_SEED}"))
    printSplit()
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
