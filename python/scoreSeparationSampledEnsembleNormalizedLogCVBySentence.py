"""Ensemble scoring with z-scored log-perplexities and 10-FOLD CROSS
VALIDATION over the SOURCE SENTENCES.

This is `scoreSeparationSampledEnsembleNormalizedLogSplitBySentence.py` with
its single 40:10 holdout replaced by a rotation. Read that script first: the
reason the split is drawn over source sentences rather than over positions in
the perplexity arrays, how sourceSentenceIds.py labels each unit with the
source sentence its prompt was built from, and the caches this reads are all
explained there and unchanged here.

What changes is how much of the data the estimate rests on. There are only 50
source sentences, so one holdout puts 10 of them in dev and reports a dev
number computed from those 10 --- a coarse draw, and the sentences that
happened to fall in dev matter more than they should. Cross validation cuts
the same seeded permutation into `N_FOLDS` blocks of 5 sentences instead of a
head and a tail, and runs the whole experiment `N_FOLDS` times: each block is
dev exactly once and train the other nine times, so every source sentence
contributes to a dev estimate exactly once and every dev estimate is still
made against training content it shares nothing with.

Each fold is a full re-run, not a re-scoring: its measurer z-score statistics
(mean/std of finite log-perplexities, pooled over all read classes) are
estimated on ITS OWN train sentences and applied unchanged to both of its
splits, so no fold's transform has seen its own dev data. Within a fold,
scoring matches the parent script exactly --- the per-unit ensemble score of a
subset is the sum of its measurers' z-scored log-perplexities, aligned
position by position over generator files common to every measurer in the
subset, and AUROC is estimated by drawing `sample_size` good and `sample_size`
bad scores `sample_times` times.

The folds are then pooled: a column's reported figure is the mean over all
`N_FOLDS * sample_times` subsample AUROCs, and its interval covers subsampling
noise and fold-to-fold noise together. The per-fold means are kept as well ---
they go in the CSV, their spread is reported as `foldstd`, and they drive the
cross-validated selection table, which is the part a single holdout cannot
produce: for each size k it picks the best-on-train subset SEPARATELY IN EVERY
FOLD and averages what those picks then score on held-out sentences. That is
an estimate of the procedure "choose an ensemble on the data you have, use it
on sentences you have not seen", rather than of one particular ensemble.

Ten folds is ten times the scoring work of one holdout, so expect a long run
even though the AUROC statistic here is computed directly (see checkAuroc)
instead of through scipy's mannwhitneyu, which also computes a p-value this
script never uses.

`--good=<class>` picks the positive class and covers what the two separate
holdout scripts needed two files for. The default "good" is each model's own
translations; `--good=deprefGood` is the depref variant --- the human
reference translations of resources/output30/deprefGood/all.txt, 39 of them
per source sentence (8 for the one WMT11 sentence with fewer distinct
references). deprefGood needs no special handling beyond the name:
buildDeprefGood.py copies its prompts verbatim from a good-class file, so
sourceSentenceIds.py labels its 1,919 units like any other class's, and
holding out a source sentence holds out all of its references together as a
side effect. Worth remembering when reading that run's output: its positive
class is human-authored while its negative classes are model-authored, so it
measures whatever perplexity says about authorship at least as much as what
it says about quality.

`--exclude=a,b,c` replaces the default list of generators whose files are not
read at all (arnir, liquidai), for when the two runs want different ones.

`--shard=I/N` computes only every N-th subset of the enumeration, starting at
I, and writes those rows to their own CSV. Every row is a self-contained
experiment --- the folds, the z-score statistics and the subsampling of one
subset are all decided without reference to any other subset --- so N shards
can run as N independent jobs and mergeCVShards.py concatenates their CSVs
into exactly the table one undivided run would have produced. That matters
because the whole enumeration takes about four hours, which is the length of
the queue slot it has to fit in; a tenth of it is comfortable.

Two details make the split exact rather than merely parallel. The subsampling
RNG is seeded per subset (see rngFor) instead of being one stream walked in
enumeration order, so a row does not depend on how many rows preceded it.
And a shard takes every N-th subset rather than a contiguous block, so that
each one gets the same mix of subset sizes and therefore roughly the same
amount of work --- the enumeration is ordered by size, and blocks of it are
not comparable. Sharding requires an explicit SEED: without one each job would
draw its own folds, and the rows would not be from the same experiment.

Usage:
    ../v/bin/python3.11 ./python/scoreSeparationSampledEnsembleNormalizedLogCVBySentence.py [--good=CLASS] [--exclude=GENERATORS] [--shard=I/N] SAMPLE_SIZE SAMPLE_TIMES [CLASSES] [PERPS_DIR] [SEED] [MAP_PATH] [CSV_PATH]
"""

import csv
import json
import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import datetime

import numpy as np
from scipy.stats import mannwhitneyu

from sourceSentenceIds import foldSourceIds, loadSourceIdMap, loadSources, unitCounts

# the positive class is a flag rather than a positional argument, so that the
# depref variant of the run differs from the plain one by one readable word
# and every existing invocation keeps working unchanged
GOOD_CLASS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--good=")), "good")
# ...and so is the generator blacklist, because one script now serves the plain
# run and the depref one and they need not agree on which generators to read
EXCLUDE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--exclude=")), None)
# ...and so is the slice of the enumeration this process is responsible for
SHARD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), None)
FLAGS = ("--good=", "--exclude=", "--shard=")
ARGV = [a for a in sys.argv if not a.startswith(FLAGS)]

if SHARD is None:
    SHARD_INDEX, SHARD_COUNT = 0, 1
else:
    try:
        SHARD_INDEX, SHARD_COUNT = (int(x) for x in SHARD.split("/"))
    except ValueError:
        sys.exit(f"--shard={SHARD} is not of the form I/N")
    if not 0 <= SHARD_INDEX < SHARD_COUNT:
        sys.exit(f"--shard={SHARD}: need 0 <= I < N")

SAMPLE_SIZE = int(ARGV[1])
SAMPLE_TIMES = int(ARGV[2])
SELECTED_CLASSES = [c.strip() for c in ARGV[3].split(",") if c.strip()] \
    if len(ARGV) > 3 else ["bad", "broken", "corrupt", "corruptTriple", "candidateCorrupt", "deprefBroken"]
PERPS_DIR = Path(ARGV[4]) if len(ARGV) > 4 else Path("resources/output30/perps")
SEED = int(ARGV[5]) if len(ARGV) > 5 else None
MAP_PATH = Path(ARGV[6]) if len(ARGV) > 6 else PERPS_DIR / "sentenceMap.json"

MEASURERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]

READ_CLASSES = [GOOD_CLASS] + [c for c in SELECTED_CLASSES if c != GOOD_CLASS]
EXCLUDED_GENERATORS = [g.strip() for g in EXCLUDE.split(",") if g.strip()] \
    if EXCLUDE is not None else ["arnir", "liquidai"]
CI = 90  # confidence interval width in percent

# the rotation: the source sentences are cut into this many folds, each of
# which is the held-out set of exactly one round
N_FOLDS = 10
SOURCE_FILE = Path("resources/en.txt_unchanged")
SOURCE_MAP_PATH = PERPS_DIR / "sourceIdMap.json"
# where to look for the class files the units (and hence the source ids) come
# from; output20 is the fallback for stems measured in output30 from an
# earlier round's generations
UNIT_DIRS = [PERPS_DIR.parent, PERPS_DIR.parent.parent / "output20"]

# keeps the depref run's CSVs apart from the plain one's, as the holdout
# scripts' `..._splitbysource_deprefgood_...` stem does
GOOD_TAG = "" if GOOD_CLASS == "good" else f"_{GOOD_CLASS.lower()}"
RUN_STEM = f"ensemble_auroc_lognorm_cv{N_FOLDS}bysource{GOOD_TAG}_ss{SAMPLE_SIZE}_st{SAMPLE_TIMES}"

# A whole run is stamped with the time it started, because there is normally no
# reason to overwrite an earlier one. A shard is not: its siblings have to be
# findable by name, and re-running one has to replace that shard rather than
# leave two versions of it in the directory the merge reads.
if len(ARGV) > 7:
    CSV_PATH = Path(ARGV[7])
elif SHARD_COUNT == 1:
    CSV_PATH = PERPS_DIR.parent / f"{RUN_STEM}_{str(datetime.datetime.now())}.csv"
else:
    CSV_PATH = (PERPS_DIR.parent / f"{RUN_STEM}_shards"
                / f"shard{SHARD_INDEX:02d}of{SHARD_COUNT:02d}.csv")

SOURCES = loadSources(SOURCE_FILE)

# the folds must be reproducible when SEED is given but still consistent
# within a run when it is not
if SEED is None and SHARD_COUNT > 1:
    sys.exit("--shard needs an explicit SEED: without one every shard would "
             "draw its own folds and its rows would not be comparable with "
             "the other shards'")
SPLIT_SEED = SEED if SEED is not None else int(np.random.SeedSequence().entropy) % 2**32

_rawCache = {}
_idsCache = {}
_sentenceMap = None
_sourceIdMap = None


class Fold:
    """One round of the rotation.

    `dev` holds this fold's source sentences and every other sentence is its
    training content. The z-score statistics and the per-file split
    assignments both depend on which sentences those are, so each fold carries
    its own caches of them rather than sharing one set across the run."""

    def __init__(self, index, dev):
        self.index = index
        self.dev = np.array(sorted(dev))
        self.assign = {}
        self.stats = {}


FOLDS = [Fold(i, dev) for i, dev
         in enumerate(foldSourceIds(SPLIT_SEED, len(SOURCES), N_FOLDS))]


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
    generator's file, whose source sentence splitAssignment looks up.

    EXCLUDED_GENERATORS is applied HERE and not only where the cache is built:
    the cache is shared with the other scoring scripts and is normally already
    on disk, so a blacklist that only filtered at build time would silently do
    nothing on every run after the first."""
    key = (measurer, cls)
    if key not in _rawCache:
        out = {}
        for stem, byMeasurer in loadSentenceMap().get(cls, {}).items():
            if stem.removeprefix("of") in EXCLUDED_GENERATORS:
                continue
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


def sourceIds(cls, stem):
    """The source id of every position of one generator file, or None when the
    file could not be labelled at all. Fold-independent, so it is read once."""
    key = (cls, stem)
    if key not in _idsCache:
        ids = sourceIdMap().get(cls, {}).get(stem)
        _idsCache[key] = None if ids is None else np.asarray(ids)
    return _idsCache[key]


def splitAssignment(fold, cls, stem, length):
    """Per-position split assignment for one generator file in one fold:
    1 train, 0 dev, -1 unusable.

    A position's side is decided entirely by WHICH SOURCE SENTENCE it was
    generated from, so every candidate translation a generator produced for
    one of this fold's sentences, every measurer's score of them, and
    everything the other classes derived from that sentence all land in dev
    together. Positions whose source sentence could not be identified --- and
    whole files with no source ids at all, which is what happens when a stem
    has perplexities but no unit file to read them off --- are dropped from
    both splits in every fold rather than guessed at."""
    key = (cls, stem)
    if key not in fold.assign:
        ids = sourceIds(cls, stem)
        if ids is None:
            fold.assign[key] = None
        else:
            fold.assign[key] = np.where(
                ids < 0, -1, np.where(np.isin(ids, fold.dev), 0, 1)).astype(np.int8)
    assign = fold.assign[key]
    if assign is None:
        return np.full(length, -1, dtype=np.int8)
    if len(assign) < length:
        # the map is built against these very arrays, so this means the cache
        # is stale; drop the unlabelled tail rather than mislabel it
        return np.concatenate([assign, np.full(length - len(assign), -1, dtype=np.int8)])
    return assign[:length]


def measurerStats(fold, measurer):
    """(mean, std) of the measurer's finite log-perps over this fold's TRAIN
    sentences only.

    Re-estimated per fold across all read classes and applied unchanged to
    both of that fold's splits, so no fold's transform ever sees the data it
    is evaluated on. A degenerate std (0 or no data) falls back to 1 so the
    transform stays defined."""
    if measurer not in fold.stats:
        chunks = []
        for cls in READ_CLASSES:
            for stem, a in rawByStem(measurer, cls).items():
                chunks.append(a[splitAssignment(fold, cls, stem, len(a)) == 1])
        values = np.concatenate(chunks or [np.empty(0)])
        values = values[np.isfinite(values)]
        if len(values) == 0:
            fold.stats[measurer] = (0.0, 1.0)
        else:
            std = values.std()
            fold.stats[measurer] = (values.mean(), std if std > 0 else 1.0)
    return fold.stats[measurer]


def ensemblePool(fold, subset, cls, train):
    """Pooled per-unit sums of z-scored log-perps for one split of `cls` in one
    fold.

    Only generator files present for every measurer in the subset are used;
    within each file the measurers are aligned by index (truncated to the
    shortest), and a position contributes only if it is finite for all
    measurers AND its source sentence belongs to the requested split of this
    fold."""
    perMeasurer = [rawByStem(m, cls) for m in subset]
    common = set(perMeasurer[0])
    for d in perMeasurer[1:]:
        common &= set(d)

    stats = [measurerStats(fold, m) for m in subset]
    means = np.array([s[0] for s in stats])[:, None]
    stds = np.array([s[1] for s in stats])[:, None]

    pooled = []
    for stem in sorted(common):
        arrays = [d[stem] for d in perMeasurer]
        length = min(len(a) for a in arrays)
        stacked = np.vstack([a[:length] for a in arrays])  # (measurers, length)
        assign = splitAssignment(fold, cls, stem, length)
        keep = np.isfinite(stacked).all(axis=0) & (assign == (1 if train else 0))
        z = (stacked[:, keep] - means) / stds
        pooled.extend(z.sum(axis=0).tolist())
    return pooled


def aurocSorted(goodSorted, bad):
    """Probability a random `bad` scores higher than a random `good`, ties
    counted half, against an already sorted `good` sample.

    Same quantity as mannwhitneyu(bad, good).statistic / (len(bad)*len(good)),
    counted directly: for each bad value, how many good values it beats plus
    half of how many it ties with."""
    lo = np.searchsorted(goodSorted, bad, side="left")
    hi = np.searchsorted(goodSorted, bad, side="right")
    return float(lo.sum() + 0.5 * (hi - lo).sum()) / (len(goodSorted) * len(bad))


def checkAuroc():
    """Assert aurocSorted agrees with the parent script's mannwhitneyu-based
    AUROC before the run starts.

    The rotation calls it tens of millions of times, which is why it is hand
    rolled at all; small-integer samples are used here so that ties --- the
    only place the two could disagree --- are everywhere."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        good = rng.integers(0, 6, size=97).astype(float)
        bad = rng.integers(0, 6, size=53).astype(float)
        mine = aurocSorted(np.sort(good), bad)
        ref = mannwhitneyu(bad, good, alternative="two-sided").statistic / (len(good) * len(bad))
        if abs(mine - ref) > 1e-9:
            sys.exit(f"AUROC self-check failed: {mine} != {ref}")


def rngFor(index):
    """The subsampling RNG of the subset at position `index` of the full
    enumeration.

    Seeded from (split seed, index) rather than taken from a single stream
    consumed in enumeration order, so a row's numbers depend only on which
    subset it is --- not on how many rows were drawn before it, and therefore
    not on whether the enumeration was cut into shards or where the cuts fell.
    Sharded and undivided runs at the same seed produce the same table, and a
    single shard can be re-run on its own without disturbing the rest."""
    return np.random.default_rng([SPLIT_SEED, index])


def sampledAurocValues(rng, good, bad):
    """The AUROC of each of SAMPLE_TIMES subsamples, as an array.

    SAMPLE_SIZE values are drawn with replacement from each pool per round;
    an empty pool gives an empty array, which aggregate() reports as nan."""
    if not len(good) or not len(bad):
        return np.empty(0)
    goods = np.sort(rng.choice(np.asarray(good), size=(SAMPLE_TIMES, SAMPLE_SIZE)), axis=1)
    bads = rng.choice(np.asarray(bad), size=(SAMPLE_TIMES, SAMPLE_SIZE))
    return np.array([aurocSorted(goods[i], bads[i]) for i in range(SAMPLE_TIMES)])


def aggregate(values):
    """(mean, lo, hi) of the CI over the subsample AUROCs of every fold pooled
    into one distribution."""
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    tail = (100 - CI) / 2
    return values.mean(), np.percentile(values, tail), np.percentile(values, 100 - tail)


def subsetStats(rng, subset, cols):
    """One subset scored over the whole rotation.

    Returns {"train": [(mean, lo, hi) per column], "dev": [...],
    "folds": {"train": [[mean per fold] per column], "dev": [...]}} --- the
    pooled cross-validated estimate and the per-fold means behind it."""
    out = {"folds": {}}
    for part, isTrain in (("train", True), ("dev", False)):
        pooled = [[] for _ in cols]
        foldMeans = [[] for _ in cols]
        for fold in FOLDS:
            data = {cls: ensemblePool(fold, subset, cls, isTrain) for cls in READ_CLASSES}
            good = data.get(GOOD_CLASS, [])
            for ci, cls in enumerate(cols):
                if cls == "combined":
                    bad = [v for c in SELECTED_CLASSES for v in data.get(c, [])]
                else:
                    bad = data.get(cls, [])
                values = sampledAurocValues(rng, good, bad)
                pooled[ci].append(values)
                foldMeans[ci].append(values.mean() if len(values) else float("nan"))
        out[part] = [aggregate(np.concatenate(v)) for v in pooled]
        out["folds"][part] = foldMeans
    return out


def foldStd(stats, part, ci):
    """Spread of a column's per-fold means --- how much the answer depends on
    which sentences were held out."""
    values = np.array(stats["folds"][part][ci])
    values = values[np.isfinite(values)]
    return values.std() if len(values) else float("nan")


def formatCells(stats):
    """Console cells: per column the cross-validated train and dev means as
    'train->dev'."""
    cells = ""
    for (tm, _, _), (dm, _, _) in zip(stats["train"], stats["dev"]):
        if np.isfinite(tm) or np.isfinite(dm):
            t = f"{tm:.3f}" if np.isfinite(tm) else "--"
            d = f"{dm:.3f}" if np.isfinite(dm) else "--"
            cells += f"{t + '->' + d:>16}"
        else:
            cells += f"{'--':>16}"
    return cells


def writeShardMeta():
    """Record what this shard was run with, beside its CSV.

    The shard file name says which slice it is and the header says which
    classes it scored, but neither says which generators were excluded, which
    seed drew the folds, or which measurers were enumerated --- and the
    directory a shard lands in is named after none of those. Two runs that
    differ only in --exclude would otherwise share a directory and merge into
    a table that is half one experiment and half another, with nothing
    visibly wrong with it. mergeCVShards.py compares these and refuses."""
    meta = {
        "good": GOOD_CLASS,
        "classes": SELECTED_CLASSES,
        "excluded": EXCLUDED_GENERATORS,
        "measurers": MEASURERS,
        "seed": SEED,
        "folds": N_FOLDS,
        "sampleSize": SAMPLE_SIZE,
        "sampleTimes": SAMPLE_TIMES,
        "perpsDir": str(PERPS_DIR),
        "shards": SHARD_COUNT,
        "shard": SHARD_INDEX,
    }
    path = CSV_PATH.with_suffix(".meta")
    with path.open("w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)


def writeCsv(labels, cols, rows):
    """One row per subset: per class column the pooled mean/lo/hi and the
    per-fold spread for both splits, then every fold's own mean, so a reader
    can redo the fold-level analysis without rerunning the scoring."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["ensemble"]
    for c in cols:
        for part in ("train", "dev"):
            header += [f"{c}_{part}_mean", f"{c}_{part}_lo", f"{c}_{part}_hi",
                       f"{c}_{part}_foldstd"]
    for c in cols:
        for part in ("train", "dev"):
            header += [f"{c}_{part}_fold{f}" for f in range(N_FOLDS)]

    def fmt(v):
        return "" if not np.isfinite(v) else f"{v:.6f}"

    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for label, stats in zip(labels, rows):
            row = [label]
            for ci in range(len(cols)):
                for part in ("train", "dev"):
                    mean, lo, hi = stats[part][ci]
                    row += [fmt(mean), fmt(lo), fmt(hi), fmt(foldStd(stats, part, ci))]
            for ci in range(len(cols)):
                for part in ("train", "dev"):
                    row += [fmt(v) for v in stats["folds"][part][ci]]
            writer.writerow(row)
    if SHARD_COUNT > 1:
        writeShardMeta()
    print(f"wrote {CSV_PATH}")


def fmtStat(stat):
    mean, lo, hi = stat
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]" if np.isfinite(mean) else "--"


def printBestPerSize(subsets, labels, rows, cols):
    """Per column: the subset with the best cross-validated TRAIN mean at each
    size, and what it scores on held-out sentences.

    One subset per size, judged on all ten folds' training halves at once ---
    the closest analogue of the single-holdout script's table."""
    width = max(16, max(len(l) for l in labels) + 2)
    for ci, cls in enumerate(cols):
        print(f"\nbest-on-train ensembles per size --- {cls}")
        print(f"{'k':>3} {'ensemble':<{width}}{'train (mean [lo,hi])':>24}{'dev (mean [lo,hi])':>24}{'dev foldstd':>13}")
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
            print(f"{k:>3} {label:<{width}}{fmtStat(stats['train'][ci]):>24}"
                  f"{fmtStat(stats['dev'][ci]):>24}{foldStd(stats, 'dev', ci):>13.3f}")
            if bestGlobal is None or stats["train"][ci][0] > bestGlobal[1]["train"][ci][0]:
                bestGlobal = (label, stats)
        if bestGlobal is not None:
            label, stats = bestGlobal
            print(f"  best on train: {label}  "
                  f"train {fmtStat(stats['train'][ci])}  dev {fmtStat(stats['dev'][ci])}")


def printFoldSelection(subsets, labels, rows, cols):
    """Per column: the SELECTION PROCEDURE cross-validated.

    At each size k every fold picks its own best-on-train subset and is then
    scored with it on its own held-out sentences; the mean of those dev scores
    estimates what "choose an ensemble on the data you have, use it on
    sentences you have not seen" is worth at that size, rather than what one
    particular ensemble is worth. The agreement column is how many folds
    picked the same subset as the most popular one --- when it is low, the
    winning subset at that size is not identifiable from 45 sentences, however
    good its score looks."""
    for ci, cls in enumerate(cols):
        print(f"\ncross-validated selection --- {cls}")
        print(f"{'k':>3}{'train of picked':>17}{'dev of picked':>15}{'dev sd':>9}{'agree':>8}  most picked")
        for k in range(1, len(MEASURERS) + 1):
            picks = []
            for f in range(N_FOLDS):
                best = None
                for subset, label, stats in zip(subsets, labels, rows):
                    if len(subset) != k:
                        continue
                    train = stats["folds"]["train"][ci][f]
                    if not np.isfinite(train):
                        continue
                    if best is None or train > best[1]:
                        best = (label, train, stats["folds"]["dev"][ci][f])
                if best is not None:
                    picks.append(best)
            if not picks:
                continue
            trainMeans = np.array([p[1] for p in picks])
            devMeans = np.array([p[2] for p in picks])
            top, agree = Counter(p[0] for p in picks).most_common(1)[0]
            print(f"{k:>3}{np.nanmean(trainMeans):>17.3f}{np.nanmean(devMeans):>15.3f}"
                  f"{np.nanstd(devMeans):>9.3f}{f'{agree}/{len(picks)}':>8}  {top}")


def printFolds():
    """The rotation itself: which source sentences each fold holds out, and
    how many units of each class that puts in each fold's dev half.

    Every labelled unit appears in exactly one fold's dev column, so the
    columns sum to the total and a fold's train count is total minus its own
    column."""
    print(f"{N_FOLDS} folds over {len(SOURCES)} source sentences, "
          f"every sentence held out exactly once "
          f"(sizes {', '.join(str(len(f.dev)) for f in FOLDS)})")
    for fold in FOLDS:
        print(f"  fold {fold.index}: {[int(i) for i in fold.dev]}")
    header = (f"{'class':<18}" + "".join(f"{'f' + str(i):>8}" for i in range(N_FOLDS))
              + f"{'total':>9}{'dropped':>9}{'generators':>12}")
    print(header)
    for cls in READ_CLASSES:
        byStem = rawByStem(MEASURERS[0], cls)
        devCounts = []
        for fold in FOLDS:
            devCounts.append(sum(int((splitAssignment(fold, cls, stem, len(a)) == 0).sum())
                                 for stem, a in byStem.items()))
        dropped = sum(int((splitAssignment(FOLDS[0], cls, stem, len(a)) == -1).sum())
                      for stem, a in byStem.items())
        print(f"{cls:<18}" + "".join(f"{n:>8}" for n in devCounts)
              + f"{sum(devCounts):>9}{dropped:>9}{len(byStem):>12}")


def main():
    checkAuroc()
    allSubsets = [s for r in range(1, len(MEASURERS) + 1)
                  for s in combinations(MEASURERS, r)]
    # every N-th subset, so the shard's workload has the same mix of subset
    # sizes as the whole enumeration rather than one end of it
    mine = list(enumerate(allSubsets))[SHARD_INDEX::SHARD_COUNT]
    subsets = [s for _, s in mine]
    labels = ["*".join(s) for s in subsets]

    cols = SELECTED_CLASSES + (["combined"] if len(SELECTED_CLASSES) > 1 else [])

    print(f"log-z-normalized ensemble subsets of {{{','.join(MEASURERS)}}} "
          f"good class={GOOD_CLASS}  "
          f"with {N_FOLDS}-fold cross validation "
          f"over the {len(SOURCES)} SOURCE SENTENCES  "
          f"sample_size={SAMPLE_SIZE}  sample_times={SAMPLE_TIMES}  "
          f"classes={','.join(SELECTED_CLASSES)}  "
          f"excluded={','.join(EXCLUDED_GENERATORS)}  "
          f"CI={CI}%" + (f"  seed={SEED}" if SEED is not None else f"  split_seed={SPLIT_SEED}")
          + (f"  shard={SHARD_INDEX}/{SHARD_COUNT} "
             f"({len(subsets)} of {len(allSubsets)} subsets)" if SHARD_COUNT > 1 else ""))
    printFolds()
    width = max(16, max(len(l) for l in labels) + 2)
    header = f"{'ensemble':<{width}}" + "".join(f"{c + ' (train->dev)':>16}" for c in cols)
    print(header)
    print("-" * len(header))

    rows = []
    for (index, subset), label in zip(mine, labels):
        stats = subsetStats(rngFor(index), subset, cols)
        rows.append(stats)
        print(f"{label:<{width}}{formatCells(stats)}", flush=True)

    if SHARD_COUNT == 1:
        printBestPerSize(subsets, labels, rows, cols)
        printFoldSelection(subsets, labels, rows, cols)
    else:
        # both tables rank subsets against each other, so they mean nothing
        # until every shard's rows are on the table
        print("\nshard run: the per-size and selection tables need the whole "
              "enumeration.\nRun ./python/mergeCVShards.py on "
              f"{CSV_PATH.parent} once all {SHARD_COUNT} shards are written.")
    writeCsv(labels, cols, rows)


if __name__ == "__main__":
    main()
