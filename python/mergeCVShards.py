"""Concatenate the shard CSVs of one sharded cross-validation run into the
single table the undivided run would have written, and print the two tables
that only exist once every row is present.

scoreSeparationSampledEnsembleNormalizedLogCVBySentence.py --shard=I/N writes
one CSV per shard into a `..._shards` directory. Each row of those files is
already final --- a subset's folds, z-score statistics and subsampled AUROCs
are computed without reference to any other subset, and the RNG is seeded per
subset --- so merging is concatenation plus a re-sort into enumeration order,
not a re-computation or an average of partial results. What the shards cannot
produce on their own are the two tables that rank subsets against each other:
the best-on-train ensemble at each size, and the cross-validated selection
procedure. Those are printed here instead, recomputed from the merged rows
exactly as the undivided script computes them from its in-memory results.

The checks are the point of the script as much as the concatenation is. A run
spread over ten jobs fails in ways one job does not: a shard can be evicted
from the queue, or die on its own, or be re-run against a changed cache while
its siblings hold the old numbers. So this refuses to write unless the shards
form a complete 0..N-1 set with identical headers and their labels are exactly
the enumeration, once each.

Usage:
    ../v/bin/python3.11 ./python/mergeCVShards.py SHARD_DIR [OUT_CSV]
"""

import csv
import datetime
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

SHARD_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if SHARD_DIR is None:
    sys.exit(__doc__.strip().splitlines()[-1].strip())

MEASURERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]

SHARD_RE = re.compile(r"^shard(\d+)of(\d+)\.csv$")


def checkMeta(byIndex):
    """Every shard's .meta against every other's, and print the one they agree
    on.

    The directory name carries the positive class, the fold count and the
    sample sizes, and the CSV header carries the scored classes --- but
    nothing on disk carries the seed, the excluded generators or the measurer
    list, and a shard re-run after any of those changed lands on top of its
    predecessor looking exactly like it. This is the check that catches it."""
    configs = {}
    for index, path in sorted(byIndex.items()):
        metaPath = path.with_suffix(".meta")
        if not metaPath.exists():
            sys.exit(f"{metaPath.name} is missing --- {path.name} was written "
                     "by a version that did not record its configuration, so "
                     "there is no way to tell whether it belongs with the rest")
        with metaPath.open() as fh:
            meta = json.load(fh)
        meta.pop("shard", None)
        configs[index] = meta

    first = configs[min(configs)]
    for index, meta in configs.items():
        if meta != first:
            differs = sorted(k for k in set(meta) | set(first)
                             if meta.get(k) != first.get(k))
            sys.exit(f"shard {index} was run with a different "
                     f"{', '.join(differs)} than shard {min(configs)} "
                     f"({[meta.get(k) for k in differs]} vs "
                     f"{[first.get(k) for k in differs]}) --- these are not "
                     "one experiment")

    print(f"  good={first['good']}  classes={','.join(first['classes'])}  "
          f"excluded={','.join(first['excluded'])}  seed={first['seed']}  "
          f"folds={first['folds']}  "
          f"ss={first['sampleSize']}  st={first['sampleTimes']}")
    return first


def readShards(directory):
    """(header, {label: row}) from every shard CSV in `directory`.

    Exits rather than returns on anything that would make the merged table a
    silent fiction: a missing or duplicated shard index, shards that disagree
    on the shape of the table, or one subset scored twice."""
    files = {}
    for path in sorted(directory.glob("*.csv")):
        m = SHARD_RE.match(path.name)
        if m is None:
            print(f"ignoring {path.name}: not a shard file", file=sys.stderr)
            continue
        index, count = int(m.group(1)), int(m.group(2))
        files.setdefault(count, {})[index] = path

    if not files:
        sys.exit(f"no shardIIofNN.csv files in {directory}")
    if len(files) > 1:
        sys.exit(f"{directory} mixes runs sharded {sorted(files)} ways --- "
                 "keep one run per directory")

    count, byIndex = next(iter(files.items()))
    missing = [i for i in range(count) if i not in byIndex]
    if missing:
        sys.exit(f"{len(missing)} of {count} shards missing: {missing}")

    checkMeta(byIndex)

    header = None
    rows = {}
    for index in range(count):
        path = byIndex[index]
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            shardHeader = next(reader, None)
            if shardHeader is None:
                sys.exit(f"{path.name} is empty")
            if header is None:
                header = shardHeader
            elif shardHeader != header:
                sys.exit(f"{path.name} has a different header from "
                         f"{byIndex[0].name} --- the shards are not one run")
            n = 0
            for row in reader:
                label = row[0]
                if label in rows:
                    sys.exit(f"{label} appears in more than one shard "
                             f"(again in {path.name})")
                rows[label] = row
                n += 1
        print(f"  {path.name}: {n} rows")
    return header, rows


def canonicalLabels():
    """The subset labels in enumeration order, so the merged file is ordered
    the way an undivided run's would be rather than by shard."""
    return ["*".join(s) for r in range(1, len(MEASURERS) + 1)
            for s in combinations(MEASURERS, r)]


def checkComplete(rows):
    """The merged label set against the enumeration it is supposed to be."""
    expected = canonicalLabels()
    missing = [l for l in expected if l not in rows]
    extra = [l for l in rows if l not in set(expected)]
    if missing:
        sys.exit(f"{len(missing)} of {len(expected)} subsets absent from the "
                 f"shards, e.g. {missing[:5]} --- a shard died partway")
    if extra:
        sys.exit(f"{len(extra)} rows are not subsets of "
                 f"{{{','.join(MEASURERS)}}}, e.g. {extra[:5]}")
    return expected


def parseHeader(header):
    """(columns, nFolds) recovered from the header, so this script does not
    have to be kept in step with the scoring script's class list."""
    cols = []
    for name in header:
        m = re.match(r"^(.*)_train_mean$", name)
        if m:
            cols.append(m.group(1))
    nFolds = sum(1 for name in header
                 if re.match(rf"^{re.escape(cols[0])}_train_fold\d+$", name))
    if not cols or not nFolds:
        sys.exit("header has no <class>_train_mean / _train_foldN columns")
    return cols, nFolds


class Table:
    """The merged rows, addressed the way the two tables want to read them."""

    def __init__(self, header, rows, order):
        self.index = {name: i for i, name in enumerate(header)}
        self.labels = order
        self.rows = [rows[l] for l in order]
        self.sizes = [l.count("*") + 1 for l in order]

    def value(self, r, name):
        cell = self.rows[r][self.index[name]]
        return float(cell) if cell else float("nan")

    def stat(self, r, cls, part):
        return tuple(self.value(r, f"{cls}_{part}_{k}")
                     for k in ("mean", "lo", "hi"))


def isFinite(x):
    return x == x  # nan is the only value that fails this


def mean(values):
    values = [v for v in values if isFinite(v)]
    return sum(values) / len(values) if values else float("nan")


def sd(values):
    values = [v for v in values if isFinite(v)]
    if not values:
        return float("nan")
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def fmtStat(stat):
    m, lo, hi = stat
    return f"{m:.3f} [{lo:.3f},{hi:.3f}]" if isFinite(m) else "--"


def printBestPerSize(table, cols):
    """Per column: the subset with the best cross-validated TRAIN mean at each
    size, and what it scores on held-out sentences."""
    width = max(16, max(len(l) for l in table.labels) + 2)
    for cls in cols:
        print(f"\nbest-on-train ensembles per size --- {cls}")
        print(f"{'k':>3} {'ensemble':<{width}}{'train (mean [lo,hi])':>24}"
              f"{'dev (mean [lo,hi])':>24}{'dev foldstd':>13}")
        bestGlobal = None
        for k in range(1, len(MEASURERS) + 1):
            best = None
            for r, size in enumerate(table.sizes):
                if size != k:
                    continue
                train = table.value(r, f"{cls}_train_mean")
                if not isFinite(train):
                    continue
                if best is None or train > best[1]:
                    best = (r, train)
            if best is None:
                continue
            r = best[0]
            print(f"{k:>3} {table.labels[r]:<{width}}"
                  f"{fmtStat(table.stat(r, cls, 'train')):>24}"
                  f"{fmtStat(table.stat(r, cls, 'dev')):>24}"
                  f"{table.value(r, f'{cls}_dev_foldstd'):>13.3f}")
            if bestGlobal is None or best[1] > bestGlobal[1]:
                bestGlobal = best
        if bestGlobal is not None:
            r = bestGlobal[0]
            print(f"  best on train: {table.labels[r]}  "
                  f"train {fmtStat(table.stat(r, cls, 'train'))}  "
                  f"dev {fmtStat(table.stat(r, cls, 'dev'))}")


def printFoldSelection(table, cols, nFolds):
    """Per column: the SELECTION PROCEDURE cross-validated.

    At each size k every fold picks its own best-on-train subset and is scored
    with it on its own held-out sentences, so the dev column estimates what
    "choose an ensemble on the data you have, use it on sentences you have not
    seen" is worth --- not what one particular ensemble is worth. The
    agreement column counts how many folds picked the most popular subset; a
    low count means the winner at that size is not identifiable from 45
    sentences, however good its score looks."""
    for cls in cols:
        print(f"\ncross-validated selection --- {cls}")
        print(f"{'k':>3}{'train of picked':>17}{'dev of picked':>15}"
              f"{'dev sd':>9}{'agree':>8}  most picked")
        for k in range(1, len(MEASURERS) + 1):
            picks = []
            for f in range(nFolds):
                best = None
                for r, size in enumerate(table.sizes):
                    if size != k:
                        continue
                    train = table.value(r, f"{cls}_train_fold{f}")
                    if not isFinite(train):
                        continue
                    if best is None or train > best[1]:
                        best = (table.labels[r], train,
                                table.value(r, f"{cls}_dev_fold{f}"))
                if best is not None:
                    picks.append(best)
            if not picks:
                continue
            top, agree = Counter(p[0] for p in picks).most_common(1)[0]
            print(f"{k:>3}{mean([p[1] for p in picks]):>17.3f}"
                  f"{mean([p[2] for p in picks]):>15.3f}"
                  f"{sd([p[2] for p in picks]):>9.3f}"
                  f"{f'{agree}/{len(picks)}':>8}  {top}")


def outPath(shardDir):
    """Where the merged table goes when the caller does not say.

    Next to the shard directory and named like an undivided run's output ---
    the shard directory is `<stem>_shards`, so the merged file is `<stem>` plus
    the time it was merged, which is what every reader of these CSVs globs
    for."""
    stem = shardDir.name
    stem = stem[:-len("_shards")] if stem.endswith("_shards") else stem + "_merged"
    return shardDir.parent / f"{stem}_{str(datetime.datetime.now())}.csv"


def main():
    if not SHARD_DIR.is_dir():
        sys.exit(f"{SHARD_DIR} is not a directory")
    print(f"reading shards from {SHARD_DIR}")
    header, rows = readShards(SHARD_DIR)
    order = checkComplete(rows)
    cols, nFolds = parseHeader(header)
    print(f"{len(order)} subsets, {len(cols)} columns "
          f"({','.join(cols)}), {nFolds} folds")

    table = Table(header, rows, order)
    printBestPerSize(table, cols)
    printFoldSelection(table, cols, nFolds)

    path = Path(sys.argv[2]) if len(sys.argv) > 2 else outPath(SHARD_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(table.rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
