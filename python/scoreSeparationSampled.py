"""Score each measurer's good/bad separation power via repeated subsampling.

Like `scoreSeparation.py`, but instead of using every pooled perplexity at once
we estimate AUROC from random subsamples. For each of `sample_times` repetitions
we draw `sample_size` good and `sample_size` bad perplexities and compute AUROC
on that draw. We then report, per measurer and per class, the mean AUROC and a
90% confidence interval (5th/95th percentiles of the subsample distribution).

`classes` is a comma-separated list of bad-ish classes to evaluate (e.g.
"bad,broken"); the output has one column per listed class plus a "combined"
column pooling exactly those listed classes. `good` is always the negative class.

AUROC is the probability that a random bad sentence scores higher perplexity
than a random good one:

    0.5 -> no distinguishing power
    1.0 -> perfect separation
   <0.5 -> the measurer ranks good as *worse* than bad (inverted)

Usage:
    ../v/bin/python3.11 ./python/scoreSeparationSampled.py SAMPLE_SIZE SAMPLE_TIMES [CLASSES] [PERPS_DIR] [SEED]
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

SAMPLE_SIZE = int(sys.argv[1])
SAMPLE_TIMES = int(sys.argv[2])
SELECTED_CLASSES = [c.strip() for c in sys.argv[3].split(",") if c.strip()] \
    if len(sys.argv) > 3 else ["bad", "broken", "corrupt", "corruptTriple", "candidateCorrupt", "deprefBroken"]
PERPS_DIR = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("resources/output30/perps")
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else None

# classes whose files we actually need to read: good (negative) + selected
READ_CLASSES = ["good"] + [c for c in SELECTED_CLASSES if c != "good"]
EXCLUDED_GENERATORS = ["arnir", "liquidai"]
CI = 90  # confidence interval width in percent


def readPerps(path):
    """Return finite perplexities from the third line, or None if invalid."""
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        return None
    line = lines[2].strip()
    if not (line.startswith("[") and line.endswith("]")):
        return None
    try:
        values = [float(v) for v in line[1:-1].split(",") if v.strip()]
    except ValueError:
        return None
    return [v for v in values if np.isfinite(v)]


def collect():
    """data[measurer][cls] = pooled perplexities across all generator files."""
    data = {}
    for byDir in sorted(PERPS_DIR.glob("by*")):
        measurer = byDir.name[2:]
        perClass = {}
        for cls in READ_CLASSES:
            pooled = []
            for f in sorted((byDir / cls).glob("*.txt")):
                if f.stem.removeprefix("of") in EXCLUDED_GENERATORS:
                    continue
                values = readPerps(f)
                if values is None:
                    print(f"skipping invalid file: {f}", file=sys.stderr)
                    continue
                pooled.extend(values)
            if pooled:
                perClass[cls] = pooled
        if perClass:
            data[measurer] = perClass
    return data


def auroc(good, bad):
    """Probability a random `bad` scores higher perplexity than a random `good`.

    Equivalent to the ROC AUC with good=negative, bad=positive. Ties count 0.5
    (mannwhitneyu's default tie handling gives exactly that)."""
    if not len(good) or not len(bad):
        return float("nan")
    u = mannwhitneyu(bad, good, alternative="two-sided").statistic
    return u / (len(bad) * len(good))


def sample(rng, pool):
    """Draw SAMPLE_SIZE values from pool (without replacement when possible)."""
    pool = np.asarray(pool)
    replace = len(pool) < SAMPLE_SIZE
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


def main():
    rng = np.random.default_rng(SEED)
    data = collect()
    # one column per selected class, plus a "combined" column pooling them all
    # (omit "combined" when only a single class is selected — it would duplicate)
    cols = SELECTED_CLASSES + (["combined"] if len(SELECTED_CLASSES) > 1 else [])

    print(f"sample_size={SAMPLE_SIZE}  sample_times={SAMPLE_TIMES}  "
          f"classes={','.join(SELECTED_CLASSES)}  "
          f"CI={CI}%" + (f"  seed={SEED}" if SEED is not None else ""))
    header = f"{'measurer':<16}" + "".join(f"{c + ' (mean [lo, hi])':>26}" for c in cols)
    print(header)
    print("-" * len(header))

    for measurer in sorted(data):
        perClass = data[measurer]
        good = perClass.get("good", [])
        cells = ""
        for cls in cols:
            if cls == "combined":
                bad = [v for c in SELECTED_CLASSES for v in perClass.get(c, [])]
            else:
                bad = perClass.get(cls, [])
            mean, lo, hi = sampledAuroc(rng, good, bad)
            if np.isfinite(mean):
                cells += f"{mean:>10.3f} [{lo:.3f},{hi:.3f}]"
            else:
                cells += f"{'--':>26}"
        print(f"{measurer:<16}{cells}")


if __name__ == "__main__":
    main()
