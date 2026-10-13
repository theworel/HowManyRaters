"""Score each measuring model's power to distinguish good from bad/broken/corrupt.

For every measurer we compute AUROC (= normalized Mann-Whitney U) treating the
pooled `good` perplexities as the negative class (should be low) and each bad
class as the positive class (should be high). AUROC is the probability that a
random bad sentence scores higher perplexity than a random good one:

    0.5 -> no distinguishing power
    1.0 -> perfect separation
   <0.5 -> the measurer ranks good as *worse* than bad (inverted)

AUROC is rank-based, so it is invariant to each measurer's perplexity scale,
which is what makes measurers comparable to each other.

Usage:
    ../v/bin/python3.11 ./python/scoreSeparation.py [PERPS_DIR]
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

PERPS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("resources/output30/perps")

CLASSES = ["good", "bad", "broken", "corrupt", "corruptTriple", "candidateCorrupt", "deprefBroken"]
BAD_CLASSES = ["bad", "broken", "corrupt", "corruptTriple", "candidateCorrupt", "deprefBroken"]
EXCLUDED_GENERATORS = ["arnir", "liquidai"]


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
        for cls in CLASSES:
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
    if not good or not bad:
        return float("nan")
    u = mannwhitneyu(bad, good, alternative="two-sided").statistic
    return u / (len(bad) * len(good))


def main():
    data = collect()
    cols = BAD_CLASSES + ["any-bad"]
    header = f"{'measurer':<16}" + "".join(f"{c:>10}" for c in cols)
    print(header)
    print("-" * len(header))

    for measurer in sorted(data):
        perClass = data[measurer]
        good = perClass.get("good", [])
        scores = []
        for cls in BAD_CLASSES:
            scores.append(auroc(good, perClass.get(cls, [])))
        anyBad = [v for cls in BAD_CLASSES for v in perClass.get(cls, [])]
        scores.append(auroc(good, anyBad))
        cells = "".join(f"{s:>10.3f}" if np.isfinite(s) else f"{'--':>10}"
                         for s in scores)
        print(f"{measurer:<16}{cells}")


if __name__ == "__main__":
    main()
