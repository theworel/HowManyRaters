"""Plot AUROC vs. sample size for each measuring model.

For a single bad-ish class (bad/broken/corrupt/corruptTriple/candidateCorrupt/
deprefBroken)
this draws one line per measuring model. The X axis is the subsample size, the
Y axis is the mean AUROC reached at that size (with a 90% confidence interval as
error bars), estimated by the same repeated-subsampling procedure as
`scoreSeparationSampled.py`.

For every sample size in SAMPLE_SIZES we draw, `sample_times` times, that many
good and that many bad perplexities (without replacement) and compute AUROC on
the draw. The point is the mean of those AUROCs; the error bar spans the
5th/95th percentiles. If a measurer has fewer than `sample_size` good or bad
perplexities pooled for the class, that point is left blank.

AUROC is the probability that a random bad sentence scores higher perplexity
than a random good one (0.5 -> no separation, 1.0 -> perfect).

Usage:
    ../v/bin/python3.11 ./python/plotAurocVsSampleSize.py SAMPLE_TIMES CLASS [PERPS_DIR] [OUT_PATH] [SEED]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

SAMPLE_TIMES = int(sys.argv[1])
CLASS = sys.argv[2]
PERPS_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("resources/output30/perps")
OUT_PATH = Path(sys.argv[4]) if len(sys.argv) > 4 \
    else PERPS_DIR.parent / "graphs" / f"auroc_vs_samplesize_{CLASS}.png"
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else None

# the seven measuring models to plot, in legend order
MEASURERS = ["ayaexp", "eu", "commandr", "llama", "gemmabig", "qwen", "mistral7b"]
SAMPLE_SIZES = [10, 20, 50, 100, 200, 500, 1000]

# generating models whose of<model>.txt files are ignored entirely
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


def poolFor(measurer, cls):
    """Pool all finite perplexities for one measurer/class, or [] if none."""
    clsDir = PERPS_DIR / f"by{measurer}" / cls
    pooled = []
    for f in sorted(clsDir.glob("*.txt")):
        if f.stem.removeprefix("of") in EXCLUDED_GENERATORS:
            continue
        values = readPerps(f)
        if values is None:
            print(f"skipping invalid file: {f}", file=sys.stderr)
            continue
        pooled.extend(values)
    return pooled


def auroc(good, bad):
    """Probability a random `bad` scores higher perplexity than a random `good`."""
    if not len(good) or not len(bad):
        return float("nan")
    u = mannwhitneyu(bad, good, alternative="two-sided").statistic
    return u / (len(bad) * len(good))


def curve(rng, good, bad):
    """Return (means, los, his) over SAMPLE_SIZES; nan where the pool is too small.

    For each sample size we subsample without replacement SAMPLE_TIMES times,
    compute AUROC per draw, then take the mean and the 5th/95th percentiles.
    A sample size larger than either pool yields nan (a blank point).
    """
    good = np.asarray(good)
    bad = np.asarray(bad)
    tail = (100 - CI) / 2
    means, los, his = [], [], []
    for size in SAMPLE_SIZES:
        if len(good) < size or len(bad) < size:
            means.append(np.nan)
            los.append(np.nan)
            his.append(np.nan)
            continue
        scores = np.array([
            auroc(rng.choice(good, size=size, replace=False),
                  rng.choice(bad, size=size, replace=False))
            for _ in range(SAMPLE_TIMES)
        ])
        means.append(scores.mean())
        los.append(np.percentile(scores, tail))
        his.append(np.percentile(scores, 100 - tail))
    return np.array(means), np.array(los), np.array(his)


def main():
    rng = np.random.default_rng(SEED)
    xs = np.array(SAMPLE_SIZES, dtype=float)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, measurer in enumerate(MEASURERS):
        good = poolFor(measurer, "good")
        bad = poolFor(measurer, CLASS)
        means, los, his = curve(rng, good, bad)

        finite = np.isfinite(means)
        n = int(finite.sum())
        label = f"{measurer} (n_good={len(good)}, n_{CLASS}={len(bad)})"
        if n == 0:
            # nothing to draw; still register the model in the legend
            ax.plot([], [], color=colors[i], marker="o", label=label)
            continue

        # small multiplicative dodge so overlapping error bars stay readable
        dodge = 1.0 + (i - (len(MEASURERS) - 1) / 2) * 0.012
        xd = xs * dodge
        yerr = np.vstack([
            np.clip(means - los, 0, None),
            np.clip(his - means, 0, None),
        ])
        ax.errorbar(xd[finite], means[finite], yerr=yerr[:, finite],
                    color=colors[i], marker="o", capsize=3, linewidth=1.5,
                    elinewidth=1, label=label)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(SAMPLE_SIZES)
    ax.set_xticklabels([str(s) for s in SAMPLE_SIZES])
    ax.set_xlabel("sample size")
    ax.set_ylabel("AUROC (mean, 90% CI)")
    ax.set_title(f"AUROC vs. sample size — good vs. {CLASS}  "
                 f"(sample_times={SAMPLE_TIMES})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
