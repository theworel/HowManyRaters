import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PERPS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("resources/output30/perps")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else PERPS_DIR.parent / "graphs"
# vector by default so the curves stay sharp when scaled into a paper;
# matplotlib picks the backend from the extension, so "svg" or "eps" also work
FORMAT = sys.argv[3].lstrip(".") if len(sys.argv) > 3 else "pdf"

# Type 42 (TrueType) instead of matplotlib's default Type 3, which several
# publishers reject and which some PDF viewers render badly at small sizes
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Same six classes as plotPerplexityDistributions.py, but drawn as three
# curves: each group pools its classes regardless of who wrote the text they
# were derived from (LLM candidate vs human reference). So the two positive
# classes are one curve, the two LLM-corruption classes are another, and the
# two MorphoDiTa corruptions -- broken from good, deprefBroken from deprefGood
# -- are the third.
GROUPS = {
    "good": ["good", "deprefGood"],
    "corrupt": ["corruptTriple", "candidateCorrupt"],
    "broken": ["broken", "deprefBroken"],
}
CLASSES = [cls for classes in GROUPS.values() for cls in classes]

# generating models whose of<model>.txt files are ignored entirely
EXCLUDED_GENERATORS = ["arnir", "liquidai", "cfourai", "gemma", "ayabig"]
COLORS = {"good": "tab:green", "corrupt": "tab:orange", "broken": "tab:red"}

# legend labels naming what the text actually is, rather than the directories
# the group was pooled from
LABELS = {
    "good": "reference translations",
    "corrupt": "semantically damaged translations",
    "broken": "morphologically damaged translations",
}


def readPerps(path):
    """Return (finite perplexities, nan count) from the third line, or None if invalid.

    The list may contain nan entries (failed measurements), which literal_eval
    cannot parse, so parse the floats manually and drop non-finite values. The
    second element of the tuple reports how many non-finite values were dropped.
    """
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
    finite = [v for v in values if np.isfinite(v)]
    nanCount = len(values) - len(finite)
    return finite, nanCount


def collect():
    """Return (data, nanCounts).

    data[measurer][group] = pooled list of perplexities across every class in
    the group and every generator file in those classes.
    nanCounts[measurer] = total number of nan/non-finite values dropped.
    """
    data = {}
    nanCounts = {}
    for byDir in sorted(PERPS_DIR.glob("by*")):
        print(byDir)
        measurer = byDir.name[2:]
        perGroup = {}
        nans = 0
        for group, classes in GROUPS.items():
            pooled = []
            for cls in classes:
                print(cls)
                for f in sorted((byDir / cls).glob("*.txt")):
                    if f.stem.removeprefix("of") in EXCLUDED_GENERATORS:
                        continue
                    print(f.stem)
                    result = readPerps(f)
                    if result is None:
                        print(f"skipping invalid file: {f}")
                        continue
                    values, nanCount = result
                    nans += nanCount
                    pooled.extend(values)
            if pooled:
                perGroup[group] = pooled
        if perGroup:
            data[measurer] = perGroup
        nanCounts[measurer] = nans
    return data, nanCounts


def plotMeasurer(measurer, perGroup, outPath):
    # small canvas on purpose: font sizes stay at matplotlib's defaults, so a
    # smaller figure means larger text once the pdf is scaled to column width
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for group in GROUPS:
        values = perGroup.get(group)
        if not values or len(set(values)) < 2:
            continue
        logVals = np.log(np.array(values))
        xs = np.linspace(logVals.min() - 0.5, logVals.max() + 0.5, 500)
        density = gaussian_kde(logVals)(xs)
        ax.plot(np.exp(xs), density, color=COLORS[group],
                label=f"{LABELS.get(group, group)} (n={len(values)})")
    ax.set_xscale("log")
    ax.set_xlabel("perplexity (log scale)")
    ax.set_ylabel("density")
    ax.set_title(f"Perplexity distributions of different classes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outPath)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, nanCounts = collect()
    print(OUT_DIR)
    for measurer, perGroup in data.items():
        outPath = OUT_DIR / f"perp_dist_merged_by{measurer}.{FORMAT}"
        plotMeasurer(measurer, perGroup, outPath)
        counts = ", ".join(f"{g}: {len(v)}" for g, v in perGroup.items())
        print(f"{outPath} ({counts})")

    print("\nnan values encountered per measurer:")
    for measurer in sorted(nanCounts):
        print(f"  {measurer}: {nanCounts[measurer]}")
    print(f"  total: {sum(nanCounts.values())}")


if __name__ == "__main__":
    main()
