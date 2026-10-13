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

CLASSES = ["good", "deprefGood", "broken", "deprefBroken", "corruptTriple", "candidateCorrupt"]

# generating models whose of<model>.txt files are ignored entirely
EXCLUDED_GENERATORS = ["arnir", "liquidai", "cfourai", "gemma", "ayabig"]
COLORS = {"good": "tab:green", "deprefGood": "tab:blue", "broken": "tab:red", "deprefBroken": "tab:pink", "corrupt": "tab:purple", "corruptTriple": "tab:brown", "candidateCorrupt": "tab:orange"}

# legend labels naming what the text actually is, rather than the directory it
# lives in: the two positive classes differ by author (LLM vs human) and each
# degraded class says which of the two it was derived from
LABELS = {
    "good": "LLM candidate",
    "deprefGood": "human reference",
    "broken": "broken LLM candidate",
    "deprefBroken": "broken human reference",
    "corrupt": "corrupt human reference",
    "corruptTriple": "corrupt human reference",
    "candidateCorrupt": "corrupt LLM candidate",
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

    data[measurer][cls] = pooled list of perplexities across all generator files.
    nanCounts[measurer] = total number of nan/non-finite values dropped.
    """
    data = {}
    nanCounts = {}
    for byDir in sorted(PERPS_DIR.glob("by*")):
        print(byDir)
        measurer = byDir.name[2:]
        perClass = {}
        nans = 0
        for cls in CLASSES:
            print(cls)
            pooled = []
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
                perClass[cls] = pooled
        if perClass:
            data[measurer] = perClass
        nanCounts[measurer] = nans
    return data, nanCounts


def plotMeasurer(measurer, perClass, outPath):
    # small canvas on purpose: font sizes stay at matplotlib's defaults, so a
    # smaller figure means larger text once the pdf is scaled to column width
    fig, ax = plt.subplots(figsize=(5, 3))
    for cls in CLASSES:
        values = perClass.get(cls)
        if not values or len(set(values)) < 2:
            continue
        logVals = np.log(np.array(values))
        xs = np.linspace(logVals.min() - 0.5, logVals.max() + 0.5, 500)
        density = gaussian_kde(logVals)(xs)
        ax.plot(np.exp(xs), density, color=COLORS[cls],
                label=f"{LABELS.get(cls, cls)} (n={len(values)})")
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
    for measurer, perClass in data.items():
        outPath = OUT_DIR / f"perp_dist_by{measurer}.{FORMAT}"
        plotMeasurer(measurer, perClass, outPath)
        counts = ", ".join(f"{c}: {len(v)}" for c, v in perClass.items())
        print(f"{outPath} ({counts})")

    print("\nnan values encountered per measurer:")
    for measurer in sorted(nanCounts):
        print(f"  {measurer}: {nanCounts[measurer]}")
    print(f"  total: {sum(nanCounts.values())}")


if __name__ == "__main__":
    main()
