"""Report which of a class's perplexity numbers are missing or unusable, so a
half-finished measurement round can be resubmitted without reading 169 files by
hand.

Three separate things can be wrong, and the report keeps them apart:

  * a rater x generator cell of the grid has no file, an empty one, or one that
    stops after the `len:` line --- the signature of a run that died before it
    printed a mean, which for the big raters (dolphin 48 GB, ibm 58 GB, darwin
    69 GB of bf16 weights) means it was started without the memory to hold the
    model. Such a file is 9 bytes and is what the resubmission list is for.
  * a cell whose mean is nan, or whose list is shorter than the class file it
    was measured off --- units were dropped, so the column is not comparable to
    the others even though the file looks finished.
  * the class file itself is short. candidateCorrupt and broken rewrite the
    good-class file in place, keeping every prompt and replacing only the
    answer, so their unit counts must equal good's exactly; a shortfall means
    the generation pass stopped early and no amount of remeasuring will fix it.
    For classes that do not rewrite good (corrupt, corruptTriple) that column is
    informational --- they are expected to differ.

The rater list is the 13 of paper3/paper4, the same set job40/makejobs40.py
submits; the generator list is whatever the class directory holds, so
generators that were never built for this class show up as an absence in the
coverage line rather than as 13 missing cells.

Run from npfl087-otmara/:
    ../v/bin/python3 ./python/auditCandidateCorrupt.py [CLASS] [OUTPUT_DIR]
"""

import re
import sys
from pathlib import Path

SEP = "-----------------------------"

CLASS = sys.argv[1] if len(sys.argv) > 1 else "candidateCorrupt"
BASE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("resources/output30")

# The 13 raters of paper3/paper4, in the paper's table order.
RATERS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b", "llama",
          "dolphin", "darwin", "ibm", "gptoss", "aya", "tower"]


def countUnits(path):
    """Units in a class file, counted the way calculatePerplexityAnswerOnly's
    getUnits() does: split on the separator, keeping the trailing block."""
    if not path.is_file():
        return None
    return len(path.read_text(encoding="utf-8", errors="replace").split(SEP))


def readPerps(path):
    """(status, values measured) for one perps file. The file is the stdout of
    calculatePerplexityAnswerOnly: a `len:` line, a `mean` line, then the list."""
    if not path.is_file():
        return ("MISSING", 0)
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return ("EMPTY", 0)
    lines = text.splitlines()
    mean = re.match(r"mean\s+(\S+)", lines[1]) if len(lines) > 1 else None
    if mean is None:
        # Died before the end: the `len:` line is there because Python flushes
        # stdout on the way out, but nothing was ever computed.
        return ("NOMEAN", 0)
    value = float(mean.group(1))
    values = re.findall(r"[-\d.eE+]+", lines[2]) if len(lines) > 2 else []
    if value != value:
        return ("NAN", len(values))
    return ("OK", len(values))


def main():
    classDir = BASE / CLASS
    if not classDir.is_dir():
        sys.exit("no such class directory: " + str(classDir))
    generators = sorted(p.stem for p in classDir.glob("*.txt"))
    if not generators:
        sys.exit("no class files in " + str(classDir))

    # output40 has no good class to compare against (its answers are WMT
    # submissions, not this pipeline's translations), so the whole comparison
    # is skipped rather than reported as every file being short.
    goodDir = BASE / "good"
    short = []
    if goodDir.is_dir():
        print("=== class files: " + CLASS + " vs good ===")
        for generator in generators:
            here = countUnits(classDir / (generator + ".txt"))
            good = countUnits(goodDir / (generator + ".txt"))
            flag = ""
            if here != good:
                short.append((generator, here, good))
                flag = "   <-- does not line up with good"
            print("  %-11s %s=%-5s good=%-5s%s" % (generator, CLASS, here, good, flag))

        built = sorted(p.stem for p in goodDir.glob("*.txt"))
        absent = [g for g in built if g not in generators]
        print("  generators with a good file but no " + CLASS + " file: "
              + (", ".join(absent) if absent else "none"))
    else:
        print("=== class files: " + CLASS + " (no good class to compare against) ===")
        for generator in generators:
            print("  %-11s %s=%s"
                  % (generator, CLASS, countUnits(classDir / (generator + ".txt"))))

    print("\n=== perps grid: %d raters x %d generators ===" % (len(RATERS), len(generators)))
    problems = []
    for rater in RATERS:
        row = []
        for generator in generators:
            path = BASE / "perps" / ("by" + rater) / CLASS / ("of" + generator + ".txt")
            status, measured = readPerps(path)
            expected = countUnits(classDir / (generator + ".txt"))
            if status != "OK":
                problems.append((rater, generator, status))
                row.append(generator + ":" + status)
            elif expected is not None and measured < expected:
                problems.append((rater, generator, "SHORT %d/%d" % (measured, expected)))
                row.append("%s:%d/%d" % (generator, measured, expected))
        print("  by%-11s %s" % (rater, ", ".join(row) if row else "-"))

    print("\n=== summary ===")
    print("  cells expected: %d   problems: %d"
          % (len(RATERS) * len(generators), len(problems)))
    for rater, generator, status in problems:
        print("    by%s / of%s: %s" % (rater, generator, status))
    for generator, here, good in short:
        print("    %s/%s.txt: %s units, good has %s -- regenerate, then remeasure "
              "of%s for every rater" % (CLASS, generator, here, good, generator))

    # of*.txt left behind by a generator that no longer has a class file: they
    # are read by nothing, but they make the directory listings misleading.
    print("\n=== perps files with no matching class file ===")
    for rater in RATERS:
        raterDir = BASE / "perps" / ("by" + rater) / CLASS
        if not raterDir.is_dir():
            print("  by%s: no %s directory" % (rater, CLASS))
            continue
        stale = [p.name for p in sorted(raterDir.glob("of*.txt"))
                 if p.stem.removeprefix("of") not in generators]
        if stale:
            print("  by%s: %s" % (rater, ", ".join(stale)))


main()
