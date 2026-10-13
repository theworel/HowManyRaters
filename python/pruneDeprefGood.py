"""Cut the built deprefGood class down to REFS_PER_SENTENCE references per
source sentence, in the class file AND in every perplexity file already
measured off it, without re-measuring anything.

buildDeprefGood.py takes the FIRST N references of each sentence's block, so
lowering N only ever drops references off the end of a block: the surviving
units are a prefix-per-sentence subset of the ones already on disk, in the same
order. calculatePerplexityAnswerOnly.py scores units strictly one at a time
(no batching, so no padding interaction between units), which makes a unit's
perplexity a function of that unit alone. Dropping the same positions from
resources/output30/perps/by<rater>/deprefGood/ofall.txt therefore leaves
exactly the numbers a re-measurement of the shortened file would produce, and
saves re-running 13 GPU jobs to get them.

The sentence with fewer references than the target (WMT id 7, "Who is
Voldemort?", which the dataset gives only 8 distinct translations) keeps all of
them: the cut is "first N per sentence", not "exactly N per sentence".

Units are grouped by the source sentence named in their own prompt, via
sourceSentenceIds.py, rather than by counting off blocks of 44 --- the same
labelling the split-by-sentence scoring uses. The class's cached
{cls: ...} entries in sentenceMap.json/sourceIdMap.json describe the old
arrays, so they are dropped and left to rebuild on the next scoring run.

Idempotent: cutting to 39 twice keeps the same 39. Prints what it would do and
changes nothing unless --apply is given; with --apply every file it rewrites is
copied to <name>.bak first.

Run from npfl087-otmara/:
    ../v/bin/python3 ./python/pruneDeprefGood.py [REFS_PER_SENTENCE] [PERPS_DIR] [--apply]
"""

import json
import shutil
import sys
from pathlib import Path

from sourceSentenceIds import (SEP, loadSources, sourceIdsForFile, sourceKeyIndex,
                               writeAtomic)

APPLY = "--apply" in sys.argv
ARGV = [a for a in sys.argv if not a.startswith("--")]

REFS_PER_SENTENCE = int(ARGV[1]) if len(ARGV) > 1 else 39
PERPS_DIR = Path(ARGV[2]) if len(ARGV) > 2 else Path("resources/output30/perps")

CLASS = "deprefGood"
CLASS_FILE = PERPS_DIR.parent / CLASS / "all.txt"
SOURCE_FILE = Path("resources/en.txt_unchanged")
CACHES = [PERPS_DIR / "sentenceMap.json", PERPS_DIR / "sourceIdMap.json"]


def readUnits(path):
    """The units of a class file, stripped of the newlines around the
    separator so they can be written back in the builder's exact format."""
    return [u.strip("\n") for u in path.read_text(encoding="utf-8").split(SEP) if u.strip()]


def formatUnits(units):
    """The builder's file format: blocks joined by a separator on its own line,
    one trailing newline."""
    return ("\n" + SEP + "\n").join(units) + "\n"


def keepMask(ids):
    """True for the first REFS_PER_SENTENCE positions of each source sentence.

    Unlabelled positions (-1) are kept rather than quietly deleted --- there
    should be none, and if there are, the mismatch is worth seeing rather than
    burying in a rewrite."""
    seen = {}
    mask = []
    for i in ids:
        if i < 0:
            mask.append(True)
            continue
        seen[i] = seen.get(i, 0) + 1
        mask.append(seen[i] <= REFS_PER_SENTENCE)
    return mask


def readPerps(path):
    """(header lines, values) of a perplexity file, or None if it is not one."""
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        return None
    line = lines[2].strip()
    if not (line.startswith("[") and line.endswith("]")):
        return None
    try:
        return lines, [float(v) for v in line[1:-1].split(",") if v.strip()]
    except ValueError:
        return None


def writeFile(path, text):
    """Back up and overwrite, or say what would happen."""
    if not APPLY:
        print(f"  would write {path}")
        return
    shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path} (backup at {path.name}.bak)")


def main():
    units = readUnits(CLASS_FILE)
    # a class file this script cannot reproduce byte for byte is one it must
    # not rewrite: the difference would be silently attributed to the cut
    if formatUnits(units) != CLASS_FILE.read_text(encoding="utf-8"):
        sys.exit(f"{CLASS_FILE} does not round-trip through the block format, refusing to rewrite it")

    ids = sourceIdsForFile(CLASS_FILE, sourceKeyIndex(loadSources(SOURCE_FILE)))
    if len(ids) != len(units):
        sys.exit(f"{CLASS_FILE}: {len(units)} units but {len(ids)} source ids")
    mask = keepMask(ids)

    perSentence = {}
    for i, k in zip(ids, mask):
        before, after = perSentence.get(i, (0, 0))
        perSentence[i] = (before + 1, after + int(k))
    print(f"{CLASS_FILE}: {len(units)} units over {len(perSentence)} source sentences "
          f"-> {sum(mask)} at {REFS_PER_SENTENCE} references per sentence")
    for i in sorted(perSentence):
        before, after = perSentence[i]
        if before >= REFS_PER_SENTENCE:
            continue
        print(f"  source {i} keeps all {before} references (fewer than {REFS_PER_SENTENCE})")
    if sum(mask) == len(units):
        print("nothing to cut")
        return

    writeFile(CLASS_FILE, formatUnits([u for u, k in zip(units, mask) if k]))

    for path in sorted(PERPS_DIR.glob(f"by*/{CLASS}/*.txt")):
        parsed = readPerps(path)
        if parsed is None:
            print(f"  skipping unreadable {path}")
            continue
        lines, values = parsed
        if len(values) != len(units):
            print(f"  skipping {path}: {len(values)} values for {len(units)} units")
            continue
        kept = [v for v, k in zip(values, mask) if k]
        mean = sum(kept) / len(kept) if kept else float("nan")
        writeFile(path, f"len: {len(kept)}\n{lines[1].split()[0]} {mean}\n"
                        f"[{', '.join(repr(v) for v in kept)}]\n")

    for cache in CACHES:
        if not cache.exists():
            continue
        data = json.loads(cache.read_text())
        if CLASS not in data:
            continue
        print(f"  dropping {CLASS} from {cache} so it rebuilds off the new files")
        if APPLY:
            shutil.copyfile(cache, cache.with_suffix(cache.suffix + ".bak"))
            del data[CLASS]
            # atomically, like every other writer of these caches, so a job
            # reading one while this runs never sees a half-written file
            writeAtomic(data, cache)

    if not APPLY:
        print("\ndry run --- pass --apply to write")


if __name__ == "__main__":
    main()
