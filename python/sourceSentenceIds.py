"""Label every generated unit with the SOURCE SENTENCE it was generated from,
so a train/dev split can be drawn over source sentences instead of positions.

The scoring scripts see only per-class arrays of perplexities, one value per
unit, and used to assume that position i means source sentence i in every
file. That assumption does not hold: the pipeline generated several candidate
translations per source sentence and then dropped some of them (empty answers,
answers in the wrong language, malformed blocks), so the surviving files run
from 50 to 191 units and position i means a different source sentence in each
of them. Splitting on positions therefore cannot separate content --- with
only 50 source sentences behind ~2,150 units per class, every source sentence
ends up on both sides of the split.

What the units do carry is the source sentence itself: each block is

    prompt:
    <few-shot examples>
    en: "<source sentence>"
    cs:
    answer:
    <output>
    -----------------------------

and the source sentence is the LAST "en: " line of the prompt. That holds for
every class --- corruptTriple and corrupt included, because
corruptToFakeRealPrompt.py rewrites their prompts into this same translation
format, and deprefGood too, because buildDeprefGood.py copies its prompts
verbatim from a good-class file --- deprefBroken keeps those same prompts and
only breaks the answer. Matching that line against
resources/en.txt_unchanged gives each unit a source id in 0..49, which is the
unit of the split.

The result, {cls: {generator_stem: [source id, ...]}}, is cached next to the
perplexity cache. Stems are the perplexity-file stems ("of<generator>") so the
map can be indexed with the same keys as buildSentencePerplexityMap.py's
output.

Only the generators the papers actually use are labelled (see
PAPER_GENERATORS); `--generators=a,b,c` overrides that. The perplexity cache
is deliberately wider than any one experiment --- it is shared, and the
scoring scripts pick their own subset out of it at read time --- but there is
nothing to gain from parsing the unit files of a generator no run reads, and
one of them (cfourai) has perplexities with no unit file behind them at all,
which produces a warning on every build.

Usage (pre-building the cache, as with buildSentencePerplexityMap.py):
    ../v/bin/python3.11 ./python/sourceSentenceIds.py [--generators=GENERATORS] [CLASSES] [PERPS_DIR] [OUT_PATH]
"""

import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

SEP = "-----------------------------"

# The thirteen models the papers generate with, which are also the thirteen
# they rate with. Everything else that has perplexities in the cache --- the
# older ayabig, gemma and cfourai runs, and arnir and liquidai before them ---
# is excluded from every experiment, so labelling their units is work whose
# result nothing reads.
#
# "all" is in the list and is not a model: deprefGood is one file of human
# reference translations rather than one file per generator, so its stem is
# "ofall", and deprefBroken --- the MorphoDiTa corruption of those same
# references --- inherits that layout. Dropping it here would leave those
# classes with valid perplexities and an empty source-id map, which is exactly
# the failure this list is otherwise tidying up after.
PAPER_GENERATORS = ["gemmabig", "ayaexp", "commandr", "eu", "qwen", "mistral7b",
                    "llama", "dolphin", "darwin", "ibm", "gptoss", "aya", "tower",
                    "all"]

# every quote style that shows up around a source sentence: the prompts quote
# it, some generators double the quotes, and the bad-class prompts wrap it in
# Czech low quotes --- none of that is part of the sentence's identity
QUOTE_CHARS = "\"'“”„‟‘’«»‹›‚"


def normalizeSentence(text):
    """Quote- and whitespace-insensitive key for matching a prompt's source
    line against a line of the source file."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if ch in QUOTE_CHARS else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip().lower()


def loadSources(path):
    """The source sentences in file order --- source id i is line i.

    en.txt_unchanged lines are "<wmt id>\\t<sentence>"; the leading id is the
    WMT11 segment id, not a contiguous index, so the line number is used
    instead and the id is dropped."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        out.append(parts[1] if len(parts) == 2 else line)
    return out


def sourceKeyIndex(sources):
    """{normalized sentence: source id} for matching prompts against."""
    byNorm = {}
    for i, sentence in enumerate(sources):
        byNorm.setdefault(normalizeSentence(sentence), i)
    return byNorm


def sourceIdsForFile(path, byNorm):
    """The source id of every unit in one class file, -1 where unmatched."""
    ids = []
    for unit in Path(path).read_text(encoding="utf-8", errors="replace").split(SEP):
        if not unit.strip():
            continue
        prompt = unit.split("\nanswer:")[0]
        found = [line[4:] for line in prompt.splitlines() if line.startswith("en: ")]
        ids.append(byNorm.get(normalizeSentence(found[-1]), -1) if found else -1)
    return ids


def unitCounts(sentenceMap, classes):
    """{cls: {stem: unit count}} read off the cached perplexity arrays.

    The count is what a unit file has to reproduce to be accepted as the
    source of a stem's perplexities."""
    out = {}
    for cls in classes:
        if not sentenceMap.get(cls):
            print(f"{cls} is not in the perplexity cache, so it has no units to label",
                  file=sys.stderr)
        byStem = {}
        for stem, byMeasurer in sentenceMap.get(cls, {}).items():
            lengths = {len(v) for v in byMeasurer.values()}
            if len(lengths) > 1:
                print(f"{cls}/{stem}: measurers disagree on length {sorted(lengths)}",
                      file=sys.stderr)
            if lengths:
                byStem[stem] = max(lengths)
        out[cls] = byStem
    return out


def buildClasses(expected, byNorm, unitDirs):
    """{cls: {stem: [source id, ...]}} for an {cls: {stem: unit count}} spec.

    A unit file is accepted only if it holds exactly as many units as the stem
    has perplexities, which is what keeps a stale file from an earlier round
    from being used to label a later one. Most stems are found in the first
    unit dir; a few (cfourai) were measured in output30 from generations that
    only exist in output20, hence the search over directories. A stem with no
    matching unit file gets no entry --- its units cannot be placed on either
    side of the split, so the scoring scripts drop them rather than guess."""
    out = {}
    for cls, byStem in expected.items():
        found = {}
        for stem in sorted(byStem):
            n = byStem[stem]
            generator = stem.removeprefix("of")
            for unitDir in unitDirs:
                path = Path(unitDir) / cls / f"{generator}.txt"
                if not path.is_file():
                    continue
                ids = sourceIdsForFile(path, byNorm)
                if len(ids) == n:
                    found[stem] = ids
                    break
                print(f"{path}: {len(ids)} units but {n} perplexities, ignoring",
                      file=sys.stderr)
            if stem not in found:
                print(f"no unit file for {cls}/{generator} ({n} perplexities): "
                      f"its units cannot be assigned to a split and will be dropped",
                      file=sys.stderr)
        out[cls] = found
    return out


def writeAtomic(data, path):
    """Write JSON via a same-directory temp file + rename, so a job reading
    the cache never sees a half-written file from a concurrent build."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def loadSourceIdMap(expected, mapPath, sourceFile, unitDirs):
    """The cached source-id map, extended on disk with any class of `expected`
    it does not already cover.

    Parsing every unit file of every class costs a few seconds and the answer
    never changes, so it is cached beside the perplexity cache; on a shared
    cluster, pre-build it once with this module's __main__ before launching
    parallel jobs so they do not race to write it."""
    mapPath = Path(mapPath)
    data = {}
    if mapPath.exists():
        with mapPath.open() as fh:
            data = json.load(fh)

    missing = [cls for cls in expected if not data.get(cls)]
    if missing:
        print(f"source id cache missing {missing}, building from {unitDirs[0]} ...",
              file=sys.stderr)
        byNorm = sourceKeyIndex(loadSources(sourceFile))
        data.update(buildClasses({c: expected[c] for c in missing}, byNorm, unitDirs))
        writeAtomic(data, mapPath)
        print(f"wrote {mapPath}", file=sys.stderr)
    return data


def sourceOrder(splitSeed, nSources):
    """The shuffled source ids every split is cut out of.

    The permutation depends on the split seed ALONE --- not on the class, not
    on how it is going to be cut --- so a source sentence lands on the same
    side everywhere at once: its candidate translations, their morphological
    corruptions, the LLM corruptions of its reference and its human references
    all move together, and nothing derived from a held-out sentence is ever
    seen in training."""
    digest = hashlib.sha256(f"{splitSeed}:sources".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.permutation(nSources)


def devSourceIds(splitSeed, nSources, nDev):
    """The held-out source sentences of a single holdout, as a set of source
    ids: the head of the permutation."""
    return {int(i) for i in sourceOrder(splitSeed, nSources)[:nDev]}


def foldSourceIds(splitSeed, nSources, nFolds):
    """The cross-validation folds, as `nFolds` disjoint sets of source ids
    covering every source sentence exactly once.

    The same permutation as devSourceIds, cut into `nFolds` blocks instead of
    a head and a tail --- 50 sentences in 10 folds is 5 sentences per fold,
    each of them dev once and train in the other nine rounds. Cutting the same
    stream means the two scripts' splits are nested rather than unrelated: at
    a given seed the first fold is a subset of the single-holdout dev set,
    which makes their numbers comparable."""
    return [{int(i) for i in block}
            for block in np.array_split(sourceOrder(splitSeed, nSources), nFolds)]


def main():
    generators = next((a.split("=", 1)[1] for a in sys.argv
                       if a.startswith("--generators=")), None)
    keep = [g.strip() for g in generators.split(",") if g.strip()] \
        if generators is not None else PAPER_GENERATORS
    argv = [a for a in sys.argv if not a.startswith("--generators=")]

    selectedClasses = [c.strip() for c in argv[1].split(",") if c.strip()] \
        if len(argv) > 1 else ["good", "bad", "broken", "corrupt", "corruptTriple",
                               "candidateCorrupt", "deprefGood", "deprefBroken"]
    perpsDir = Path(argv[2]) if len(argv) > 2 else Path("resources/output30/perps")
    outPath = Path(argv[3]) if len(argv) > 3 else perpsDir / "sourceIdMap.json"
    sourceFile = Path(argv[4]) if len(argv) > 4 else Path("resources/en.txt_unchanged")

    unitDirs = [perpsDir.parent, perpsDir.parent.parent / "output20"]
    sentenceMapPath = perpsDir / "sentenceMap.json"
    if not sentenceMapPath.exists():
        # the counts to check unit files against come from the perplexity
        # arrays, so that cache has to exist first
        sys.exit(f"{sentenceMapPath} does not exist -- run buildSentencePerplexityMap.py first")
    with sentenceMapPath.open() as fh:
        sentenceMap = json.load(fh)

    sources = loadSources(sourceFile)
    byNorm = sourceKeyIndex(sources)
    print(f"{len(sources)} source sentences, {len(byNorm)} distinct")

    expected = unitCounts(sentenceMap, selectedClasses)
    dropped = sorted({stem for byStem in expected.values() for stem in byStem
                      if stem.removeprefix("of") not in keep})
    expected = {cls: {stem: n for stem, n in byStem.items()
                      if stem.removeprefix("of") in keep}
                for cls, byStem in expected.items()}
    if dropped:
        print(f"not labelling {len(dropped)} generator(s) outside the kept set: "
              f"{', '.join(s.removeprefix('of') for s in dropped)}")

    data = buildClasses(expected, byNorm, unitDirs)

    # Keep the classes this invocation was not asked about, for the same
    # reason buildSentencePerplexityMap.py does: the argument names what to
    # refresh, and dropping the rest costs the next run a silent rescan.
    # An empty rebuild never replaces a non-empty entry --- a class absent
    # from the perplexity cache labels nothing, and overwriting good ids with
    # that is how deprefGood came to have valid perplexity files and an empty
    # source-id map.
    if outPath.exists():
        with outPath.open() as fh:
            cached = json.load(fh)
        for cls, byStem in cached.items():
            if data.get(cls) or not byStem:
                continue
            if cls in data:
                print(f"{cls}: rebuilt empty, keeping the cached "
                      f"{sum(len(v) for v in byStem.values())} units --- "
                      f"it is missing from {sentenceMapPath}, not from disk",
                      file=sys.stderr)
            data[cls] = byStem

    for cls, byStem in sorted(data.items()):
        units = sum(len(v) for v in byStem.values())
        unmapped = sum(1 for v in byStem.values() for i in v if i < 0)
        covered = len({i for v in byStem.values() for i in v if i >= 0})
        print(f"{cls}: {len(byStem)} generators, {units} units, "
              f"{covered}/{len(sources)} source sentences, {unmapped} unmapped")

    writeAtomic(data, outPath)
    print(f"wrote {outPath}")


if __name__ == "__main__":
    main()
