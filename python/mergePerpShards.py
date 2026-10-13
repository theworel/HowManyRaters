"""Concatenate the shard files of one sharded perplexity measurement into the
single perps file the undivided run would have written.

calculatePerplexityAnswerOnly.py --shard=I/N measures a contiguous block of a
class file's units and writes the usual three-line perps file for just that
block. Each value is final: units are scored strictly one at a time, with no
batching and therefore no padding interaction between them, so a unit's
perplexity is a function of that unit and the rater alone. Merging is
concatenation in shard order, not a re-computation or an average of partial
results, and the result is bit-identical to an undivided run --- same values,
same order, and a mean summed in the same order.

Contiguous blocks are what make that true. A unit whose answer tokenizes to
nothing is dropped from the output with no record of its position, so the
array cannot be indexed back to units; blocks abut regardless, while a strided
split could not be put back in order.

The checks are as much the point as the concatenation. A measurement spread
over twenty jobs fails in ways one job does not: a shard can be evicted from
the queue, or die on its own, or be re-run against a different rater or a
rebuilt class file while its siblings hold the old numbers. So this refuses to
write unless the shards form a complete 0..N-1 set, agree on rater and class
file, and account between them for exactly as many units as that file holds.

Usage, from npfl087-otmara/:
    ../v/bin/python3.11 ./python/mergePerpShards.py SHARD_DIR [OUT_TXT]

SHARD_DIR is the `of<stem>_shards` directory the jobs write into; the default
output is the `of<stem>.txt` beside it, which is where every reader of these
files looks.
"""

import json
import re
import shutil
import sys
from pathlib import Path

SEP = "-----------------------------"
SHARD_RE = re.compile(r"^shard(\d+)of(\d+)\.txt$")


def readPerps(path):
    """(attempted, values) from one three-line perps file.

    The first line is how many units the run was given, the third is the
    perplexities of those that survived the zero-answer-token guard; the two
    differ exactly when something was dropped, which is why both are read."""
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        sys.exit(f"{path.name} has {len(lines)} lines, not the 3 of a perps "
                 "file --- the job died before it finished writing")
    if not lines[0].startswith("len:"):
        sys.exit(f"{path.name} does not start with a 'len:' line")
    values = lines[2].strip()
    if not (values.startswith("[") and values.endswith("]")):
        sys.exit(f"{path.name} line 3 is not a perplexity array --- the job "
                 "was probably killed mid-write")
    try:
        attempted = int(lines[0].split(":", 1)[1])
        parsed = [float(v) for v in values[1:-1].split(",") if v.strip()]
    except ValueError as exc:
        sys.exit(f"{path.name}: {exc}")
    return attempted, parsed


def checkMeta(byIndex):
    """Every shard's .meta against every other's; returns the one they agree on.

    The file name says which slice a shard is, but nothing in the perps format
    says which rater measured it or which class file it came from. Two raters'
    shards in one directory would merge into a table that looks fine and is
    half one measurement and half another. This is the check that catches it."""
    configs = {}
    for index, path in sorted(byIndex.items()):
        metaPath = path.with_suffix(".meta")
        if not metaPath.exists():
            sys.exit(f"{metaPath.name} is missing --- {path.name} was written "
                     "by a job that did not record its configuration, so there "
                     "is no way to tell whether it belongs with the rest")
        meta = json.loads(metaPath.read_text())
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
                     "one measurement")

    print(f"  rater={first['rater']}  class={first['class']}  "
          f"units={first['units']}  file={first['unitFile']}")
    return first


def readShards(directory):
    """(meta, [(attempted, values), ...] in shard order) for one directory."""
    files = {}
    for path in sorted(directory.glob("*.txt")):
        m = SHARD_RE.match(path.name)
        if m is None:
            print(f"ignoring {path.name}: not a shard file", file=sys.stderr)
            continue
        index, count = int(m.group(1)), int(m.group(2))
        files.setdefault(count, {})[index] = path

    if not files:
        sys.exit(f"no shardIIofNN.txt files in {directory}")
    if len(files) > 1:
        sys.exit(f"{directory} mixes measurements sharded {sorted(files)} ways "
                 "--- keep one measurement per directory")

    count, byIndex = next(iter(files.items()))
    missing = [i for i in range(count) if i not in byIndex]
    if missing:
        sys.exit(f"{len(missing)} of {count} shards missing: {missing} --- "
                 "re-run them and merge again")

    meta = checkMeta(byIndex)

    shards = []
    for index in range(count):
        attempted, values = readPerps(byIndex[index])
        dropped = attempted - len(values)
        note = f", {dropped} dropped" if dropped else ""
        print(f"  {byIndex[index].name}: {attempted} units{note}")
        shards.append((attempted, values))
    return meta, shards


def expectedSlices(total, count):
    """The unit counts calculatePerplexityAnswerOnly.py's shardOf() hands each
    shard, so the merge can check that the shards really do tile the file."""
    return [(i + 1) * total // count - i * total // count for i in range(count)]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[-4].strip())
    shardDir = Path(sys.argv[1])
    if not shardDir.is_dir():
        sys.exit(f"{shardDir} is not a directory")

    print(f"reading shards from {shardDir}")
    meta, shards = readShards(shardDir)

    total = meta["units"]
    expected = expectedSlices(total, len(shards))
    actual = [attempted for attempted, _ in shards]
    if actual != expected:
        bad = [i for i, (a, e) in enumerate(zip(actual, expected)) if a != e]
        sys.exit(f"shards {bad} measured {[actual[i] for i in bad]} units where "
                 f"the split of {total} gives {[expected[i] for i in bad]} --- "
                 "the class file changed under the jobs, or they were not all "
                 f"run with /{len(shards)}")

    values = [v for _, shardValues in shards for v in shardValues]
    dropped = total - len(values)
    mean = sum(values) / len(values) if values else float("nan")
    print(f"{len(shards)} shards, {total} units, {len(values)} scored"
          + (f", {dropped} dropped for having no answer tokens" if dropped else "")
          + f", mean {mean}")

    path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        shardDir.parent / (shardDir.name[:-len("_shards")] + ".txt")
    if path.exists():
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
        print(f"  {path.name} existed, backed up to {path.name}.bak")
    path.parent.mkdir(parents=True, exist_ok=True)
    # exactly what an undivided run prints: the unit count it was given, the
    # mean over what it scored, then the array
    path.write_text(f"len: {total}\nmean {mean}\n[{', '.join(repr(v) for v in values)}]\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
