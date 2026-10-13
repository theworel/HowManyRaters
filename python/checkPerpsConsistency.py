import sys
from pathlib import Path

# Checks that every measuring model (perps/by<model>/) has the same
# <class>/of<genmodel>.txt files as all the others, and that the "len: N"
# line agrees across measurers for each (class, genmodel) pair.
# Usage: checkPerpsConsistency.py [perps_dir]
#   perps_dir defaults to resources/output30/perps (run from npfl087-otmara/)

DEFAULT_PERPS = "resources/output30/perps"

# excluded both as measurers (by<model>/) and as generators (of<model>.txt):
# arnir/nllb/ayabig have almost no measurements, cfourai duplicates commandr,
# liquidai is too weak at Czech
IGNORE = {"arnir", "cfourai", "nllb", "liquidai", "ayabig"}

CLASSES = {"good", "broken", "corruptTriple", "candidateCorrupt", "deprefBroken"}
# deprefBroken has the single generator stem "ofall", like deprefGood; the
# check is the same, that every measurer has that one file and agrees on its
# length. candidateCorrupt is expected to be short one generator: dolphin is
# deliberately not measured there (see jobs/runMissingCandidateCorrupt.sh).


def readLen(path):
    """Return (lens, error). lens is a list of ints from 'len: N' lines."""
    try:
        text = path.read_text()
    except OSError as e:
        return None, f"read error: {e}"
    if text.strip() == "":
        return None, "empty file"
    lens = []
    for line in text.splitlines():
        if line.startswith("len:"):
            try:
                lens.append(int(line.split(":", 1)[1].strip()))
            except ValueError:
                return None, f"unparseable len line: {line!r}"
    if not lens:
        return None, "no 'len:' line"
    return lens, None


def main():
    perps = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PERPS)
    if not perps.is_dir():
        print(f"perps dir not found: {perps}")
        sys.exit(2)

    measurers = sorted(d.name[2:] for d in perps.iterdir()
                       if d.is_dir() and d.name.startswith("by")
                       and d.name[2:] not in IGNORE)
    print(f"measurers ({len(measurers)}): {' '.join(measurers)}")
    print(f"ignored: {' '.join(sorted(IGNORE))}\n")

    # files[measurer] = set of "class/ofX.txt" relative paths
    files = {}
    for m in measurers:
        rels = set()
        for cls_dir in (perps / f"by{m}").iterdir():
            if not cls_dir.is_dir() or cls_dir.name not in CLASSES:
                continue
            for f in cls_dir.iterdir():
                if (f.name.startswith("of") and f.name.endswith(".txt")
                        and f.name[2:-4] not in IGNORE):
                    rels.add(f"{cls_dir.name}/{f.name}")
        files[m] = rels

    union = set().union(*files.values())
    discrepancies = 0

    print("=== missing files (vs union of all measurers) ===")
    for m in measurers:
        missing = sorted(union - files[m])
        if missing:
            discrepancies += len(missing)
            print(f"by{m} is missing {len(missing)}:")
            for rel in missing:
                print(f"  {rel}")
    if all(files[m] == union for m in measurers):
        print("none — all measurers have identical file sets")
    print()

    print("=== len mismatches / unreadable files ===")
    ok = 0
    for rel in sorted(union):
        lens_by_measurer = {}
        problems = []
        for m in measurers:
            if rel not in files[m]:
                continue
            lens, err = readLen(perps / f"by{m}" / rel)
            if err:
                problems.append(f"  by{m}: {err}")
                continue
            if len(lens) > 1:
                problems.append(f"  by{m}: multiple len lines: {lens}")
            lens_by_measurer[m] = lens[0]

        distinct = set(lens_by_measurer.values())
        if len(distinct) > 1:
            groups = {}
            for m, n in lens_by_measurer.items():
                groups.setdefault(n, []).append(m)
            problems.append("  len disagrees: " + "; ".join(
                f"len={n} [{' '.join(sorted(ms))}]"
                for n, ms in sorted(groups.items())))

        if problems:
            discrepancies += len(problems)
            print(f"{rel}:")
            for p in problems:
                print(p)
        else:
            ok += 1

    print(f"\n{ok}/{len(union)} (class, genmodel) pairs fully consistent, "
          f"{discrepancies} discrepancies total")
    sys.exit(1 if discrepancies else 0)


main()
