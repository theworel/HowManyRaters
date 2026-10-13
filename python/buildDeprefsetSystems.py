"""Build the output40 deprefset class: one prompt/answer file per WMT system.

WMT_data/deprefset/en-cs/ holds one JSONL file per participating MT system,
each with 50 records -- one per source sentence of the deprefset test suite.
The candidate translation is in the "hypothesis" field and "source_line" is the
1-based line number in en.txt, i.e. exactly the 50 sources the rest of the
pipeline translates (en.txt and resources/en.txt_unchanged are identical).

Like buildDeprefGood.py, this reuses the prompts of a good-class file verbatim,
so an output40 unit differs from an output30 good unit only in its answer and
the two are directly comparable under calculatePerplexityAnswerOnly.py.

System names carry spaces and punctuation ("Cohere CAT+", "Qwen 3.5 - 9b"), so
each becomes a filesystem-safe slug; resources/output40/systems.tsv records the
slug -> original name mapping.

Run from npfl087-otmara/:
    ../v/bin/python3 ./python/buildDeprefsetSystems.py
"""
import glob
import json
import os
import re
import sys

# Numbered English sources; source_line indexes this file, first line = 1.
EN_FILE = "./resources/en.txt_unchanged"
JSONL_DIR = "../WMT_data/deprefset/en-cs"
# Any good-class file: the prompts are byte-identical across models, and we
# reuse them verbatim so output40 units differ from good only in the answer.
PROMPT_SOURCE = "./resources/output30/good/qwen.txt"
OUT_DIR = "./resources/output40/deprefset"
MAP_FILE = "./resources/output40/systems.tsv"

SEP = "-----------------------------"


def slugify(name):
  """Filesystem-safe short name: "Cohere CAT+" -> cohere_cat_plus."""
  s = name.lower().replace("+", " plus ")
  s = re.sub(r"[^a-z0-9]+", "_", s)
  return s.strip("_")


def getPrompts(path):
  """One prompt per source sentence, verbatim from a good-class file.

  Those files hold 50 groups of 3 units, one group per source sentence in
  en.txt_unchanged order, with the same prompt inside each group.
  """
  units = [u for u in open(path, encoding="utf-8").read().split(SEP) if u.strip()]
  if len(units) != 150:
    sys.exit("expected 150 units in " + path + ", got " + str(len(units)))
  prompts = []
  for g in range(50):
    group = [units[g * 3 + k].split("\nanswer:\n")[0].lstrip("\n") for k in range(3)]
    if not group[0] == group[1] == group[2]:
      sys.exit("prompts differ inside group " + str(g) + " of " + path)
    prompts.append(group[0])
  return prompts


def countSources(path):
  return sum(1 for line in open(path, encoding="utf-8") if line.strip())


def readSystem(path):
  """source_line -> hypothesis, for one system's JSONL file."""
  hypotheses = {}
  with open(path, encoding="utf-8") as file:
    for line in file:
      if not line.strip():
        continue
      record = json.loads(line)
      line_no = record["source_line"]
      if line_no in hypotheses:
        sys.exit("duplicate source_line " + str(line_no) + " in " + path)
      hypotheses[line_no] = record["hypothesis"]
  return hypotheses


prompts = getPrompts(PROMPT_SOURCE)
nSources = countSources(EN_FILE)
if nSources != len(prompts):
  sys.exit("en file has " + str(nSources) + " sources, prompt file has "
           + str(len(prompts)))

os.makedirs(OUT_DIR, exist_ok=True)

slugs = {}
for path in sorted(glob.glob(os.path.join(JSONL_DIR, "*.jsonl"))):
  system = os.path.basename(path)[: -len(".jsonl")]
  slug = slugify(system)
  if slug in slugs:
    sys.exit("slug collision: " + system + " and " + slugs[slug] + " -> " + slug)
  slugs[slug] = system

  hypotheses = readSystem(path)
  missing = [i for i in range(1, nSources + 1) if i not in hypotheses]
  if missing:
    print(system + ": no hypothesis for source lines "
          + ", ".join(str(i) for i in missing), file=sys.stderr)

  blocks = []
  for i in range(1, nSources + 1):
    if i not in hypotheses:
      continue
    # The hypothesis is the system's raw output and is kept verbatim; only
    # trailing whitespace goes, so the block stays well-formed. A few systems
    # emit several lines (commentary, alternative renderings) -- getUnits()
    # splits on the separator line alone, so multi-line answers are fine.
    hypothesis = hypotheses[i].rstrip()
    # Same answer scaffolding as good/ and deprefGood/, so the measured token
    # span matches those classes.
    blocks.append(prompts[i - 1] + "\nanswer:\n" + '"cs: "' + hypothesis + '""')

  outPath = os.path.join(OUT_DIR, slug + ".txt")
  with open(outPath, "w", encoding="utf-8") as file:
    file.write(("\n" + SEP + "\n").join(blocks) + "\n")
  print(slug + "\t" + system + "\t" + str(len(blocks)) + " units")

with open(MAP_FILE, "w", encoding="utf-8") as file:
  file.write("slug\tsystem\n")
  for slug in sorted(slugs):
    file.write(slug + "\t" + slugs[slug] + "\n")

print("systems: " + str(len(slugs)))
