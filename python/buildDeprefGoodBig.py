"""Build the deprefGoodBig class: deprefGood at 1000 references per sentence.

Same class as deprefGood in every respect except size -- human reference
translations from the many-references dataset, prompts copied verbatim from a
good-class file, one shared resources/output30/deprefGoodBig/all.txt rather
than one file per generating model -- so see buildDeprefGood.py for what the
class is and why it exists. This only raises the per-sentence count from 39 to
1000.

The dataset is much deeper than deprefGood's 39 suggests: it holds 15.4M
references for these 50 source sentences, from 8 (WMT id 7, "Who is
Voldemort?") to 2.9M (id 109) each. 1000 is therefore a cap rather than a
quota, and eight of the fifty sentences fall short of it and contribute
everything they have:

    id 7: 8      id 583: 154    id 805: 81     id 1551: 680
    id 325: 315  id 646: 480    id 559: 425    id 1668: 351

so the class is 42 x 1000 + 2494 = 44,494 units, roughly 23x deprefGood. That
multiplies the cost of measuring it by the same factor -- one rater's job goes
from ~1,900 to ~44,500 sequential perplexity computations -- so the jobs in
job30/deprefGoodBig/ofall/ want a walltime set explicitly rather than the
queue default.

Each sentence's block in the datafile is shuffled by the dataset authors --- at
id 109, references 1, 50, 500, 1500 and 2999 differ throughout the sentence
rather than in a shared tail --- so the first 1000 are a random sample, the
same argument that licensed deprefGood's first 39. It also makes the two
classes nest: every deprefGood unit is a deprefGoodBig unit, in the same
per-sentence order.

Run from npfl087-otmara/ (streams the whole 2.7 GB datafile, so give it time):
    ../v/bin/python3 ./python/buildDeprefGoodBig.py
"""
import buildDeprefGood

REFS_PER_SENTENCE = 1000
OUT_FILE = "./resources/output30/deprefGoodBig/all.txt"

if __name__ == "__main__":
  buildDeprefGood.build(REFS_PER_SENTENCE, OUT_FILE)
