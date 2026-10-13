"""Build the deprefGoodMedium class: deprefGood at 200 references per sentence.

The cut between deprefGood's 39 and deprefGoodBig's 1000 -- see
buildDeprefGood.py for what the class is. Only the per-sentence cap changes.

At 200, three of the fifty source sentences run out of references and
contribute everything the dataset has for them:

    id 7: 8    id 805: 81    id 583: 154

so the class is 47 x 200 + 243 = 9,643 units, about 5x deprefGood and a fifth
of deprefGoodBig. That is roughly what one shard of the two big cuts measures,
so it needs no sharding of its own: job30/deprefGoodMedium/ofall/by<rater>.sh
is a single job per rater, like deprefGood's.

As with every cut, the per-sentence blocks are shuffled by the dataset authors
and the cap is applied to the front of each, so the four classes nest: every
deprefGood unit is a deprefGoodMedium unit is a deprefGoodBig unit is a
deprefGoodBigger unit, in the same per-sentence order.

Run from npfl087-otmara/ (streams the whole 2.7 GB datafile, so give it time):
    ../v/bin/python3 ./python/buildDeprefGoodMedium.py
"""
import buildDeprefGood

REFS_PER_SENTENCE = 200
OUT_FILE = "./resources/output30/deprefGoodMedium/all.txt"

if __name__ == "__main__":
  buildDeprefGood.build(REFS_PER_SENTENCE, OUT_FILE)
