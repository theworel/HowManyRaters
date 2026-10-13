"""Build the deprefGoodBigger class: deprefGood at 5000 references per sentence.

The third and largest cut of the same class -- see buildDeprefGood.py for what
deprefGood is and buildDeprefGoodBig.py for the 1000-reference cut. Only the
per-sentence cap changes.

At 5000, nineteen of the fifty source sentences run out of references and
contribute everything the dataset has for them:

    id 7: 8       id 583: 154   id 1121: 1944   id 679: 3800
    id 805: 81    id 325: 315   id 1725: 2226   id 232: 4700
    id 1668: 351  id 559: 425   id 535: 2336
    id 646: 480   id 1551: 680  id 1893: 2592
    id 1539: 1024 id 370: 1068  id 1070: 2600
    id 1244: 1204 id 1836: 1652

so the class is 31 x 5000 + 27,640 = 182,640 units: 4.1x deprefGoodBig and 95x
deprefGood.

That last number is the point to weigh before measuring it.
calculatePerplexityAnswerOnly.py scores units strictly one at a time, so one
rater's job is 182,640 sequential forward passes, and thirteen raters is 2.4M.
The jobs in job30/deprefGoodBigger/ofall/ need an explicitly raised walltime,
and quite possibly splitting -- there is no sharding support in the
measurement script, so a job that outruns its walltime loses everything it
computed rather than resuming.

As with the smaller cuts, the per-sentence blocks are shuffled by the dataset
authors and the cap is applied to the front of each, so the three classes nest:
every deprefGood unit is a deprefGoodBig unit is a deprefGoodBigger unit, in
the same per-sentence order.

Run from npfl087-otmara/ (streams the whole 2.7 GB datafile, so give it time):
    ../v/bin/python3 ./python/buildDeprefGoodBigger.py
"""
import buildDeprefGood

REFS_PER_SENTENCE = 5000
OUT_FILE = "./resources/output30/deprefGoodBigger/all.txt"

if __name__ == "__main__":
  buildDeprefGood.build(REFS_PER_SENTENCE, OUT_FILE)
