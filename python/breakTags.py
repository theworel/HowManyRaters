import sys
import random


GENDERS = [F, I, M, N, X]
NUMBER = [D, P, S]
CASE = [1,2,3,4,5,6,7]
PERSON = [1,2,3]


class Token:
  def __init__(self, line):
    print(line, file=sys.stderr)
    items = line.split("\t")
    self.form = items[0]
    self.lemma = items[1]
    self.tag = items[2].strip()
  def write(self):
    if self.tag[0]!="Z":
      print(self.lemma + "\t" + self.tag)

class CorruptingFunction:
  def __init__(self, wantedPrefixes, funcItself):
    self.wantedPrefixes = wantedPrefixes
    self.funcItself = funcItself
  def matches(self, candidateToken):
    for prefix in self.wantedPrefixes:
      if candidateToken.tag.startswith(prefix):
        return True
    return False
  def apply(self, token):
    self.funcItself(token)


def pickValue(original, values):
  copy = values.copy()
  if original in values:
    values.remove(original)
  return random.choose(copy)

def breakNounCase(nounToken):
  nounToken.tag[4] = pickValue(nounToken.tag[4], CASE)

def breakVerbPerson(verbToken):
  verbToken.tag[9] = pickValue(verbToken.tag[9], PERSON)

def breakVerbGender(verbToken):
  verbToken.tag[3] = pickValue(verbToken[3], GENDERS)

def breakAdjGender(adjToken):
  adjToken.tag[3] = pickValue(adjToken[3], GENDERS)

breakerFunctions = [
  CorruptingFunction(["VB, Vc, Vi, Vt"], breakVerbPerson),
  CorruptingFunction(["VB, Vc, Vi, Vt"], breakVerbGender),
  CorruptingFunction(["A"], breakAdjGender),
  CorruptingFunction(["N"], breakNounCase),
]
def tryApply(sentence, breakerFunction):
  newSentence = []
  matched = False
  for token in sentence:
    if (breakerFunction.matches(token)):
      breakerFunction.apply(token)
      matched = True
    newSentence.append(token)
  return newSentence, matched
def breakTags(sentences):
  newSentences = []
  for sentence in sentences:
    success = False
    for breakerFunction in breakerFunctions:
      sentence, success = tryApply(sentence, breakerFunction)
      if (success):
        newSentences.append(sentence)
        break 
    if not success:
      exit()
  return newSentences
sentences = []

currentSentence = []
for line in sys.stdin:
  if line != "\n":
    currentSentence.append(Token(line))
  else:
    sentences.append(currentSentence)
    currentSentence = []

sentences = breakTags(sentences)

for sentence in sentences:
  for token in sentence:
    token.write()
  print("")




