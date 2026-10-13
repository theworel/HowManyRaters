import sys
import random


GENDERS = ['F', 'I', 'M', 'N', 'X']
NUMBER = ['D', 'P', 'S']
CASE = ['1','2','3','4','5','6','7']
PERSON = ['1','2','3']


class Token:
  def __init__(self, line):
    # print(line, file=sys.stderr)
    items = line.split("\t")
    self.form = items[0]
    self.lemma = items[1]
    self.tag = items[2].strip()
  def write(self):
    if self.tag[0]!="Z" and self.tag not in ["UNK", ",", ".", "CD"]:
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
  return random.choice(copy)

def strReplace(original, index, new):
  return original[:index] + new + original[index + 1:]

def breakVerbPerson(verbToken):
  verbToken.tag = strReplace(verbToken.tag, 7, pickValue(verbToken.tag[9], PERSON))

def breakVerbGender(verbToken):
  verbToken.tag = strReplace(verbToken.tag, 3, pickValue(verbToken.tag[3], GENDERS))

def breakAdjGender(adjToken):
  adjToken.tag = strReplace(adjToken.tag, 3, pickValue(adjToken.tag[3], GENDERS))

# breakerFunctions = [
#   CorruptingFunction(["VB", "Vc", "Vi", "Vt"], breakVerbPerson),
#   CorruptingFunction(["VB", "Vc", "Vi", "Vt"], breakVerbGender),
#   CorruptingFunction(["A"], breakAdjGender),
#   CorruptingFunction(["N"], breakNounCase),
# ]
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
      print("nothing worked", file=sys.stderr)
      for token in sentence:
        print(token.form, file = sys.stderr)
        print(token.tag, file = sys.stderr)
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

# sentences = breakTags(sentences)

for sentence in sentences:
  for token in sentence:
    token.write()
    # print(token.lemma, token.tag, file=sys.stderr)
  print("")




