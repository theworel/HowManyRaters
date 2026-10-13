import subprocess
import sys
import random

PROMPT = "prompt"
ANSWER = "answer"
MORPHO_RUN_ANALYZE_SCRIPT = "/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/bash/short_bad_md_analyze_cs.sh"
MORPHO_RUN_ANALYZE_ARGS = ""
MORPHO_RUN_GENERATE_SCRIPT = "/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/bash/short_bad_md_generate_cs.sh"
MORPHO_RUN_GENERATE_ARGS = ""
GENDERS = ['F', 'I', 'M', 'N', 'X']
NUMBER = ['D', 'P', 'S']
CASE = ['1','2','3','4','5','6','7']
PERSON = ['1','2','3']

dia = [',', '.', '!', '?', '"', ":","“" ]
nospaceafter = ['"', "„"]

class Token:
  def __init__(self, line):
    items = line.split("\t")
    self.form = items[0]
    self.lemma = items[1]
    self.tag = items[2].strip()
  def write(self):
    if self.tag[0]!="Z":
      print(self.lemma + "\t" + self.tag)
  def toStr(self):
    if self.tag[0]!="Z":
      return (self.lemma + "\t" + self.tag)
    return ""


def runScript(scriptfile, args, input_string): 
  p = subprocess.Popen(scriptfile, shell=True, stdout = subprocess.PIPE, stdin = subprocess.PIPE)
  output,_ = p.communicate(input = input_string.encode())
  decoded = output.decode()

  return (decoded)

def analyzeLine(line):
  l = runScript(MORPHO_RUN_ANALYZE_SCRIPT, MORPHO_RUN_ANALYZE_ARGS, line)
  return l

def generateLine(line, original, missing):
  l = runScript(MORPHO_RUN_GENERATE_SCRIPT, MORPHO_RUN_GENERATE_ARGS, line)
  if (len(missing) > 0):
    split = l.split("\n")
    result = ""
    j = 0
    for i in range(len(original)):
      next = ""
      if i in missing:
        if (i+1) in missing:
          next = original[i+1].split("\t")[0].strip()
        else:
          next = split[j].split("\t")[0].strip()
          if next == "" and (i + 1) < len(original):
            next = original[i+1].split("\t")[0].strip()
        character = original[i].split("\t")[0].strip()
        result += character
        if not (next in dia or character in nospaceafter):
          result += " "
      else:
        if (i+1) in missing:
          next = original[i+1].split("\t")[0].strip()
        else:
          next = split[j+1].split("\t")[0].strip()
          if next == "" and (i + 1) < len(original):
            next = original[i+1].split("\t")[0].strip()
        character = split[j].split("\t")[0].strip()
        if (character == ""):
          character = original[i].split("\t")[0].strip() 
          result += character
          if not (next in dia or character in nospaceafter):
            result += " "
        else:
          result += character
          if not (next in dia or character in nospaceafter):
            result += " "
        j+=1
    return result[:-1]
  return l

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
    copy.remove(original)
  return random.choice(copy)

def strReplace(original, index, new):
  return original[:index] + new + original[index + 1:]

def breakNounCase(nounToken):
  nounToken.tag = strReplace(nounToken.tag, 4, pickValue(nounToken.tag[4], CASE))

def breakVerbPerson(verbToken):
  verbToken.tag = strReplace(verbToken.tag, 7, pickValue(verbToken.tag[9], PERSON))

def breakVerbGender(verbToken):
  verbToken.tag = strReplace(verbToken.tag, 3, pickValue(verbToken.tag[3], GENDERS))

def breakAdjGender(adjToken):
  adjToken.tag = strReplace(adjToken.tag, 3, pickValue(adjToken.tag[3], GENDERS))

breakerFunctions = [
  CorruptingFunction(["VB", "Vc", "Vi", "Vt"], breakVerbPerson),
  CorruptingFunction(["VB", "Vc", "Vi", "Vt"], breakVerbGender),
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

def breakTags(sentence):
  success = False
  for breakerFunction in breakerFunctions:
    sentence, ssuccess = tryApply(sentence, breakerFunction)
    success = success or ssuccess
  if (success):
    return sentence
  print("nothing worked", file=sys.stderr)
  for token in sentence:
    print(token.form, file = sys.stderr)
    print(token.tag, file = sys.stderr)
  exit()

def prepForGenerate(line):
  l = ""
  transformedTokens = []
  original = []
  for token in line.split("\n"):
    if (token.strip() == ""):
      continue
    t = Token(token)
    original.append(token)
    transformedTokens.append(t)
  transformedTokens = breakTags(transformedTokens)
  out = ""
  missing = []
  i = 0
  for token in transformedTokens:
    tokenStr = token.toStr()
    if(len(tokenStr) > 0):
      out += tokenStr + "\n"
    else:
      missing.append(0 + i)
    i+=1
  return out, original, missing

def fixLine(line):
  result = ""
  for l in line.split("\n"):
    l= l.strip()
    if(len(l) > 0):
      result += l.split("\t")[0] + " "
  return result

def breakLine(line):
  l = analyzeLine(line)
  l, original, missing = prepForGenerate(l)
  l = generateLine(l, original, missing)
  return l



start = int(sys.argv[1])
end = int(sys.argv[2])

for i, line in enumerate(sys.stdin):
  if i < start:
    continue
  if i >= end:
    break
  if line.strip() == "":
    continue
  parts = line.split("\t")
  wrong = breakLine(parts[3])

  print(parts[0] + "\t" + parts[1] + "\t" + parts[2] + "\t" + wrong, file=sys.stderr)
  print(parts[0] + "\t" + parts[1] + "\t" + parts[2] + "\t" + wrong)