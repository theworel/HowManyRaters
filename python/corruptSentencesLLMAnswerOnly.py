"""Build the candidateCorrupt class: a model LLM-corrupts its OWN translations.

The `corrupt` class is each model's corruption of the *gold* Czech references
(corruptSentencesLLM.py over resources/fullimage.txt). `candidateCorrupt` is the
same corruption -- same corrupt_cs.txt few-shot, same prompt construction --
applied instead to the model's own candidate translations, i.e. its `good`
class.

Structurally this is the LLM twin of breakSentencesWithMD_cs_answerOnly.py
(which builds `broken` by morphological corruption): the good-class file is
streamed block by block, every prompt is copied verbatim and only the answer is
replaced by the model's corrupted rewrite of it. So the output lines up block
for block with good/ and broken/, the English source in each prompt still
matches the sentence in the answer, and no separateAnswers pass is needed --
unlike the gold-reference route, which has to re-attach the source by line
index.

Usage (reads the good-class file on stdin, writes the class file on stdout):
  ../v/bin/python3.11 ./python/corruptSentencesLLMAnswerOnly.py <model> \
      < ./resources/output30/good/<model>.txt \
      > ./resources/output30/candidateCorrupt/<model>.txt

or via the wrapper, which also post-processes the result:
  bash ./bash/candidateCorrupt.sh <model>
"""
from modelMap import model_map, prompt_method_map
from removeSpecialTokens import clean_text
import re
import sys
import torch

PROMPT = "prompt"
ANSWER = "answer"
SEPARATOR = "-----------------------------"

# Labels the few-shot teaches the model to emit; it echoes them back into the
# answer often enough to be worth stripping. "cs:" shows up because the
# good-class answers it is rewriting carry it.
LABEL = re.compile(r'^\s*(wrong|correct|cs)\s*:\s*', re.IGNORECASE)

modelName = sys.argv[1]
model, tokenizer = model_map[modelName]()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda:", torch.cuda.is_available(), file=sys.stderr)
if(torch.cuda.is_available()):
    torch.backends.cuda.enable_cudnn_sdp(False)
model.to(device)
model.eval()


def promptModel(role, prompt):
  return prompt_method_map[modelName](model, tokenizer, role, prompt)


def getCorruptFewShot():
  with open("./resources/prompts/corrupt_cs.txt") as promptFile:
    result = promptFile.read()
  return result


fewShot = getCorruptFewShot()


def normalizeSource(line):
  """Strip the wrapping quotes and any echoed label off a good-class answer.

  corruptSentencesLLM.py strips quotes off the gold line, so `correct:` gets a
  bare sentence. The good-class answers this reads instead are messier -- several
  models wrapped their translation in a stray cs: label as well -- so the label
  comes off too, and the quotes twice, to arrive at the same shape."""
  line = line.strip().strip("\"\'")
  line = LABEL.sub("", line)
  return line.strip().strip("\"\'")


def createFewShotPrompt(sourceText):
  return fewShot + "correct: " + normalizeSource(sourceText) + "\"\n" + "wrong:\n";


def cleanAnswer(answer):
  """Reduce the model's reply to the single corrupted sentence.

  Thinking models prefix a reasoning block, so keep only what follows the last
  </think>; then take the first line that survives special-token and label
  stripping. Returns "" when the model gave nothing usable -- the caller emits
  an empty answer rather than falling back to the uncorrupted sentence, which
  would put a good translation into a bad class."""
  if "</think>" in answer:
    answer = answer.rsplit("</think>", 1)[1]
  answer = clean_text(answer)
  for line in answer.splitlines():
    line = LABEL.sub("", line).strip()
    if line:
      return line
  return ""


def corruptLine(line):
  prompt = createFewShotPrompt(line)
  print(prompt, file=sys.stderr)
  answer = promptModel("user", prompt)
  print(answer, file=sys.stderr)
  cleaned = cleanAnswer(answer)
  if cleaned == "":
    print("empty corruption for: " + line.strip(), file=sys.stderr)
  return cleaned


state = PROMPT
for line in sys.stdin:
  if line.strip() == "":
    continue
  if (line.strip() == "answer:" and state == PROMPT):
    state = ANSWER
    print(line, end="")
    continue
  if (line.strip() == SEPARATOR):
    state = PROMPT
    print(line, end="", flush=True)
    continue
  if state == PROMPT:
    print(line, end="")
  if state == ANSWER:
    print(corruptLine(line))
