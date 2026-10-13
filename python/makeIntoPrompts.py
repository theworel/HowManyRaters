import sys

lineMap = {}

brokenLinesFilePath = sys.argv[1]
sourceFilePath = sys.argv[2]
promptFilePath = sys.argv[3]

def writePromptAndAnswer(promptBase, sourceSentence, answer):
  print("prompt:")
  print(promptBase, end="")
  print("en:", sourceSentence, end="")
  print("answer:")
  print(answer)
  print("-----------------------------")

with open(brokenLinesFilePath, 'r') as brokenLinesFile, open(sourceFilePath, 'r') as sourceFile, open(promptFilePath, 'r') as promptFile:
  promptBase = promptFile.read
  for line in sourceFile:
    parts = line.split("\t")
    lineMap[parts[0]] = parts[1]
  for line in brokenLinesFile:
    parts = line.split("\t")
    original = lineMap[parts[0]]
    answer = parts[3]
    writePromptAndAnswer(promptBase, original, answer)
