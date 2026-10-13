from modelMap import model_map, prompt_method_map
import sys


model, tokenizer = model_map[sys.argv[1]]()
filenameGold = sys.argv[2]

def promptModel(role, prompt):
  inputs = None
  return prompt_method_map[sys.argv[1]](model, tokenizer, role, prompt)

def getGoodTranslationsFewShot():
  with open("./resources/prompts/corrupt_cs3.txt") as promptFile:
    result = promptFile.read()
  return result

def createFewShotPrompt(sourceText):
  return getGoodTranslationsFewShot() + "correct: " + sourceText.strip() + "\"\n" + "wrong:\n";  

def getTranslations(sourceSentencesFile):
  translations = []
  with open(sourceSentencesFile) as srcFile:
    for line in srcFile.readlines():
      prompt = createFewShotPrompt(line)
      t = promptModel("user", prompt)
      print(prompt, file=sys.stderr)
      translations.append("prompt:\n" + prompt + "answer:\n" + t)
      print(t, file=sys.stderr)
  return translations

print("start")
translations = getTranslations(filenameGold)
for translation in translations:
  print(translation)
  print("-----------------------------")