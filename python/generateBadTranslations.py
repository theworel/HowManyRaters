from modelMap import model_map
from modelMap import prompt_method_map
import sys
from transformers import pipeline


model, tokenizer = model_map[sys.argv[1]]()
filenameToTranslate = sys.argv[2]
srclang = sys.argv[3]
targetlang = sys.argv[4]
if (srclang not in ["cs", "en"]):
  print("invalid srclang:", srclang)
  print("invalid srclang:", srclang, file=sys.stderr)
if (targetlang not in ["cs", "en"]):
  print("invalid targetlang:", targetlang)
  print("invalid targetlang:", targetlang, file=sys.stderr)


def promptModel(role, prompt):
  inputs = None
  return prompt_method_map[sys.argv[1]](model, tokenizer, role, prompt)

def getGoodTranslationsFewShot(sourcelang, targetlang, shots = 5):
  result = "" 
  if (sourcelang == "cs" and targetlang == "en"):
    with open("./resources/prompts/cs_to_en_wrong.txt") as promptFile:
      result = promptFile.read() + "\n"
    #for _ in range(shots):
    #  result += "cs: " + getLine(csFile) + "\n" + "en: " + getLine(enFile) + "\n" 
  elif (sourcelang == "en" and targetlang == "cs"):
    with open("./resources/prompts/en_to_cs_wrong.txt") as promptFile:
      result = promptFile.read() + "\n"
    # for _ in range(shots):
    #   result += "en: " + getLine(enFile) + "\n" + "cs: " + getLine(csFile) + "\n" 
  return result

def createFewShotPrompt(sourceText, sourcelang, targetlang):
  return getGoodTranslationsFewShot(sourcelang, targetlang) + sourcelang + ": „" + sourceText.strip() + "\"\n" + targetlang + ":"; 

def getTranslations(sourceSentencesFile, sourcelang, targetlang):
  translations = []
  with open(sourceSentencesFile) as srcFile:
    for line in srcFile.readlines():
      prompt = createFewShotPrompt(line, sourcelang, targetlang)
      t = promptModel("user", prompt)
      print(prompt, file=sys.stderr)
      translations.append("prompt:\n" + prompt + "answer:\n" + t)
      print(t, file=sys.stderr)
  return translations

translations = getTranslations(filenameToTranslate, srclang, targetlang)
for translation in translations:
  print(translation)
  print("-----------------------------")