from modelMap import model_map
import sys
from transformers import pipeline


model, tokenizer = model_map[sys.argv[1]]()
filenameCz = sys.argv[2]
filenameEn= sys.argv[3]
filenameToTranslate = sys.argv[4]
srclang = sys.argv[5]
targetlang = sys.argv[6]

def promptModel(role, prompt):
  messages = [
      {"role": role, "content": prompt},

  ]
  inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
  ).to(model.device)

  outputs = model.generate(**inputs)
  return (tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


def getGoodTranslationsFewShot(sourcelang, targetlang, shots = 5):
  def getLine(file):
    line = file.readline()
    return line[line.find("\t")+1:]
  with open(filenameCz) as csFile, open(filenameEn) as enFile:
    result = ""
    if (sourcelang == "cs" and targetlang == "en"):
      for _ in range(shots):
        result += "cs: " + getLine(csFile) + "\n" + "en: " + getLine(enFile) + "\n" 
    elif (sourcelang == "en" and targetlang == "cs"):
      for _ in range(shots):
        result += "en: " + getLine(enFile) + "\n" + "cs: " + getLine(csFile) + "\n" 
  return result

def createFewShotPrompt(sourceText, sourcelang, targetlang):
  return getGoodTranslationsFewShot(sourcelang, targetlang) + sourcelang + ": " + sourceText + "\n" + targetlang + ":"; 

def getTranslations(sourceSentencesFile, sourcelang, targetlang):
  translations = []
  with open(sourceSentencesFile) as srcFile:
    for line in srcFile.readlines():
      prompt = createFewShotPrompt(line, sourcelang, targetlang)
      t = promptModel("user", prompt)
      translations.append(t)
  return translations

translations = getTranslations(filenameToTranslate, srclang, targetlang)
for translation in translations:
  print(translation)