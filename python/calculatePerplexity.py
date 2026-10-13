import getaya
import geteu
from modelMap import model_map
import sys
import torch

GOODFILES = ["/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/resources/good/ayabig_en_cs.txt"]
BADFILERANDOM = ["/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/resources/bad/purerandom.txt"]
BADFILEBADPROMPT = ["/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/resources/bad/darwinclean.txt"]
BADFILECORRUPT = ["/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/resources/bad/cfouraicorrupted.txt"]

def compute_perplexity_for_batch(input_texts, model_id):
    print("len:", len(input_texts))
    model, tokenizer = model_map[model_id]()
    inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :] 
    shift_labels = input_ids[:, 1:] 

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)
    negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)
    perplexities = torch.exp(negative_log_likelihood)
    mean_perplexity_score = torch.mean(perplexities)

    return {
        "perplexities": perplexities.tolist(),
        "mean_perplexity": mean_perplexity_score.item()
    }

def getUnits(file):
    units = []
    unit = ""
    for line in file.readlines():
        if (line.strip() == "-----------------------------"):
           units.append(unit)
           unit = ""
           continue
        unit += line
    units.append(unit)
    return units


def computePerplexityFromFile(filepath, model_id):
    with open(filepath) as file:
        perplexities = compute_perplexity_for_batch(getUnits(file), model_id)
    return perplexities


model_id = sys.argv[1]

meanGood = 0
allGood = []

meanBad = 0
allBad = []

meanBadPrompt = 0
allBadPrompt = []

corruptBad = 0
allCorruptBad = []
for goodFile in GOODFILES:
    perplexities = computePerplexityFromFile(goodFile, model_id)
    meanGood += perplexities["mean_perplexity"]
    allGood.extend(perplexities["perplexities"])

meanGood /= len(GOODFILES)

for badFile in BADFILERANDOM:
    perplexities = computePerplexityFromFile(badFile, model_id)
    meanBad += perplexities["mean_perplexity"]
    allBad.extend(perplexities["perplexities"])

meanBad /= len(BADFILERANDOM)

for badFile in BADFILEBADPROMPT:
    perplexities = computePerplexityFromFile(badFile, model_id)
    meanBadPrompt += perplexities["mean_perplexity"]
    allBadPrompt.extend(perplexities["perplexities"])

for badFile in BADFILECORRUPT:
    perplexities = computePerplexityFromFile(badFile, model_id)
    corruptBad += perplexities["mean_perplexity"]
    allCorruptBad.extend(perplexities["perplexities"])


print("meanGood", meanGood)
print("meanBad", meanBad)
print("meanBadPrompt", meanBadPrompt)
print("meanBadCorrupt", corruptBad)
