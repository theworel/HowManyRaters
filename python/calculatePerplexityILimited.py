import getaya
import geteu
from modelMap import model_map
import sys
import torch

file = sys.argv[2]
start = int(sys.argv[3])
end = int(sys.argv[4])

def compute_perplexity_for_batch(input_texts, model_id):
    print("len:", len(input_texts))
    model, tokenizer = model_map[model_id]()
    perplexities = []

    # one unit at a time: a padded batch materializes batch x seq x vocab
    # logits at once, which OOMs on large-vocab models like gemma
    for text in input_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:].to(shift_logits.device)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.device, log_probs.dtype)
        negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].to(log_probs.device).sum(dim=-1)
        perplexities.append(torch.exp(negative_log_likelihood).item())

    mean_perplexity_score = sum(perplexities) / len(perplexities) if perplexities else float("nan")

    return {
        "perplexities": perplexities,
        "mean_perplexity": mean_perplexity_score
    }

def getUnits(file, start, end):
    units = []
    unit = ""
    for line in file.readlines():
        if (line.strip() == "-----------------------------"):
           units.append(unit)
           unit = ""
           continue
        unit += line
    units.append(unit)
    return units[start:end]


def computePerplexityFromFile(filepath, model_id, start, end):
    with open(filepath) as file:
        perplexities = compute_perplexity_for_batch(getUnits(file, start, end), model_id)
    return perplexities


model_id = sys.argv[1]

meanGood = 0
allGood = []

perplexities = computePerplexityFromFile(file, model_id, start, end)
meanGood += perplexities["mean_perplexity"]
allGood.extend(perplexities["perplexities"])

print("mean", meanGood)
print(allGood)
