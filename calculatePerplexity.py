import getaya
import geteu
from modelMap import model_map
import sys
import torch

def compute_perplexity_for_batch(input_texts, model_id):
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



