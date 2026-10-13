import getaya
import geteu
from modelMap import model_map
import sys
import torch

# `--shard=I/N` measures only a contiguous slice of the file's units, so one
# class can be spread over N independent jobs and mergePerpShards.py can
# concatenate their outputs into the file one undivided run would have
# written. Added for deprefGoodBigger, whose 182,640 units do not fit a queue
# slot. The flag is filtered out of argv, so the several hundred existing
# invocations -- `<model> <file>`, no flags -- behave exactly as before.
SHARD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), None)
ARGV = [a for a in sys.argv if not a.startswith("--shard=")]

if SHARD is None:
    SHARD_INDEX, SHARD_COUNT = 0, 1
else:
    try:
        SHARD_INDEX, SHARD_COUNT = (int(x) for x in SHARD.split("/"))
    except ValueError:
        sys.exit(f"--shard={SHARD} is not of the form I/N")
    if not 0 <= SHARD_INDEX < SHARD_COUNT:
        sys.exit(f"--shard={SHARD}: need 0 <= I < N")

file = ARGV[2]

def compute_perplexity_for_batch(input_texts, model_id):
    print("len:", len(input_texts))
    model, tokenizer = model_map[model_id]()

    # Most model loaders leave everything on CPU (no device_map / .to(cuda)),

    # so move the model onto the GPU when one is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda:", torch.cuda.is_available(), file=sys.stderr)
    if (torch.cuda.is_available()):
        torch.backends.cuda.enable_cudnn_sdp(False)
    # A loader that used device_map (getdolphin) has already placed the weights,
    # possibly split over GPU and CPU. Moving them again would need a second
    # full copy -- 48 GB for dolphin -- and would break accelerate's offload
    # hooks, so feed such a model on the device its first block ended up on.
    if getattr(model, "hf_device_map", None):
        device = model.device
    else:
        model.to(device)
    model.eval()

    # Mistral-7B's tokenizer ships without a pad token, so padding=True below
    # raises ValueError. Reuse the eos token as the pad token for it.
    if model_id == "mistral7b" or model_id == "llama":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    answer_marker = "\nanswer:\n"
    # Locating the answer start by tokenizing the prompt separately is fragile:
    # byte-level BPE (Qwen) merges whitespace across the "\nanswer:\n" boundary
    # and overshoots -- it can leave zero answer tokens -> 0/0 -> nan. When the
    # tokenizer is "fast" it can report per-token character offsets, so we
    # locate the answer start within the SAME tokenization, which is robust to
    # tokenizer quirks. Slow tokenizers fall back to the separate-prompt
    # heuristic.
    use_offsets = tokenizer.is_fast

    # One unit at a time: a padded batch materializes batch x seq x vocab
    # logits at once, which OOMs on large-vocab models like gemma.
    perplexities = []
    for text in input_texts:
        if use_offsets:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                return_offsets_mapping=True,
            )
            offsets = inputs.pop("offset_mapping")
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Build a mask that is 1 only for answer token positions.
        answer_mask = torch.zeros_like(attention_mask)
        idx = text.find(answer_marker)
        if use_offsets:
            char_start = (idx + len(answer_marker)) if idx != -1 else len(text)
            for j, (s, e) in enumerate(offsets[0].tolist()):
                if s == e:  # special token
                    continue
                if s >= char_start:
                    answer_mask[0, j:] = 1
                    break
        else:
            # Tokenize the prompt part alone to find where the answer starts
            # (prompt is still full context). The zero-answer-token guard
            # below drops any unit this gets wrong.
            if idx == -1:
                prompt_part = text
            else:
                prompt_part = text[:idx + len(answer_marker)]
            toks = tokenizer(prompt_part, return_tensors="pt", truncation=True)
            answer_start = toks["input_ids"].shape[1]
            answer_mask[0, answer_start:] = 1
        answer_mask = answer_mask * attention_mask

        # Move everything the forward pass and downstream math touch onto the same
        # device as the model, otherwise the model(...) call fails on device mismatch.
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        answer_mask = answer_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:].to(shift_logits.device)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        shifted_answer_mask = answer_mask[:, 1:].to(log_probs.device, log_probs.dtype)
        target_log_probs = target_log_probs * shifted_answer_mask
        # Guard against units with zero answer tokens (would be 0/0 -> nan) and
        # drop them from the mean instead of poisoning the results.
        denom = shifted_answer_mask.sum(dim=-1)
        if denom.item() <= 0:
            continue
        negative_log_likelihood = -target_log_probs.sum(dim=-1) / denom
        perplexities.append(torch.exp(negative_log_likelihood).item())

    mean_perplexity_score = sum(perplexities) / len(perplexities) if perplexities else float("nan")

    return {
        "perplexities": perplexities,
        "mean_perplexity": mean_perplexity_score
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


def shardOf(units):
    """The contiguous slice of `units` this process is responsible for.

    Contiguous, unlike the every-N-th slices the CV scoring script cuts. There
    the enumeration is ordered by subset size, so a block of it is not a
    representative sample of the work; here every unit carries the same
    few-shot prompt and differs only in one source sentence and one answer, so
    blocks already cost about the same.

    Contiguity is also what makes the merge a plain concatenation. A unit whose
    answer tokenizes to nothing is dropped above with no record of which unit
    it was, so the output array cannot be indexed back to unit positions --- a
    strided split could not be re-interleaved afterwards, while blocks in
    shard order simply abut."""
    total = len(units)
    start = SHARD_INDEX * total // SHARD_COUNT
    stop = (SHARD_INDEX + 1) * total // SHARD_COUNT
    if SHARD_COUNT > 1:
        print(f"shard {SHARD_INDEX}/{SHARD_COUNT}: units [{start},{stop}) of {total}",
              file=sys.stderr)
    return units[start:stop]


def computePerplexityFromFile(filepath, model_id):
    with open(filepath) as file:
        perplexities = compute_perplexity_for_batch(shardOf(getUnits(file)), model_id)
    return perplexities


model_id = ARGV[1]

meanGood = 0
allGood = []

perplexities = computePerplexityFromFile(file, model_id)
meanGood += perplexities["mean_perplexity"]
allGood.extend(perplexities["perplexities"])

print("mean", meanGood)
print(allGood)
