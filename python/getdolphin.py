
import os

import torch

from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForImageTextToText,QuantoConfig

# Dolphin is Mistral-Small-3.1-24B shaped: 24.0B parameters, i.e. 48 GB of bf16
# weights. That is the whole memory story of this model -- the units are short,
# so logits and activations are noise next to it.
#
# Two things keep the peak at one copy of those 48 GB. The dtype is pinned to
# bfloat16 so a checkpoint read cannot land in fp32 (96 GB), and when a GPU is
# present accelerate is asked to place each shard on its final device as it is
# read. Without device_map the weights land on CPU first and are only then moved
# by the caller's .to(cuda), which needs the 48 GB on both devices at once.
# Without a GPU nothing changes from before: everything stays on CPU, which is
# how all of output40 and deprefGood were measured.
#
# Even so, one copy is 48 GB, so a run still needs a job with mem>=64gb -- an
# interactive/frontend run cannot hold this model. Set DOLPHIN_QUANT=int8 (~24
# GB) or int4 (~12 GB) to quantize the weights on the fly with optimum-quanto
# when that much memory is not available. Quantized perplexities are NOT
# comparable to the bf16 numbers already in resources/output30 and
# resources/output40 -- switching means remeasuring every dolphin file.
def getModelAndTokenizer():
    model_id = "dphn/Dolphin-Mistral-24B-Venice-Edition"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=getToken(), fix_mistral_regex=True)

    kwargs = {"token": getToken(), "dtype": torch.bfloat16}
    quant = os.environ.get("DOLPHIN_QUANT")
    if quant:
        kwargs["quantization_config"] = QuantoConfig(weights=quant)
    if torch.cuda.is_available():
        # quanto cannot quantize on the fly onto a map that spills to CPU or
        # disk, so a quantized model is pinned to the one GPU; an unquantized
        # one may be spread over GPU and CPU.
        kwargs["device_map"] = {"": 0} if quant else "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    return model,tokenizer

def promptModel(model, tokenizer, role, prompt):
    messages = [
        {"role": role, "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=True,
      return_dict=True,
      return_tensors="pt",
      max_new_tokens=2000,
      max_length=2500
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=2000)
    return (tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
