import os

import torch

from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForCausalLM,QuantoConfig

# Darwin is the heaviest rater in the set: 34.7B parameters, i.e. 69 GB of bf16
# weights (a 256-expert MoE, so nearly all of that is expert weights that a
# single unit barely touches -- it still all has to be resident).
#
# Same two guards as getdolphin: the dtype is pinned so a checkpoint read cannot
# land in fp32, and when a GPU is present accelerate places each shard on its
# final device as it is read. Without device_map the weights land on CPU and are
# only then moved by the caller's .to(cuda), which needs the 69 GB on both
# devices at once. Without a GPU nothing changes: everything stays on CPU, which
# is how output40 was measured.
#
# One copy is still 69 GB, so a run needs mem>=80gb and cannot be done
# interactively. Set DARWIN_QUANT=int8 (~35 GB) or int4 (~17 GB) to quantize on
# the fly with optimum-quanto when that is out of reach. Quantized perplexities
# are NOT comparable to the bf16 numbers already in resources/output30 and
# resources/output40 -- switching means remeasuring every darwin file.
def getModelAndTokenizer():
    model_id = "FINAL-Bench/Darwin-36B-Opus"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=getToken())

    kwargs = {"token": getToken(), "dtype": torch.bfloat16}
    quant = os.environ.get("DARWIN_QUANT")
    if quant:
        kwargs["quantization_config"] = QuantoConfig(weights=quant)
    if torch.cuda.is_available():
        # quanto cannot quantize on the fly onto a map that spills to CPU or
        # disk, so a quantized model is pinned to the one GPU; an unquantized
        # one may be spread over GPU and CPU.
        kwargs["device_map"] = {"": 0} if quant else "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

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
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1000)
    return (tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
