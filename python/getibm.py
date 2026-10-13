import os

from hftoken import getToken
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig


# Granite 4.1-30b is 28.9B parameters, i.e. 58 GB of bf16 weights.
#
# Same two guards as getdolphin: the dtype is pinned so a checkpoint read cannot
# land in fp32, and when a GPU is present accelerate places each shard on its
# final device as it is read. Without device_map the weights land on CPU and are
# only then moved by the caller's .to(cuda), which needs the 58 GB on both
# devices at once. Without a GPU nothing changes: everything stays on CPU, which
# is how output40 was measured.
#
# One copy is still 58 GB, so a run needs mem>=72gb and cannot be done
# interactively. Set IBM_QUANT=int8 (~29 GB) or int4 (~15 GB) to quantize on the
# fly with optimum-quanto when that is out of reach. Quantized perplexities are
# NOT comparable to the bf16 numbers already in resources/output30 and
# resources/output40 -- switching means remeasuring every ibm file.
def getModelAndTokenizer():
    model_id = "ibm-granite/granite-4.1-30b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=getToken(), trust_remote_code=True)

    kwargs = {"token": getToken(), "trust_remote_code": True, "dtype": torch.bfloat16}
    quant = os.environ.get("IBM_QUANT")
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
    outputs = model.generate(**inputs, max_new_tokens=2000)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
