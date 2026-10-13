from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForCausalLM

def getModelAndTokenizer():
    model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=getToken(), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=getToken(), trust_remote_code=True)

    return model,tokenizer

def promptModel(model, tokenizer, role, prompt):
    messages = [
        {"role": role, "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
  	  messages,
	  add_generation_prompt=True,
      enable_thinking=False,
	  tokenize=True,
	  return_dict=True,
	  return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1000)
    return (tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))