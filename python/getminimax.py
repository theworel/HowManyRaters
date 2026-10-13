from hftoken import getToken
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def getModelAndTokenizer():
    model_id = "MiniMaxAI/MiniMax-M2.7"
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
      tokenize=True,
      return_dict=True,
      return_tensors="pt",
      max_new_tokens=2000,
      max_length=2500
    ).to(torch.device("cuda"))

    model.to(torch.device("cuda"))
    outputs = model.generate(**inputs, max_new_tokens=2000)
    return (tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))