from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForCausalLM

def getModelAndTokenizer():
  model_id = "arnir0/Tiny-LLM"
  tokenizer = AutoTokenizer.from_pretrained(model_id, token=getToken())
  model = AutoModelForCausalLM.from_pretrained(model_id, token=getToken())

  return model, tokenizer

def promptModel(model, tokenizer, role, prompt):
  inputs = tokenizer.encode(prompt, return_tensors="pt")
  outputs = model.generate(inputs, max_length=1024)
  result = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return result[len(prompt):]