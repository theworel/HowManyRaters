from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForCausalLM

def getModelAndTokenizer():
    model_id = "Nanbeige/Nanbeige4-3B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=getToken(), trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=getToken(), trust_remote_code=True, torch_dtype='auto', device_map='auto')

    return model,tokenizer

def promptModel(model, tokenizer, role, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids.to('cuda'), max_new_tokens=50000)
    resp = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    return resp
