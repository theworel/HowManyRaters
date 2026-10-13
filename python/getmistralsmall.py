from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForCausalLM

def getModelAndTokenizer():
    model_id = "llmat/Mistral-Small-24B-Instruct-2501-NVFP4"
    tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True, token=getToken())

    return model,tokenizer

def promptModel(model, tokenizer, role, prompt):
  pass
  # messages = [
  #     {"role": role, "content": prompt},
  # ]
  # sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)
  # prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
  # outputs = model.generate(prompts, sampling_params)
  # return outputs[0].outputs[0].text


