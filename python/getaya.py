from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForCausalLM

def getModelAndTokenizer():
    model_id = "CohereLabs/tiny-aya-global"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=getToken())
    model = AutoModelForCausalLM.from_pretrained(model_id, token=getToken())

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