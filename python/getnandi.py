from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForCausalLM

def getModelAndTokenizer():
    model_id = "FrontiersMind/Nandi-Mini-150M-Tool-Calling"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=getToken())
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=getToken())

    return model,tokenizer

def promptModel(model, tokenizer, role, prompt):
    messages = [
        {"role": role, "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
      **inputs,
      max_new_tokens=500,
      do_sample=True,
      temperature=0.3,
      top_p=0.90,
      top_k=20,
      repetition_penalty=1.1,
    )
    
    generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response