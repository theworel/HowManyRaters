from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForCausalLM

def getModelAndTokenizer():
    model_id = "CohereForAI/c4ai-command-r7b-12-2024"
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
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1000)
    return (tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
