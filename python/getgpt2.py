from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
def getModelAndTokenizer():
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=getToken())
    model = AutoModelForCausalLM.from_pretrained(model_id, token=getToken())

    return model,tokenizer

def promptModel(model, tokenizer, role, prompt):
    messages = [
        {"role": role, "content": prompt},
    ]
    generator = pipeline("text-generation", model="gpt2")
    return generator(prompt, max_length = 2000, num_return_sequences = 1)