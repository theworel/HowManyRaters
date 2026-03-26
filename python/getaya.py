from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereLabs/tiny-aya-global"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def getModelAndTokenizer():
    model_id = "CohereLabs/tiny-aya-global"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    return model,tokenizer

