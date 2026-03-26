from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "utter-project/EuroLLM-9B-Instruct"

def getModelAndTokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    return model,tokenizer
