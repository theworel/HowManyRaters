from transformers import AutoTokenizer, AutoModelForCausalLM
from hftoken import getToken

model_id = "utter-project/EuroLLM-9B-Instruct"

def getModelAndTokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=getToken())
    model = AutoModelForCausalLM.from_pretrained(model_id, token=getToken())

    return model,tokenizer
