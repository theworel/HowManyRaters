from hftoken import getToken
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# NLLB-200 is an encoder-decoder (seq2seq) translation model, NOT a causal LM.
# It does not use chat templates; it translates between fixed language codes
# (e.g. "eng_Latn", "ces_Latn") via the tokenizer's src_lang and a
# forced_bos_token_id for the target language.
#
# NOTE: the causal-LM perplexity path in calculatePerplexity*.py
# (model(input_ids) -> outputs.logits with a left-shift over input_ids)
# does NOT apply to this model as-is, since seq2seq scoring needs
# decoder_input_ids / labels. Use this getter for translation generation.

SRC_LANG = "eng_Latn"
TGT_LANG = "ces_Latn"

def getModelAndTokenizer():
    model_id = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=SRC_LANG, token=getToken())
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=getToken())

    return model,tokenizer

def promptModel(model, tokenizer, role, prompt):
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
        max_new_tokens=1000,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
