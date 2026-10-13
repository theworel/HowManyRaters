import geteu
import getaya
import getarnir
import getmistralsmall
import getqwen
import getayabig
import getnandi
import getdolphin
import getminimax
import getdeepseek

import getnanbeige
import getnvidia
import getliquidai
import getibm
import getgemma
import getgptoss
import getdarwin
import getcfourai

import getayaexp
import getcommandr
import getgemmabig
import getllama
import getmistral7b
import getnllb
import gettower

model_map = {
  "eu": geteu.getModelAndTokenizer, 
  "aya": getaya.getModelAndTokenizer,
  "arnir": getarnir.getModelAndTokenizer,
  "mistral_small": getmistralsmall.getModelAndTokenizer,
  "ayabig": getayabig.getModelAndTokenizer,
  "dolphin": getdolphin.getModelAndTokenizer,
  "nandi": getnandi.getModelAndTokenizer,
  "minimax": getminimax.getModelAndTokenizer,
  "deepseek": getdeepseek.getModelAndTokenizer,
  "nanbeige": getnanbeige.getModelAndTokenizer,
  "nvidia": getnvidia.getModelAndTokenizer,
  "liquidai": getliquidai.getModelAndTokenizer,
  "ibm": getibm.getModelAndTokenizer,
  "gemma": getgemma.getModelAndTokenizer,
  "gptoss": getgptoss.getModelAndTokenizer,
  "darwin": getdarwin.getModelAndTokenizer,
  "cfourai": getcfourai.getModelAndTokenizer,
  "ayaexp": getayaexp.getModelAndTokenizer,
  "commandr": getcommandr.getModelAndTokenizer,
  "gemmabig": getgemmabig.getModelAndTokenizer,
  "llama": getllama.getModelAndTokenizer,
  "mistral7b": getmistral7b.getModelAndTokenizer,
  "nllb": getnllb.getModelAndTokenizer,
  "qwen": getqwen.getModelAndTokenizer,
  "tower": gettower.getModelAndTokenizer
  }

prompt_method_map = {
  "eu": geteu.promptModel,
  "aya": getaya.promptModel,
  "arnir": getarnir.promptModel,
  "mistral_small": getmistralsmall.promptModel,
  "qwen": getqwen.promptModel,
  "ayabig": getayabig.promptModel,
  "dolphin": getdolphin.promptModel,
  "nandi": getnandi.promptModel,
  "minimax": getminimax.promptModel,
  "deepseek": getdeepseek.promptModel,
  "nanbeige": getnanbeige.promptModel,
  "nvidia": getnvidia.promptModel,
  "liquidai": getliquidai.promptModel,
  "ibm": getibm.promptModel,
  "gemma": getgemma.promptModel,
  "gptoss": getgptoss.promptModel,
  "darwin": getdarwin.promptModel,
  "cfourai": getcfourai.promptModel,
  "ayaexp": getayaexp.promptModel,
  "commandr": getcommandr.promptModel,
  "gemmabig": getgemmabig.promptModel,
  "llama": getllama.promptModel,
  "mistral7b": getmistral7b.promptModel,
  "nllb": getnllb.promptModel,
  "tower": gettower.promptModel
}
