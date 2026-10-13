import re
import sys

NEW_PROMPT = """prompt:
Create a Czech translations for the given english sentence, format the answer exactly as in the examples, don't add anything else

en: "The morning was cold but sunny."
cs: "Ráno bylo chladné, ale slunečné."

en: "We waited for the train for full two hours."
cs: "Než vlak přijel, strávili jsme dvě hodiny čekáním."

en: "I have never seen such a beautiful sunset."
cs: "Tak krásný západ slunce jsem ještě neviděl."
"""

GOLD_FILE = "/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/resources/fullimageen.txt"


def process_block(block):
    # Extract all correct sentences
    correct_matches = re.findall(r'correct:\s*"(.*?)"', block, re.DOTALL)
    
    if not correct_matches:
        return None
    
    last_correct = correct_matches[-1]

    # Remove existing answer section if present
    block = re.sub(r'answer:\s*.*', '', block, flags=re.DOTALL)

    # Replace prompt section
    block = re.sub(r'prompt:\s*.*?(?=correct:)', NEW_PROMPT + "\n", block, flags=re.DOTALL)

    # Build new block
    new_block = block.strip()
    new_block += f'\nanswer:\n{last_correct}\n'

    return new_block

def extract_answer(block: str) -> str | None:
    # Find the answer section
    match = re.search(r'answer:\s*(.*)', block, re.DOTALL)
    if not match:
        return None

    answer_part = match.group(1).strip()

    # If there is a </think>, take only what comes after it
    if '</think>' in answer_part:
        answer_part = answer_part.split('</think>', 1)[1].strip()

    return answer_part

def getGoodTranslationsFewShot():
  with open("./resources/prompts/en_to_cs.txt") as promptFile:
    result = promptFile.read() + "\n"
  return result

def createFewShotPrompt(sourceText, sourcelang, targetlang):
  return getGoodTranslationsFewShot() + sourcelang + ": \"" + sourceText.strip() + "\"\n" + targetlang + ":\n";  

def process_file(input_path):
  with open(GOLD_FILE) as goldFile, open(input_path) as infile:
    inContent = infile.read().split("-----------------------------")
    goldLines = goldFile.readlines()

    processed_blocks = []
    for i in range(0, 50):
        block = inContent[i]
        goldLine = goldLines[i]
        answer = extract_answer(block)
        new_block = createFewShotPrompt(goldLine, "en", "cs") + answer
        if new_block:
            processed_blocks.append(new_block)

    # Join blocks back
    output_content = '\n-----------------------------\n'.join(processed_blocks)
    print(output_content)

if __name__ == "__main__":
    process_file(sys.argv[1])