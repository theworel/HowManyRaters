import re
import sys

source_en = "/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/resources/fullimageen.txt"

def split_variants(text, own_prompt, en_lines_, endThinkTag):
    # Split dataset into blocks
    blocks = [b.strip() for b in text.split('-----------------------------') if b.strip()]
    results = []

    i = 0
    for block in blocks:
        print(i)
        lastEnOwn = en_lines_[i].strip()
        # Extract prompt part and answer part
        prompt_match = re.search(r'prompt:\n(.*?)\nwrong:\n(.*?)answer:', block, re.S)
        answer_match = re.search(r'answer:\n(.*)', block, re.S)

        if not prompt_match:
            print("no prompt match")
            continue
            
        if not answer_match:
            print("no answer match")
            continue

        prompt_base = prompt_match.group(1).strip()
        answers = answer_match.group(1).split("\n")

        answersBlob = answer_match.group(1)
        if (answersBlob.rfind(endThinkTag) == -1):
            answers = answersBlob.split("\n")
        else:
            answers = answersBlob[answersBlob.rfind(endThinkTag) + len(endThinkTag):].split("\n")

        en_lines = [line for line in prompt_base.splitlines() if line.strip().startswith('correct:')]

        if not en_lines:
          i+=1
          continue

        # Take the last one and strip the prefix
        last_en_line = en_lines[-1]
        last_en = last_en_line.split('correct:', 1)[1].strip()

        # Remove the unfinished last pair from prompt
        prompt_clean = re.sub(r'en:\s*".*?"\s*$', '', prompt_base.strip(), flags=re.S)

        # Generate one example per variant
        for ans in answers:
            if(len(ans.strip()) == 0):
                continue
            new_prompt = (
                "prompt:\n"
                + own_prompt
                + f'\n\nen: {lastEnOwn}\ncs:'
            )

            new_answer = f'answer:\n"{ans.replace("wrong:", "cs:")}"'
            print(new_answer)

            results.append(new_prompt + "\n" + new_answer)
        i+=1
    return "\n-----------------------------\n".join(results)



if __name__ == "__main__":
    new_prompt = ""
    with open(sys.argv[3], "r", encoding="utf-8") as f:
        new_prompt = f.read()
    
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = f.read()

    with open(source_en, "r", encoding="utf-8") as f:
        enLines = f.read().split("\n")
    output = split_variants(data, new_prompt, enLines, sys.argv[4])

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(output)