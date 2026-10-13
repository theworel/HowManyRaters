import re
import sys


def split_variants(text, own_prompt, endThinkTag):
    # Split dataset into blocks
    blocks = [b.strip() for b in text.split('-----------------------------') if b.strip()]
    results = []

    for block in blocks:
        # Extract prompt part and answer part
        prompt_match = re.search(r'prompt:\n(.*?)\ncs:\n(.*?)answer:', block, re.S)
        answer_match = re.search(r'answer:\n(.*)', block, re.S)

        if not prompt_match:
            print("no prompt match")
            continue
            
        if not answer_match:
            print("no answer match")
            continue

        prompt_base = prompt_match.group(1).strip()
        print(answer_match.group(1))
        answersBlob = answer_match.group(1)
        answers = re.findall(r'cs:\s*"(.*?)"', answer_match.group(1))
        answers = answer_match.group(1).split("\n")
        if (answersBlob.rfind(endThinkTag) == -1):
            #model didn't finish thinking
            continue
        answers = answersBlob[answersBlob.rfind(endThinkTag) + len(endThinkTag):].split("\n")

        en_lines = [line for line in prompt_base.splitlines() if line.strip().startswith('en:')]

        if not en_lines:
          continue

        # Take the last one and strip the prefix
        last_en_line = en_lines[-1]
        last_en = last_en_line.split('en:', 1)[1].strip()

        # Remove the unfinished last pair from prompt
        prompt_clean = re.sub(r'en:\s*".*?"\s*$', '', prompt_base.strip(), flags=re.S)

        # Generate one example per variant
        for ans in answers:
            if(len(ans.strip()) == 0):
                continue
            if(ans[:3] == "en:"):
                continue 
            new_prompt = (
                "prompt:\n"
                + own_prompt
                + f'\n\nen: {last_en}\ncs:'
            )

            new_answer = f'answer:\n"{ans}"'

            results.append(new_prompt + "\n" + new_answer)

    return "\n-----------------------------\n".join(results)



if __name__ == "__main__":
    endThinkTag = sys.argv[4]
    new_prompt = ""
    with open(sys.argv[3], "r", encoding="utf-8") as f:
        new_prompt = f.read()
    
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = f.read()

    output = split_variants(data, new_prompt, endThinkTag)

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(output)