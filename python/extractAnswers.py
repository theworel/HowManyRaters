#!/usr/bin/env python3
"""
Read a pipeline output file from stdin (format: prompt:/answer:/----... blocks)
and print only the answers, prefixed with '---------' on its own line whenever
the prompt changes relative to the previous block.
"""
import sys

SEPARATOR = "-----------------------------"

def parse_blocks(text):
    blocks = text.split(SEPARATOR)
    result = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if "prompt:" not in block or "answer:" not in block:
            continue
        after_prompt = block.split("prompt:", 1)[1]
        prompt_text, answer_part = after_prompt.split("answer:", 1)
        result.append((prompt_text.strip(), answer_part.strip()))
    return result

def main():
    text = sys.stdin.read()
    blocks = parse_blocks(text)
    prev_prompt = None
    for prompt, answer in blocks:
        if prompt != prev_prompt:
            print("---------")
            prev_prompt = prompt
        print(answer)

if __name__ == "__main__":
    main()
