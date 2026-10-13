#!/usr/bin/env python3
"""
Remove LLM special/control tokens from pipeline output files.

Usage:
  removeSpecialTokens.py <file_or_dir> [<file_or_dir> ...]

If no arguments are given, it cleans the default output30 dirs
(good/, bad/, broken/, corrupt/, corruptTriple/, candidateCorrupt/,
deprefBroken/, deprefBrokenMedium/). Directories
are processed recursively over *.txt files. Files are edited in place.

To add more tokens, just append to SPECIAL_TOKENS below.
"""
import os
import sys

# Special tokens to strip. Add more here as you find them.
SPECIAL_TOKENS = [
    "<|END_OF_TURN_TOKEN|>",
    "<|END_RESPONSE|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<|message|>",
    "<|channel|>",
    "<|eot_id|>",
    "<|return|>",
    "<|start|>",
    "<|end|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "<end_of_turn>",
    "<start_of_turn>",
    "<EOS_TOKEN>",
    "<BOS_TOKEN>",
    "<think>",
    "</think>",
    "<s>",
    "</s>",
    "<bos>",
    "<eos>",
    "<pad>",
    "<unk>",
]

DEFAULT_DIRS = [
    "resources/output30/good",
    "resources/output30/bad",
    "resources/output30/broken",
    "resources/output30/corrupt",
    "resources/output30/corruptTriple",
    "resources/output30/candidateCorrupt",
    "resources/output30/deprefBroken",
    "resources/output30/deprefBrokenMedium",
]


def clean_text(text):
    for tok in SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text


def clean_file(path):
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    cleaned = clean_text(original)
    if cleaned != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"cleaned  {path}")
    else:
        print(f"no change {path}")


def collect_files(target):
    if os.path.isdir(target):
        for root, _, files in os.walk(target):
            for name in sorted(files):
                if name.endswith(".txt"):
                    yield os.path.join(root, name)
    elif os.path.isfile(target):
        yield target
    else:
        print(f"skip (not found): {target}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DIRS
    for target in targets:
        for path in collect_files(target):
            clean_file(path)
