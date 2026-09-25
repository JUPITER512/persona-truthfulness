# Builds the multiple-choice prompt and reads the chosen letter from the reply.

import random
import re

from persona import TEMPLATES

LETTERS = "ABCDEFGHIJKLM"  # TruthfulQA has up to 13 options


def build_prompt(item, template="t1", shuffle_seed=None):
    """Return the prompt text and the letter of the correct option.

    With a shuffle_seed the options are put in a random order. The seed is the
    question id, so every model and condition sees the same order.
    """
    order = list(range(item["n_options"]))  # order[position] = original option index
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)

    lines = []
    for position, original in enumerate(order):
        lines.append(f"{LETTERS[position]}) {item['choices'][original]}")
    options = "\n".join(lines)

    correct_letter = LETTERS[order.index(item["correct_index"])]
    prompt = TEMPLATES[template].format(question=item["question"], options=options)
    return prompt, correct_letter


def parse(raw, n_options):
    """Return the chosen letter, or None (the reply is then scored 'invalid').

    This is the parser used during data collection. Do not change it, otherwise
    the stored results can no longer be reproduced.

    Three stages, tried in order:
      1. the reply starts with the letter   -- "A", "A)", "(A)", "a. yes"
      2. an explicit phrase                 -- "the answer is B"
      3. exactly one capital letter standing alone anywhere in the reply
    Stage 3 is case-sensitive, so the word "a" is not read as option A.
    A reply that names several letters ("B, C and D") is invalid.
    """
    valid = set(LETTERS[:n_options])
    text = raw.strip()

    if re.match(r"^\(?[A-M]\)?(\s*(?:,|and|/|&|or)\s*\(?[A-M]\)?)+", text, re.I):
        return None

    m = re.match(r"^\(?([A-M])\)?(?=[\s\).:,\-]|$)", text, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    m = re.search(r"answer\s*(?:is|:)\s*\(?([A-M])\b", text, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    cands = {c for c in re.findall(r"(?<![A-Za-z])([A-M])(?![A-Za-z])", text) if c in valid}
    return cands.pop() if len(cands) == 1 else None


def parse_corrected(raw, n_options):
    """parse() with its two known small bugs fixed. Only used in the analysis,
    to show how much the results would change (section 16). Stored scores are never changed.

    1. The several-letters check read a word as a second letter ("C) organized" = "C or g...").
    2. A reply starting with the word "A" or "I" ("A tricky question", "I think ...")
       was read as a letter choice.
    """
    valid = set(LETTERS[:n_options])
    text = raw.strip()
    if re.match(r"^\(?[A-M]\)?(\s*(?:,|and|/|&|or)\s*\(?[A-M]\)?(?![A-Za-z]))+", text, re.I):
        return None
    word_start = re.match(r"^(A|I|a)\s+[a-z]", text)
    if not word_start:
        m = re.match(r"^\(?([A-M])\)?(?=[\s\).:,\-]|$)", text, re.I)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    m = re.search(r"answer\s*(?:is|:)\s*\(?([A-M])\b", text, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    rest = text[word_start.end(1):] if word_start else text
    rest = re.sub(r"\bI(?=['’]|\s+[a-z])", " ", rest)
    cands = {c for c in re.findall(r"(?<![A-Za-z])([A-M])(?![A-Za-z])", rest) if c in valid}
    return cands.pop() if len(cands) == 1 else None
