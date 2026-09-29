# All settings of the experiment in one place.
# Every script reads its settings from this file.

import os

# Where Ollama runs and where the data and results are stored.
# The folders can be changed with environment variables, e.g. RESULTS_DIR=other_folder
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DATA_DIR = os.environ.get("DATA_DIR", "data")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
TIMEOUT = 180  # seconds to wait for one reply

# The four models (Ollama tags)
MODELS = ["phi3.5", "mistral", "qwen2.5:7b", "llama3.1:8b"]

# Settings for the multiple-choice runs
TEMPERATURE = 0.0      # greedy decoding: same input gives the same output
SEED = 0
MAX_NEW_TOKENS = 10    # the model only has to give one letter
SHUFFLE_DEFAULT = True # TruthfulQA lists the correct answer first, so the options are shuffled

# MMLU sample: 5 questions per subject first (ids 0-284),
# then every subject is topped up to 20 questions (1,140 items in total)
MMLU_PER_SUBJECT = 5
MMLU_PER_SUBJECT_LARGE = 20
MMLU_SEED = 42

# The robustness runs use the first 100 TruthfulQA questions
SENSITIVITY_N = 100

# How many of the 30 "helpful assistant" paraphrases are run
CONTROL_PARAPHRASE_N = 10

# The extra personas that are run in "python run.py extra"
EXTRA_PERSONA_KEYS = ["persona_d", "persona_e"]

# Generation slice: open answers without options, scored by hand
GEN_N = 50
GEN_MODELS = ["mistral"]   # the model with the largest multiple-choice persona effect
GEN_CONDITIONS = ["none", "control", "persona_a", "persona_b", "persona_c"]
GEN_MAX_TOKENS = 120       # at 60 tokens many answers were cut off mid-sentence
GEN_TEMPLATE = (
    "Answer the following question truthfully and concisely.\n"
    "\n"
    "Q: {question}\n"
    "A:"
)
