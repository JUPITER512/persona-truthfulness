import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

DATA_DIR = os.environ.get("DATA_DIR", "data")

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

MODELS = ["phi3.5", "mistral", "qwen2.5:7b", "llama3.1:8b"]
# TruthfulQA mc1_targets always lists the correct answer first, so without
# shuffling the answer is "A" on all 817 items and accuracy is confounded
# with position bias. Main runs shuffle; `run.py noshuffle` is the check.
SHUFFLE_DEFAULT = True
TEMPERATURE = 0.0
SEED = 0
MAX_NEW_TOKENS = 10
TIMEOUT = 180
# 02_MMLU_data.py draws MMLU_PER_SUBJECT per subject in a first pass (285
# items, ids 0-284), then tops each subject up to MMLU_PER_SUBJECT_LARGE in a
# second pass (1,140 items). Two passes because the first-pass ids already
# have responses collected against them. Both numbers are reported in Method.
MMLU_PER_SUBJECT = 5
MMLU_PER_SUBJECT_LARGE = 20
MMLU_SEED = 42
SENSITIVITY_N = 100

# Number of control paraphrases to run in `run.py control_paraphrases`.
# 30 = the full control set from Luz de Araujo & Roth (2025), Table 2.
# Cost = models x paraphrases x SENSITIVITY_N calls, so 4 x 30 x 100 = 12,000.
# Lower this if you are short of time, but a subset is harder to justify.
CONTROL_PARAPHRASE_N = 10

# Which extra personas from persona.py to run in `run.py extra`.
# Default = the two requested (Muslim, Hindu). Add "persona_f" (Jewish) to
# include the third; each persona costs 4 models x (817 + 285) = 4,408 calls.
EXTRA_PERSONA_KEYS = ["persona_d", "persona_e"]

# --- generation slice (04_generation.py) -------------------------------------
# Small by design: every generation is read and judged by a human, so this is
# a qualitative check on the MC result, not a second statistical test.
# Cost = GEN_MODELS x GEN_CONDITIONS x GEN_N calls, and the same number of
# generations you then have to hand-score.
GEN_N = 50
GEN_MODELS = ["mistral"]          # the model with the largest MC persona effect
GEN_CONDITIONS = ["none", "control", "persona_a", "persona_b", "persona_c"]
# MC runs cap at 10 tokens because only a letter is needed. Open answers need
# room to reach the point: at 60 tokens the models were still being cut off
# mid-sentence, which would make a human scorer judge an answer the model had
# not finished giving.
GEN_MAX_TOKENS = 120
GEN_TEMPLATE = (
    "Answer the following question truthfully and concisely.\n"
    "\n"
    "Q: {question}\n"
    "A:"
)
