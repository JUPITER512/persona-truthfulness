<<<<<<< HEAD
# Effects-of-Persona-Prompting
=======
<<<<<<< HEAD
<<<<<<< HEAD
# Effects-of-Persona-Prompting
=======
# Persona Conditioning and Truthfulness

Code and data for a term paper in *Advanced Topics in NLP*, Universität Trier,
summer semester 2026.

**Research question.** Does conditioning a language model on an Argyle-style
demographic backstory reduce truthfulness on imitative-falsehood questions, and
does the effect concentrate in categories where the human-plausible answer is
false?

**Design.** Four open-weight models are asked 817 TruthfulQA (MC1) questions and
1,140 MMLU questions under five conditions: no system message, a neutral
"helpful assistant" control, and three demographic backstories. MMLU is the
control dataset: it separates a specific effect on imitative falsehoods from
general degradation of knowledge. 87,393 model responses in total.

---

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Linux/macOS
pip install -r requirements.txt

ollama pull phi3.5
ollama pull mistral
ollama pull qwen2.5:7b
ollama pull llama3.1:8b

python verify.py               # checks everything before you spend hours
```

`verify.py` writes nothing and takes seconds. It checks the Python version,
packages, project files, the five frozen conditions, both datasets, that Ollama
is reachable with all four models, that one test call works, that the parser
behaves, and whether any results are incomplete or duplicated. **Fix every FAIL
before running an experiment.**

---

## Reproducing the study

Build the datasets, then run the experiments. Every run is resumable: stop with
Ctrl+C and re-run the same command, and it continues where it stopped.

```bash
python 01_Load_data.py      # TruthfulQA  -> data/truthfulqa_items.json  (817 items)
python 02_MMLU_data.py      # MMLU        -> data/mmlu_items.json  (1,140 items)
```

`02_MMLU_data.py` draws the sample in two seeded passes so that ids assigned
by the original 5-per-subject run keep their meaning; it refuses to write if an
existing id would come to mean a different question.

### Experiments

| Command | What it collects | Rows | Approx. time |
|---|---|---|---|
| `python run.py pilot` | 50 questions, sanity check | 1,000 | 15 min |
| `python run.py main` | TruthfulQA, 4 models x 5 conditions | 16,340 | ~13 h |
| `python run.py mmlu` | MMLU, 4 models x 5 conditions | 22,800 | ~19 h |
| `python run.py templates` | prompt format t1/t2/t3, 100 questions | 6,000 | ~5 h |
| `python run.py noshuffle` | unshuffled options, 100 questions | 2,000 | ~1.5 h |
| `python run.py shuffle` | shuffled repeat = noise floor | 2,000 | ~1.5 h |
| `python run.py temp` | temperature 0.7, seeds 1/2/3 | 6,000 | ~5 h |
| `python run.py control_paraphrases` | 10 paraphrases of the control prompt | 4,000 | ~3 h |
| `python run.py extra` | two extra personas, t1/t2/t3 | 26,712 | ~22 h |
| `python 04_generation.py` | 250 open-ended answers | 250 | 20 min |

Times are for an RTX 4050 (6 GB) with CPU offload; a server with more VRAM will
be considerably faster. Running `main` and `mmlu` is enough to reproduce the
headline result; the rest are sensitivity checks.

**Never run two copies of the same command at once.** There is no file lock, and
two writers produce duplicate rows.

### Analysis

```bash
python analyse.py                          # all statistics -> results/analysis/*.csv
python check_pilot.py results/pilot.jsonl  # accuracy plus 20 raw outputs to read
```

### Generation slice

```bash
python 04_generation.py                    # collect 250 open-ended answers
python 05_score_generation.py sample       # look at a few
python 05_score_generation.py sheet        # build a BLIND scoring sheet
#   ... fill the 'score' column by hand: 1 true, 0 false, x unjudgeable ...
python 05_score_generation.py merge        # join scores back and summarise
```

The sheet hides the condition and shuffles the rows, because knowing which
persona produced a sentence would bias the scorer — which is the effect this
study measures. There is no judge model and no similarity metric.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Every setting. Change experiment parameters here, not in the scripts |
| `persona.py` | The five frozen conditions, three templates, 30 control paraphrases, three extra personas |
| `backend.py` | One function: send a chat request to Ollama |
| `run.py` | The multiple-choice runner and the answer parser |
| `04_generation.py` | Open-ended generation slice |
| `05_score_generation.py` | Blind scoring sheet and merge |
| `analyse.py` | All statistics; writes CSVs for the appendix |
| `check_pilot.py` | Quick per-condition summary plus raw outputs |
| `verify.py` | Pre-flight check |

---

## How it works

**Conditions.** `none` sends no system message at all; `control` sends "You are a
helpful assistant."; the three personas send a four-sentence backstory varying
nine attributes (age, race, gender, location, education, occupation, income,
politics, religion). Only the values differ between personas, never the
structure.

**Option shuffling.** TruthfulQA's `mc1_targets` always lists the correct answer
first, so without shuffling the answer would be "A" on all 817 questions and
accuracy would be confounded with position bias. Options are shuffled using the
question's own id as the seed, so the same question is presented in the same
order under every condition — which is what makes the paired statistics valid.

**Parsing.** Three stages: the reply opens with the letter; or says "the answer
is X"; or contains exactly one standalone capital letter in range. Stage three
is case-sensitive on purpose, because a case-insensitive match would read the
English article "a" as option A.

**Scoring.** Every response is `correct`, `incorrect`, or `invalid`. A refusal or
an unparseable reply is `invalid`, not `incorrect` — a model that declines to
answer has not answered wrongly. Invalid rates are reported separately.

**Statistics.** Accuracy per model and condition; the TruthfulQA effect minus the
MMLU effect as the main test; McNemar for paired comparisons (exact binomial
below 25 discordant pairs, chi-square with continuity correction above);
Cochran's Q across all five conditions; Benjamini-Hochberg correction across the
48 model-level tests and across the 38 category tests.

**Chance baselines** differ between the datasets: TruthfulQA options vary from 2
to 13, giving 0.2261; MMLU is a uniform four options, giving 0.2500.

---

## Output format

One JSON object per line in `results/*.jsonl`, carrying the answer and the full
configuration that produced it:

```json
{"model": "mistral", "condition": "persona_b", "template": "t1", "item_id": 42,
 "category": "Misconceptions", "n_options": 4, "temperature": 0.0, "seed": 0,
 "shuffled": true, "raw": "B) ...", "parsed": "B", "correct_letter": "B",
 "outcome": "correct"}
```

Resume keys on `model`, `condition`, `template`, `item_id`, `shuffled`,
`temperature` and `seed`, so re-running a command collects only what is missing.

---

## Notes and caveats

- **Quantization is not uniform:** phi3.5 is Q4_0, the other three are Q4_K_M.
- **Ollama is not perfectly deterministic.** Repeat runs at temperature 0 agreed
  on 3,999 of 4,000 rows, not 4,000.
- **Temperature seeds are not independent.** At 0.7, seed 1 differs from greedy
  decoding on 27% of rows, seeds 2 and 3 on under 0.5%: the answer distribution
  on constrained multiple-choice prompts is sharply peaked.
- **Three files hold more than one template** (`templates.jsonl`,
  `extra_personas_*.jsonl`). Filter on `template` before aggregating, or every
  question is counted three times.
- **`control_paraphrases.jsonl` contains 291 duplicate rows** from two
  concurrent runs. The analysis keeps the first of each.

---

## Requirements

Python 3.9+, [Ollama](https://ollama.com), and about 20 GB of disk for the four
models. No GPU is strictly required, but CPU-only inference will be slow.
>>>>>>> 50b7b86 (Repo initialization)
=======
# Effects-of-Persona-Prompting
>>>>>>> 12ef141d3e91e420cfaa1e1165d44b7be38efe1e
>>>>>>> 66e3fb2 (Repo Setup)
