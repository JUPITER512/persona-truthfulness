<!-- # CLAUDE.md

Project context for Claude Code. Read automatically at the start of every session.

---

## Who and what

I am Syed Ali Murtaza Bokhari, a Master's student in NLP at Universität Trier. This
repository is the empirical component of my term paper for **Advanced Topics in NLP**
(summer semester 2026). Supervisor: **Christoph Hau**. Due **30 September 2026**.

I am a beginner at empirical research and at Python tooling. Explain things rather than
just doing them, and tell me when I am about to make a mistake.

The paper is ~10 pages of body text (±15%), APA 7, individual work.

---

## The research question

**Does Argyle-style demographic backstory conditioning reduce truthfulness on
imitative-falsehood questions, and does the effect concentrate in categories where the
human-plausible answer is false?**

**Hypothesis (directional, falsifiable):** persona conditioning degrades accuracy
specifically on imitative-falsehood questions, concentrated in categories where the common
human belief is wrong, while leaving general knowledge accuracy unaffected.

### The result so far: the hypothesis is not supported

Data collection is complete. The measured effect runs **opposite** to the prediction:

- On TruthfulQA, personas mostly **raise** accuracy (mistral +5 to +7 pp over `none`).
- On MMLU, personas **lower** accuracy for three of four models (−1 to −5 pp).
- The TQA−MMLU difference is **positive in 11 of 12 comparisons** — the reverse of the
  predicted negative-on-TQA, flat-on-MMLU pattern.

This is a legitimate result. Hau's email framed it symmetrically ("which personas improve
**or degrade** performance"), and the literature splits three ways on the sign. Do not
rescue the hypothesis with speculation; report the falsification.

### Why this question exists

Two papers from my course sit on opposite sides of a contradiction:

- **Lin, Hilton & Evans (2022), TruthfulQA** — models reproduce human false beliefs. This
  is a **defect**, named *imitative falsehood*. Headline finding: inverse scaling.
- **Argyle et al. (2023), "Out of One, Many"** — conditioning on demographic backstories
  reproduces human response patterns. This is the **goal** of silicon sampling, called
  *algorithmic fidelity*.

Both describe the same capability. One field wants to remove it, the other to increase it.

### Prior work and how I differ

Closest study: **Luz de Araujo, P. H., & Roth, B. (2025).** "Helpful assistant or fruitful
facilitator? Investigating how personas affect language model behavior." *PLoS ONE, 20*(6),
e0325664. 162 identity/occupation **labels**, 7 LLMs, 5 datasets, control-persona and
empty-persona settings. Up to 38.56 pp spread on TruthfulQA for GPT-3.5, 4.58 pp for GPT-4.

**Corrected contribution claim.** They used **both TruthfulQA and MMLU** (their RQ1), so I
am *not* the first to pair them. What they did not do is compare the **magnitude** of the
persona effect across the two datasets. My contribution is the difference-in-differences,
plus the intervention (multi-attribute backstories, not identity labels) and the framing
(methodological validity for silicon sampling, not prompt engineering for practitioners).

The direction of the effect is **contested**: one study finds accuracy drops, another finds
it rises (arXiv 2510.22170), a third finds it depends on implied competence (Wharton
"Playing Pretend", 2025). **I have not verified those two — do not cite them from any
description; check the originals.**

---

## Supervisor requirements (approved 27 August 2026)

| # | What he asked | Status |
|---|---|---|
| 1 | Use the university server if hardware blocks you | **Not needed** — all four models ran locally. Tell him so |
| 2 | Cite Luz de Araujo & Roth in Related Work | **Outstanding** — the only requirement still to write |
| 3 | MMLU as part of the main evaluation, not an add-on | **Done** — 22,800 rows, Table 1 beside TruthfulQA |
| 4 | Address MC weaknesses; consider a generation variant | **Measured** — format, ordering, sampling and a generation slice |

---

## Experimental design

### Conditions

| Key | System prompt |
|---|---|
| `none` | *(no system message at all)* |
| `control` | "You are a helpful assistant." |
| `persona_a` | 54-year-old white woman, rural Ohio, high school, retail, $38k, conservative Republican, weekly church |
| `persona_b` | 27-year-old Black man, Atlanta, bachelor's, software developer, $85k, liberal Democrat, no religious services |
| `persona_c` | 41-year-old Hispanic woman, Phoenix, some college, nurse, $56k, independent, occasional church |

Extensions live in `persona.py` alongside the frozen five, as separate dicts with their
own condition names so they cannot collide:

- `EXTRA_PERSONAS` — `persona_d` (Arab American, Muslim), `persona_e` (South Asian,
  Hindu), `persona_f` (white, Jewish; drafted, not active).
- `CONTROL_PARAPHRASES` — `ctrl_p01`–`ctrl_p30`, the 30 paraphrases of "helpful assistant"
  from Luz de Araujo & Roth Table 2. `ctrl_p01` is byte-identical to `control`.

**The five frozen conditions must stay byte-identical.** `verify.py` checks this.

All backstories use identical sentence structure and the same nine attributes. Only values
change.

### Datasets

- **TruthfulQA** (`truthfulqa/truthful_qa`), MC1, **817** questions, **38** categories.
  Requires joining `multiple_choice` to `generation` on question text — only `generation`
  carries categories, and the keys must be `.strip()`ed or one item is lost.
- **MMLU** (`cais/mmlu`, config `all`), **1,140** questions, **57** subjects, 20 per
  subject. Built in two seeded passes — see `02_MMLU_data.py` below.

### Models (4, local via Ollama)

`phi3.5` (Q4_0) · `mistral` (Q4_K_M) · `qwen2.5:7b` (Q4_K_M) · `llama3.1:8b` (Q4_K_M)

All 4-bit. Quantization is **not** uniform across models and must be reported.

### Statistics

| Test | For |
|---|---|
| MC1 accuracy | Main outcome, both datasets |
| **TQA delta − MMLU delta** | The actual hypothesis test |
| McNemar | Baseline vs one persona — data is **paired** |
| Cochran's Q | Omnibus across all five conditions |
| Benjamini–Hochberg | Two families: 48 model-level tests, and 38 categories per comparison |
| Invalid rate | Refusals and unparseable replies, reported separately |
| Run-to-run agreement | Noise floor: 100% at temperature 0 |

**Never** use BLEURT, cosine similarity, F1, or perplexity here. They measure resemblance
to a reference string, not truthfulness, and would undermine the "objective scoring without
a judge model" justification Hau accepted.

---

## Repository

```
config.py               all settings, environment-overridable
persona.py              five FROZEN conditions, templates, control paraphrases,
                        extra personas — all prompts in one file
backend.py              Ollama HTTP call (Hugging Face backend removed)
01_Load_data.py         TruthfulQA, joined across configs
02_MMLU_data.py         MMLU 20/subject (1,140), ids preserved, guarded
run.py                  the multiple-choice experiment runner
04_generation.py        generation slice, 250 open-ended answers
05_score_generation.py  blind scoring sheet + merge
check_pilot.py          sanity check + raw output dump
analyse.py              statistics, 8 sections, CSVs for the appendix
data/                   generated
results/                JSONL logs, one object per line
```

### Run modes

```
python run.py pilot                50 questions, sanity check
python run.py main                 TruthfulQA, 4 models, 5 conditions
python run.py mmlu                 MMLU, 4 models, 5 conditions
python run.py templates            prompt-format sensitivity (t1, t2, t3)
python run.py noshuffle            answer-order check, unshuffled
python run.py shuffle              shuffled repeat = noise floor
python run.py temp                 temperature 0.7, seeds 1/2/3
python run.py control_paraphrases  prompt-sensitivity control
python run.py extra                persona_d / persona_e
python 04_generation.py            generation slice
python 05_score_generation.py sheet | merge | sample
python analyse.py                  all statistics
```

### Data collected — 87,393 model calls

```
main.jsonl                16,340   4 models x 5 conditions x 817 TruthfulQA
mmlu.jsonl                22,800   4 models x 5 conditions x 1,140 MMLU
extra_personas_tqa.jsonl  19,608   persona_d/e, 3 templates
extra_personas_mmlu.jsonl  7,104   persona_d/e
templates.jsonl            6,000   3 formats, 100 questions, 4 models
control_paraphrases.jsonl  4,291   10 paraphrases, 100 questions
noshuffle + shuffle        4,000   order effect + noise floor
temp_1/2/3.jsonl           6,000   temperature 0.7, three seeds
pilot.jsonl                1,000   sanity check
generation.jsonl             250   open-ended, mistral, 5 conditions
```

---

## Rules for working in this repo

**Do not edit `persona.py`.** The five conditions are frozen. Changing them makes 87,000
collected rows incomparable. New conditions go in a new file with new names.

**Do not change the parser without re-running the pilot.** Parsing rules are part of my
reported configuration.

**Never silently drop invalid outputs.** Every result is `correct`, `incorrect`, or
`invalid`. A refusal is not a wrong answer — the invalid rate is a finding in its own right.

**Every result row must log the full configuration** — model, condition, template,
temperature, seed, shuffle flag, raw output, parsed letter. That log is my appendix and the
basis of my reproducibility claim. Do not remove fields.

**Runs must remain resumable.** `run.py` keys on
`(model, condition, template, item_id, shuffled, temperature, seed)` and skips what exists.

**Never run two copies of the same mode at once.** There is no file lock. Two writers
produced 291 duplicate rows in `control_paraphrases.jsonl`.

**Keep seeds fixed and explicit.** `SEED = 0`, `MMLU_SEED = 42`.

---

## Known issues and hazards

### `02_MMLU_data.py` is guarded — do not remove the guard
Ids 0–284 were assigned by the original 5-per-subject draw, and 22,800 collected rows are
keyed by them. The script aborts rather than write a dataset in which an existing id would
mean a different question.

### Three files mix templates
`templates.jsonl`, `extra_personas_tqa.jsonl`, `extra_personas_mmlu.jsonl` each hold t1, t2
and t3. Anything loading them without filtering by template counts every question three
times. `analyse.py` handles this; new code must too.

### `control_paraphrases.jsonl` has 291 duplicate rows
From two concurrent runs; 2 of them disagree. `analyse.py` keeps the first per key, so the
numbers are valid, but dedupe before the appendix.

### Ollama is not perfectly deterministic
3,999 of 4,000 identical on repeat runs at temperature 0 — one row differed. Report
**99.9%**, not 100%.

### Temperature seeds are not independent
At 0.7, seed 1 differs from greedy on 27% of rows; seeds 2 and 3 on 0.3% and 0.2%. The
distribution is sharply peaked on constrained MC prompts, so most seeds reproduce greedy.
You have **one** draw that moved, not three.

### Chance baseline is not 25%
TruthfulQA MC1 option counts vary from 2 to 13, so mean chance is **0.2261**. MMLU is a
clean **0.2500**. Report both and note the format mismatch in Limitations.

---

## Status

**Done:** all data collection (87,393 calls), full analysis pipeline, generation slice
collected, code reviewed and bug-swept.

**Not done:**
1. **The paper — 0 of ~10 pages.** This is the critical path.
2. `git init` + GitHub — a submission requirement, not optional.
3. Hand-scoring the 250 generations (`generation_sheet.csv`), ~45 minutes, blind.
4. Appendix, title page, Eigenständigkeitserklärung, GenAI declaration.
5. Reply to Hau — he offered a follow-up and never got one.

**Open decisions to fix and justify in Method, before reading the results they affect:**
`MIN_CATEGORY_N` (currently 1), `INVALID_AS_INCORRECT` (currently True), which baseline is
primary (`none` or `control`), whether to correct across the 48 model-level tests, and how
to report the temperature check.

---

## Hardware

Laptop: RTX 4050 (6 GB VRAM), i7 13th gen, 32 GB RAM. Ollama offloads overflow layers to
CPU. **One model at a time.** MC outputs cap at 10 tokens; generation at 120.

---

## Academic integrity — read this before offering to write anything

My course permits AI for **phrasing, editing, coding assistance, and ideation**. It does
**not** permit AI to substitute for my own work. Specifically:

- The research question, ideas, and plan must originate from me
- The work must be predominantly my own human effort
- I must be able to explain every word and every line I submit
- **If I cannot explain my own text or code, that is explicitly not a permitted use**
- Uncorrected AI errors are my mistake — zero tolerance
- I must mark GenAI-generated passages in an appendix and list every tool by product name
- The declaration I sign says GenAI was used only "in the manner permitted **in writing**
  by the examiner"

**So: do not write my term paper.** If I ask you to, remind me of this instead of complying.
This includes the appendix, which is where the GenAI declaration itself lives.

**Do** help me with: debugging, code review, explaining what code does, structuring
sections, reviewing my drafts and telling me what is weak, APA formatting, and tightening
prose I have written.

**Verify citations** you give me and tell me to check them against the original. An error
you introduce becomes my mistake.

### Code I must be able to explain
- Why `SHUFFLE_DEFAULT = True` (mc1_targets lists the correct answer first, so unshuffled
  the answer was `A` on all 817 items)
- Why the parser's third stage is case-sensitive (the article "a" would parse as option A)
- Why a refusal is `invalid`, not `incorrect`
- Why McNemar and not a t-test (paired data; only discordant pairs carry information)
- Why the exact binomial test when b+c < 25
- Why Benjamini–Hochberg and not Bonferroni (FDR, less conservative)
- Why `02_MMLU_data.py` samples in two passes instead of just raising `MMLU_PER_SUBJECT`

---

## How to work with me

Give me **one concrete task at a time**, small enough to finish in one sitting. Do not give
me a list of ten things.

Be concise. Short question, short answer.

If I keep asking for recaps or proposing new experiments instead of writing, **say so
directly** — that pattern means I am stuck, not that the plan is wrong. Data collection is
finished; more experiments are avoidance. Ask what is actually blocking me.

Tell me when I am wrong. I would rather hear it now than from my examiner in October. -->
