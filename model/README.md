# Sakina comfort-generation model — technical notes

## What this is

A small (4-layer, 4-head, 192-dim, ~1.8M parameter) character-level
GPT, trained **from scratch** on a corpus of ~340K characters built from
~2,000 short labeled Arabic examples (`<USER>/<EMOTION>/<COMFORT>` cells).
Same architecture family as the project's earlier `out-sakina-v09`
checkpoint, retrained on the expanded 11-emotion dataset.

## What it can and can't do (please read before using this in a demo)

- **Loss dropped from 5.28 → 1.19** over training — the model is
  genuinely learning structure (Arabic character frequencies, common
  bigrams/trigrams, roughly where `<COMFORT>` sections tend to end).
- It **cannot reliably produce clean, coherent multi-word Arabic
  sentences on its own.** On a random 40-example sample, the quality
  gate rejects **~90%+** of raw generations (duplicated words, dropped
  letters, run-on fragments). See `evaluation/comparison_report.md`
  section 2 for the measured number and raw examples.
- This is **not a bug to fix with better prompting or sampling
  parameters** — it's a fundamental scale/data ceiling. Char-level
  GPTs trained from scratch typically need tens of millions of
  characters (and/or a bigger network) before they produce fluent free
  text. ~340K characters and 1.8M parameters is closer to "learns
  Arabic looks like this" than "can write like this."

## Why we kept it anyway, and how it's used safely

Rather than either (a) pretending the raw output is production-quality,
or (b) throwing the generative component away entirely, the pipeline
(`backend/response_composer.py`) uses it as a **gated best-effort
layer**:

1. The model generates a candidate acknowledgment line.
2. `model/inference.py::is_bad_generation()` checks it against length,
   repetition, control-token, and similarity-to-a-reviewed-line rules.
3. If it fails (as it does ~90%+ of the time right now), the system
   falls back to a curated, human-reviewed line from
   `data/knowledge.py::COMFORT_BANK` — 4 rotated variations per emotion,
   so it doesn't feel like the same fixed sentence every time either.

This means the product is **reliable today** (every response the user
sees has been human-reviewed, directly or via the fallback bank) while
leaving a real path to improve, rather than a dead end.

## What would actually close the gap

If "the AI genuinely writes the comfort line" is a hard requirement for
a later version, the highest-leverage change is **not** training this
architecture longer — it's switching strategy:

1. **Fine-tune a small pretrained Arabic language model** (e.g. an
   AraGPT2/AraBERT-family causal LM, or a small open Arabic instruction
   model) on this same `<USER>/<EMOTION>/<COMFORT>` corpus, instead of
   training a transformer from random initialization. A model that
   already knows Arabic grammar/vocabulary needs far less data to learn
   *this task's register* than one learning Arabic itself from 2,000
   examples.
2. Grow the dataset, especially the 4 new emotion classes (currently
   30-44 examples each — see the classifier caveat in
   `evaluation/comparison_report.md`).
3. Keep the same quality-gate philosophy either way — even a good model
   answering on religious/emotional topics should stay reviewable, not
   fully improvisational.

## Files

- `nano_gpt.py` — the GPT architecture (unchanged from the original
  nanoGPT reference implementation).
- `tokenizer.py` — character-level tokenizer, build/save/load.
- `prepare_corpus.py` — builds the training corpus + tokenizer from
  `data/emotions_dataset.json`.
- `train.py` — resumable training script (`--chunk N` trains N more
  iterations and checkpoints; safe to interrupt and re-run).
- `inference.py` — loads the trained checkpoint, generation +
  quality gate.
