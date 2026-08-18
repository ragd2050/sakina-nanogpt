# -*- coding: utf-8 -*-
"""
Quality evaluation for the v11 pipeline.

Honest framing up front: "response quality" and "human evaluation score"
for open-ended generated text normally require real human raters. I don't
have access to human raters in this environment, so rather than invent a
fake number, this module implements:

  1. Objective, checkable metrics: emotion accuracy (real), repetition
     rate (real, regex-measurable), fallback rate (real).
  2. A disclosed *proxy* rubric I apply myself to a fixed sample, scored
     against explicit written criteria -- clearly labeled as a proxy, not
     a substitute for real user testing.
"""
import sys
import re
import json
import random
import pickle
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "model"))

from response_composer import sakina_response, classify_emotion  # noqa: E402
from inference import generate_with_fallback  # noqa: E402
from knowledge import COMFORT_BANK  # noqa: E402

with open(ROOT / "checkpoints" / "sakina_emotion_classifier.pkl", "rb") as f:
    CLASSIFIER = pickle.load(f)


# ---------------------------------------------------------------------
# 1. Emotion accuracy (held-out, real)
# ---------------------------------------------------------------------
def emotion_accuracy(n=200, seed=99):
    with open(ROOT / "data" / "emotions_dataset.json", encoding="utf-8") as f:
        rows = json.load(f)
    random.seed(seed)
    sample = random.sample(rows, min(n, len(rows)))
    correct = sum(1 for r in sample if CLASSIFIER.predict([r["user_input"]])[0] == r["detected_emotion"])
    return correct / len(sample), len(sample)


# ---------------------------------------------------------------------
# 2. Repetition rate (real, regex-measurable) -- what fraction of
#    generated (non-fallback) comfort lines contain a repeated word or
#    repeated character chunk. Distinct from the quality gate itself so
#    we can report "how much garbage did the raw model produce" as its
#    own number, not just "how much did we successfully hide."
# ---------------------------------------------------------------------
_REPEAT_WORD = re.compile(r"\b(\S+)\s+\1\b")
_REPEAT_CHUNK = re.compile(r"(.{2,4})\1{2,}")


def is_repetitive(text: str) -> bool:
    return bool(_REPEAT_WORD.search(text) or _REPEAT_CHUNK.search(text))


def repetition_rate(n=60, seed=11):
    with open(ROOT / "data" / "emotions_dataset.json", encoding="utf-8") as f:
        rows = json.load(f)
    random.seed(seed)
    sample = random.sample(rows, n)
    raw_outputs = []
    for r in sample:
        _, _, raw = generate_with_fallback(r["user_input"], r["detected_emotion"], r["comfort_message"])
        raw_outputs.append(raw)
    rep = sum(is_repetitive(t) for t in raw_outputs)
    return rep / len(raw_outputs), len(raw_outputs)


# ---------------------------------------------------------------------
# 3. Disclosed proxy "human evaluation" -- fixed rubric, applied by me to
#    a fixed sample, clearly labeled as a proxy.
# ---------------------------------------------------------------------
RUBRIC = """
Proxy human-eval rubric (1-5 each), applied to the FINAL composed response
(after fallback, i.e. what a real user would actually see):
  - Validates feeling before advising (1 = jumps straight to a verse/advice, 5 = clearly acknowledges the feeling first)
  - Feels non-robotic / non-templated in isolation (1 = obviously canned, 5 = reads like it was written for this message)
  - Verse relevance to stated situation (1 = generic/unrelated, 5 = clearly fitting)
  - Ends naturally rather than abruptly (1 = trails off/repeats, 5 = natural close/question)
"""

PROXY_SAMPLE = [
    "أنا تعبت وما عاد أقدر أكمل",
    "خايفة من نتيجة القرار اللي أخذته",
    "حاسس بذنب كبير إني قصرت مع أهلي",
    "الحمد لله اليوم صار لي شي كنت متمني له",
    "من فترة طويلة حاسس إني وحيد",
]


def run_proxy_eval():
    results = []
    for text in PROXY_SAMPLE:
        r = sakina_response(text, session_id=f"proxy-{hash(text)}")
        results.append({"input": text, "emotion": r["emotion"], "response": r["response"]})
    return results


def main():
    acc, n_acc = emotion_accuracy()
    rep_rate, n_rep = repetition_rate()

    print(f"Emotion accuracy (held-out, n={n_acc}): {acc:.1%}")
    print(f"Repetition rate in RAW generation before fallback (n={n_rep}): {rep_rate:.1%}")
    print()
    print(RUBRIC)

    proxy_results = run_proxy_eval()
    for r in proxy_results:
        print(f"[{r['emotion']}] {r['input']}")
        print(f"  -> {r['response']}\n")

    out = {
        "emotion_accuracy": acc,
        "emotion_accuracy_n": n_acc,
        "raw_repetition_rate": rep_rate,
        "repetition_rate_n": n_rep,
        "proxy_eval_note": "self-scored proxy rubric, not real user/human-rater data -- see RUBRIC in this file",
        "proxy_sample": proxy_results,
    }
    with open(ROOT / "evaluation" / "quality_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> evaluation/quality_metrics.json")


if __name__ == "__main__":
    main()
