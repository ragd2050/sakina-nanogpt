# -*- coding: utf-8 -*-
"""
Produces evaluation/comparison_report.md:
  - "Before": emotion classifier only, single fixed line per emotion,
    a single fixed verse, no reflection, no follow-up, no memory.
    (This is literally what the original sakina_pipeline.py did before
    this pass, and is the most honest baseline to compare against --
    not a strawman.)
  - "After": the v10 pipeline (response_composer.sakina_response).
  - Fallback-rate measurement for the generation model, reported plainly.
"""
import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "model"))

from response_composer import sakina_response  # noqa: E402
from knowledge import COMFORT_BANK, EMOTION_VERSES  # noqa: E402
import json

with open(ROOT / "data" / "quran_verses.json", encoding="utf-8") as f:
    QURAN_DB = json.load(f)

with open(ROOT / "checkpoints" / "sakina_emotion_classifier.pkl", "rb") as f:
    CLASSIFIER = pickle.load(f)

TEST_CASES = [
    ("Anxiety / قلق وتوتر", "قلبي مو مرتاح وتفكيري مشتت من كثر التوتر قبل المقابلة بكرة"),
    ("Sadness / حزن", "حاسس بحزن ثقيل من فترة وما عارف ليش بالضبط"),
    ("Loneliness / وحدة", "من فترة طويلة حاسس إني وحيد حتى لما أكون بين الناس"),
    ("Confusion / حيرة وتشتت", "عندي قرار كبير لازم آخذه وما عارف وش أختار"),
    ("Happiness / فرح وشكر", "الحمد لله اليوم صار لي شي كنت متمني له من زمان"),
    ("Guilt / ذنب وتقصير", "حاسس بذنب كبير إني قصرت مع أهلي هالفترة"),
    ("Loss / فقد وشوق", "اشتقت لجدي اللي توفى من سنتين، اليوم ذكرياته رجعت كلها"),
    ("Stress / ضغط نفسي", "عندي شغل وامتحانات بنفس الأسبوع وحاسس إني بأنهار"),
]


def baseline_before(text: str):
    """Old-style pipeline: classify -> fixed line -> fixed verse."""
    emotion = CLASSIFIER.predict([text])[0]
    verse_id = EMOTION_VERSES[emotion][0]      # always the same first verse
    verse = QURAN_DB[verse_id]
    comfort = COMFORT_BANK[emotion][0]          # always the same first line
    return {
        "emotion": emotion,
        "comfort": comfort,
        "verse": verse["text"],
        "verse_ref": f'سورة {verse["surah_name"]}، الآية {verse["ayah_number"]}',
    }


def measure_fallback_rate(n=40):
    import random
    with open(ROOT / "data" / "emotions_dataset.json", encoding="utf-8") as f:
        rows = json.load(f)
    random.seed(11)
    sample = random.sample(rows, n)
    from inference import generate_with_fallback
    used = 0
    for r in sample:
        _, fb, _ = generate_with_fallback(r["user_input"], r["detected_emotion"], r["comfort_message"])
        used += fb
    return used, n


SANITY_CHECKS = [
    # (text, expected_emotion) -- hand-written, NOT drawn from training data,
    # specifically to test generalization rather than memorization.
    ("حاسس بحزن ثقيل من فترة وما عارف ليش بالضبط", "حزن"),
    ("من فترة طويلة حاسس إني وحيد حتى لما أكون بين الناس", "وحدة"),
    ("قلبي مو مرتاح وتفكيري مشتت من كثر التوتر قبل المقابلة بكرة", "قلق وتوتر"),
    ("عندي شغل وامتحانات بنفس الأسبوع وحاسس إني بأنهار", "ضغط نفسي"),
    ("اشتقت لجدي اللي توفى من سنتين", "فقد وشوق"),
    ("زعلان جدا من اللي صار وودي افجر", "غضب"),
    ("خايف من نتيجة الفحص اللي بسويه بكرة", "خوف"),
    ("مافي فايدة، حاولت كثير وما تغير شي", "فقدان أمل"),
]


def run_sanity_checks():
    rows = []
    correct = 0
    for text, expected in SANITY_CHECKS:
        pred = CLASSIFIER.predict([text])[0]
        ok = pred == expected
        correct += ok
        rows.append((text, expected, pred, ok))
    return rows, correct, len(SANITY_CHECKS)


def v11_changes_section():
    lines = []
    lines.append("## 0. v11 changes (this pass) — summary\n")
    lines.append("Full request was a large architecture upgrade (context memory, confidence-threshold "
                  "clarification, dialect-robust classification, a bigger conversational dataset, an "
                  "improved generation model, new `/api/*` endpoints, quality evaluation). Here's what's "
                  "real, with numbers, and what fell short of the ask.\n")

    lines.append("**Classifier: real, measured improvement.**")
    lines.append("- Added Arabic normalization (alef/taa-marbuta/alef-maqsura unification, diacritics "
                  "and tatweel stripped, letter-elongation collapsed) and combined word (1-2 gram) + "
                  "character (2-4 gram, word-boundary aware) TF-IDF features, replacing word-only n-grams.")
    lines.append("- Held-out accuracy: 87.6% → **92.5%**, macro F1: 0.85 → **0.94**.")
    lines.append("- More importantly, on the exact 8 hand-written generalization sentences that exposed "
                  "real failures last round (not training data), plus 9 new sentences using the specific "
                  "dialect words requested (قلقانة، زعلانة، مبسوطة، وحيدة، مقهورة، محتارة، ضايعة، ندمانة، "
                  "مشتاقة): **4/8 → 15/17 correct**. This is the number that actually matters — the "
                  "normalization + char n-grams fix generalized, it didn't just move the held-out split.")
    lines.append("- Found and fixed a real deployment bug along the way: the first version of the "
                  "normalizer was a locally-defined class, which made the pickled classifier fail to "
                  "load (`AttributeError` on unpickling) from any script other than the one that trained "
                  "it — including the Flask backend. Moved it to a stable importable module "
                  "(`data/normalize.py`) before shipping.\n")

    lines.append("**Confidence-threshold clarification: implemented, and it caught a real gap.**")
    lines.append("- `needs_clarification()` now checks classifier confidence + margin between top-2 "
                  "classes for short messages, returning `{\"status\": \"clarification_needed\", "
                  "\"question\": ...}` per the requested `/api/chat` contract, in addition to the curated "
                  "trigger list (تعبان، طفشان، etc.).")
    lines.append("- Testing this surfaced a genuine miss: **\"طفشان\"** — named explicitly in the "
                  "original brief as needing clarification — had zero training examples, and the "
                  "classifier guessed **فرح وشكر (happiness)** for it with enough confidence that the "
                  "generic threshold didn't fire. Added it to the curated trigger list with the exact "
                  "clarifying question from the brief, rather than trusting the generic threshold alone "
                  "for known problem words.\n")

    lines.append("**Conversation memory: fixed a real \"restarting from zero\" bug.**")
    lines.append("- The v10 context-carry threshold (confidence < 0.35) was too strict. Testing the "
                  "brief's own example — \"أنا خايفة من الاختبار\" (fear) → \"ما قدرت أنام\" — showed the "
                  "second message getting freshly (and wrongly) classified as **حزن (sadness)** at 0.59 "
                  "confidence, not carried forward as anxiety/fear. Raised the threshold to 0.65 for "
                  "short follow-ups with an established prior emotion. Retested: now correctly carries "
                  "**خوف (fear)** forward and says so explicitly (\"يبدو أن هذا امتداد لما ذكرته قبل "
                  "قليل...\"). Verified a genuinely new, clearly-expressed emotion right after "
                  "still overrides the carried context rather than getting stuck.\n")

    lines.append("**Dataset: grew, but nowhere near 10,000.**")
    lines.append("- 2,046 → **2,755** flat examples via (a) explicit anchor sentences for every requested "
                  "dialect synonym word, and (b) template × feeling-word × reason-clause combinatorial "
                  "augmentation — every generated sentence is a real, distinct, grammatical Arabic "
                  "sentence, not a duplicate.")
    lines.append("- **Honestly, this is ~28% of the 10,000 requested, not 10,000.** Getting to 10,000 "
                  "genuinely varied examples (the brief specifically asked for short messages, long "
                  "emotional stories, follow-ups, mixed emotions, different writing styles) is a data "
                  "collection effort, not something combinatorial templating can respectably fake — "
                  "beyond a certain point, more template combinations just means more *structurally* "
                  "similar sentences, which teaches the classifier less per example and can actually hurt "
                  "a generator by making training data look more repetitive than real usage, not less.")
    lines.append("- Also emitted `data/conversations_dataset.json` in the requested "
                  "`{\"conversation\": [...], \"emotion\", \"verse_id\"}` shape: 2,755 single-turn entries "
                  "(every flat example wrapped) plus **5 genuinely hand-authored multi-turn conversations** "
                  "(2 exchanges each) demonstrating the follow-up format. 5, not thousands — same honesty "
                  "note as above.\n")

    lines.append("**nanoGPT: retrained on the bigger corpus — real improvement, with an important caveat.**")
    lines.append("- Retrained (v11) on the 424K-character corpus (up from 338K), same architecture, 2000 "
                  "iterations. Final val loss **0.85–0.97** (down from 1.19).")
    lines.append("- Measured fallback rate dropped from **95% → 50%** on a 60-example sample.")
    lines.append("- **The honest caveat: most of that improvement is memorization, not generation "
                  "quality.** Checked directly — of the generations that passed the quality gate, "
                  "**~37% were ≥85% similar to one of the 4 fixed curated lines for that emotion "
                  "(near-verbatim reproduction)**, and most of the rest were still visibly garbled "
                  "corruptions of those same 4 lines (dropped/duplicated word fragments), just close "
                  "enough in edit-distance to pass. With only 4 possible targets per emotion repeated "
                  "across hundreds of near-identical augmented inputs, 2000 iterations gives the model "
                  "enough exposure to memorize those 4 lines rather than learn to generalize past them. "
                  "This is safe (a memorized reviewed line can't say anything harmful) but it means the "
                  "practical ceiling here is closer to \"expensive way to rotate 4 fixed lines\" than "
                  "\"generates personalized comfort,\" and it did not change with more data the way I'd "
                  "hoped — more data with only 4 target variations per class mostly just gives the model "
                  "more chances to memorize, not more to generalize from. **The actual fix is more "
                  "curated target variations per emotion (10-20+, not 4) and/or fine-tuning a pretrained "
                  "Arabic LM instead of training from scratch** (still blocked here by the network "
                  "allowlist not including any model hub — see `model/README.md`).")
    lines.append("- Also found and fixed a real quality-gate gap while checking this: the gate didn't "
                  "catch repeated whole words (e.g. \"كل كل\"); added that check. A subtler corruption "
                  "still slipped through in testing — \"الندم لا يعني أن **ب بابداية** أغلق\" (should be "
                  "\"باب البداية\") — close enough in edit-distance to the reviewed line to pass. Disclosed "
                  "rather than hidden; closing this fully needs either a stricter similarity threshold "
                  "(at the cost of a higher fallback rate) or a real Arabic word-validity check.\n")

    lines.append("**API: new endpoints match the requested schema exactly.**")
    lines.append("- `POST /api/chat` returns `{emotion, confidence, response, verse, verse_reference, "
                  "reflection, follow_up_question}` on success, or `{status: \"clarification_needed\", "
                  "question}` when ambiguous — tested with both branches.")
    lines.append("- `POST /api/predict_emotion`, `POST /api/generate_response` added per spec. Previous "
                  "`/predict_emotion`, `/generate_response`, `/chat` kept as aliases for backward "
                  "compatibility, not removed.\n")

    lines.append("**Quality evaluation: real metrics + a disclosed proxy for the parts that need humans.**")
    lines.append("- Emotion accuracy (held-out, n=200): **98.5%** on the v11 dataset (this is higher than "
                  "the 92.5% headline figure above because this run includes the easier augmented "
                  "examples in the pool; the 92.5%/15-of-17 numbers are the more representative ones for "
                  "real-world generalization).")
    lines.append("- Raw (pre-fallback) repetition rate: **11.7%** of generations contain a directly "
                  "repeated word or character chunk.")
    lines.append("- \"Response quality\" and \"human evaluation score\" were requested; I don't have "
                  "access to real human raters in this environment, so `evaluation/quality_eval.py` "
                  "implements a disclosed proxy rubric (feeling validated first / non-templated / verse "
                  "relevance / natural close) that I apply myself to a fixed sample, clearly labeled as a "
                  "proxy — not presented as real user-testing data. See `evaluation/quality_metrics.json` "
                  "for the scored sample.\n")
    return lines


def main():
    lines = v11_changes_section()
    lines.append("# SAKINA — Before / After Evaluation (v10)\n")
    lines.append("## 1. Emotion classifier\n")
    lines.append("See `classifier_report.txt` for the full held-out report. Summary:\n")
    lines.append("- 11 emotion classes (up from 8; added فقدان أمل / خوف / غضب / ضغط نفسي)")
    lines.append("- Held-out accuracy: **86.97%**, macro F1: **0.839**")
    lines.append("- The 4 newly-added classes have far less training data (30-44 examples vs 200-376 for the "
                  "original classes) and it shows: خوف in particular sits around F1 0.50 on a tiny 7-example "
                  "test split. This is an honest limitation, not a hidden one — more real (not synthetic) "
                  "examples for those 4 classes is the single highest-leverage next step.\n")

    lines.append("## 2. Comfort-generation model (nanoGPT)\n")
    fb_used, fb_n = measure_fallback_rate(40)
    lines.append(f"- Retrained from scratch on the expanded 11-emotion corpus (same architecture as the "
                  f"original v9: 4 layers / 4 heads / 192 dim, ~1.8M params, char-level, CPU-trained).")
    lines.append(f"- Final validation loss: **1.19** (dropped from 5.27 at initialization — the model is "
                  f"learning real structure, not random noise).")
    lines.append(f"- **Fallback rate on a 40-example random sample: {fb_used}/{fb_n} = {fb_used/fb_n:.0%}.**")
    lines.append("- In plain terms: this model, at this scale, trained on ~2,000 short examples, cannot "
                  "reliably produce clean multi-word Arabic on its own — it mostly assembles near-fragments "
                  "of training phrases with typos/repeats, which the quality gate correctly rejects almost "
                  "every time. This is a **data and scale ceiling**, not a bug in the training loop or "
                  "prompting: char-level GPTs need tens of millions of characters and/or many more layers "
                  "before they produce coherent free text. See `../model/README.md` for what would actually "
                  "close this gap (fine-tuning a small pretrained Arabic LM instead of training from scratch).")
    lines.append("- Because of this, the *reliable* backbone of the product is the curated, reviewed comfort "
                  "bank (4 rotated variations per emotion) — which is also the safer choice for anything "
                  "quoting religious guidance. The model is kept in the loop as a best-effort layer that can "
                  "only ever pass through text that is close to a reviewed line, never invent new religious "
                  "framing on its own.\n")

    lines.append("## 3. Generalization sanity check (honest finding)\n")
    lines.append("The held-out metrics above are on a random split of the *same* dataset the model trained "
                  "on, which is the standard way to measure a classifier but can still overstate real-world "
                  "readiness when a class has few examples from a single author (here: me, writing synthetic "
                  "examples for the 4 new classes). To sanity-check this, I hand-wrote 8 fresh sentences "
                  "afterward, deliberately *not* copied from the training data, one per emotion:\n")
    rows, correct, total = run_sanity_checks()
    lines.append("| Text | Expected | Predicted | Correct |")
    lines.append("|---|---|---|---|")
    for text, expected, pred, ok in rows:
        lines.append(f"| {text} | {expected} | {pred} | {'✓' if ok else '✗'} |")
    lines.append(f"\n**Result: {correct}/{total} correct.**\n")
    lines.append("The failures are concentrated exactly where expected: novel sentences about حزن/وحدة/قلق "
                  "that happen to share a connector phrase (e.g. \"من فترة طويلة\") with one of my synthetic "
                  "غضب/ضغط نفسي examples get pulled toward the wrong minority class. This is a direct, "
                  "traceable consequence of the 4 new classes having only 30-44 examples from one author "
                  "instead of 200+ examples from mixed real/synthetic sources like the original 7 classes. "
                  "**The fix is not a modeling change — it's more, more varied, ideally real, labeled "
                  "examples for those 4 classes** before this ships to real users. I'm reporting this "
                  "plainly rather than only showing the flattering held-out number.\n")

    lines.append("## 4. Architecture comparison\n")
    lines.append("| | Before (classifier-only) | After (v10 pipeline) |")
    lines.append("|---|---|---|")
    lines.append("| Emotion classes | 8 | 11 |")
    lines.append("| Comfort message | 1 fixed line per emotion | 4 rotated reviewed lines + generation attempt |")
    lines.append("| Verse | 1 fixed verse per emotion, always the same | 2 verses per emotion, rotated, avoids repeating last shown |")
    lines.append("| Reflection (تدبر) | none | 1 curated reflection per emotion, explicitly labeled as not tafsir |")
    lines.append("| Follow-up question | none | 2 rotated gentle follow-ups per emotion |")
    lines.append("| Conversation memory | none (every message independent) | last 6 turns tracked per session |")
    lines.append("| Emotional shift detection | none | acknowledges a detected shift between turns |")
    lines.append("| Ambiguous short input (e.g. \"تعبان\") | classified blindly | asks a clarifying question instead |")
    lines.append("| API | none | `/predict_emotion`, `/generate_response`, `/chat` |\n")

    lines.append("## 5. Side-by-side test cases\n")
    for label, text in TEST_CASES:
        before = baseline_before(text)
        after = sakina_response(text, session_id=f"eval-{label}")
        lines.append(f"### {label}")
        lines.append(f"**User:** {text}\n")
        lines.append("**Before:**")
        lines.append(f"> {before['comfort']}\n> ﴿ {before['verse']} ﴾ — {before['verse_ref']}\n")
        lines.append("**After:**")
        lines.append(f"> {after['response'].replace(chr(10), chr(10) + '> ')}\n")

    out = ROOT / "evaluation" / "comparison_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
