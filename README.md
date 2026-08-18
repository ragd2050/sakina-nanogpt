# سَكِينَة (Sakina) — v13 emotional AI pipeline

**Start here:** `evaluation/v12_vs_v13.md` — what changed this pass, with the
test that verified each claim. `evaluation/v13_evaluation_report.md` for the
formal A-G evaluation (classifier, dialect words, ambiguous inputs,
multi-turn memory, nanoGPT quality, the v13 clarification fix, multi-turn
dataset audit — kept separate, never conflated). `evaluation/v13_run_metadata.json`
pins the exact checkpoint/dataset versions those numbers were run against —
**do not compare v13 numbers to v11/v12 reports without checking it first.**

## Before you run anything

- **Quran audio (`/api/reciters`, `/api/audio`) is written to the exact
  endpoint pattern given in the v13 brief, but still untested** — this
  sandbox has no network path to the Quran Foundation API, in v12 or now.
  Set real `QF_CLIENT_ID`/`QF_CLIENT_SECRET` and test it against the live
  API yourself before deploying.
- **The frontend needs the backend actually running** to do anything — it's
  not a demo. `python3 backend/app.py`, then open `frontend/index.html`; the
  chat header shows live connection status.
- **`data/conversations_dataset.json` (the v12 file) is 99.8% single-turn**
  despite its shape — see `evaluation/analyze_conversation_dataset.py` and
  section G of the v13 report. Use `data/conversations_dataset_v13.json`
  (1,798 verified genuine multi-turn conversations) for anything that
  actually needs multi-turn data.

## Running it

```bash
pip install -r requirements.txt

# rebuild data (only needed if you edit data/knowledge.py or the builders)
cd data && python3 build_dataset.py && python3 build_dataset_v11.py && python3 build_dataset_v12.py
python3 build_dataset_v13_multiturn.py   # NEW in v13: genuine multi-turn conversations

# retrain classifier (unchanged in v13 -- not the bottleneck, see brief section 3)
cd ../backend && python3 emotion_classifier.py

# retrain nanoGPT (resumable -- run repeatedly until it prints "DONE.")
cd ../model && python3 prepare_corpus.py
python3 train.py --chunk 350

# run the backend
cd ../backend
export QF_CLIENT_ID=your_id QF_CLIENT_SECRET=your_secret   # for audio, optional
python3 app.py    # http://localhost:5000

# open frontend/index.html in a browser (backend must be running)

# run evaluation
cd ../evaluation
python3 evaluate_v13.py
python3 analyze_conversation_dataset.py --file data/conversations_dataset.json
python3 analyze_conversation_dataset.py --file data/conversations_dataset_v13.json

# run automated tests
cd .. && python3 -m pytest tests/ -v
```

## Architecture

```
user message
   │
   ├─▶ safety check (self-harm/crisis keywords) → dedicated crisis response, bypasses everything else
   │
   ├─▶ pending-clarification check (v13, NEW) → if the previous turn asked
   │       "تعباً جسدياً أم نفسياً؟" and this message answers it (e.g. "نفسي"),
   │       resolve directly instead of reclassifying from zero
   │
   ├─▶ curated ambiguity triggers (تعبان, طفشان, ...) → clarifying question
   │       (sets pending-clarification for the next turn)
   │
   ├─▶ lexical anchor check (سعيدة, وحيدة, مظلومة, ...) → deterministic emotion, skip classifier
   │
   ├─▶ confidence/margin clarification check (short + uncertain) → clarifying question
   │       (also sets pending-clarification, generic two-option form)
   │
   ├─▶ emotion classification (TF-IDF word+char n-grams + SVM, 12 classes)
   │        └─▶ short + low-confidence + prior emotion exists → carry prior emotion forward
   │
   ├─▶ verse selection (2 curated verses per emotion, rotated, is_excerpt-aware,
   │       ALWAYS from data/quran_verses.json -- never model-generated, see
   │       tests/test_sakina.py::TestQuran::test_verse_text_is_never_sourced_from_the_model)
   │
   ├─▶ comfort-line generation
   │        ├─▶ nanoGPT attempts a line
   │        ├─▶ hardened quality gate (repeats, standalone letters, incomplete
   │        │   generations, vocabulary validity, cross-emotion contamination)
   │        └─▶ passes ~22-23% of the time (n=100, v13 -- see Limitations);
   │            otherwise falls back to one of 15 curated, rotated reviewed
   │            lines for that emotion (grown from 8 in v12)
   │
   ├─▶ reflection (curated; skipped for very short messages)
   │
   ├─▶ follow-up question (curated, rotated per session)
   │
   └─▶ composed response + conversation memory updated
```

## API

### `POST /api/chat` — main endpoint
```json
// request
{"message": "أنا خايفة من الاختبار", "session_id": "optional"}
// response (success)
{
  "success": true, "session_id": "...", "needs_clarification": false,
  "emotion": "خوف", "confidence": 0.86,
  "response": "...",
  "verse": {"id": "20:46", "text": "...", "surah_name": "طه", "ayah_number": 46, "is_excerpt": false},
  "reflection": "...", "follow_up_question": "..."
}
// response (ambiguous, e.g. "طفشان")
{"success": true, "needs_clarification": true, "question": "..."}
// response (safety trigger)
{"success": true, "is_safety_response": true, "response": "..."}
```

### `POST /api/predict_emotion`, `POST /api/generate_response`, `POST /api/reset_session`, `GET /api/health`
Per spec — see `backend/app.py`.

### `GET /api/reciters`, `GET /api/audio?verse_id=13:28&recitation_id=7`
v13: corrected to the exact endpoint pattern given in the brief
(`{QF_API_BASE}/quran/recitations/{recitation_id}?chapter_number={chapter_number}`,
matched by `verse_key`) — replacing v12's untested `/by_chapter/{chapter}`
guess. **Still untested against the real API** (no network access here).
`/api/audio` now returns `recitation_id` in the response per the brief's
exact schema. Degrades to a clear JSON error (HTTP 503) rather than
crashing if credentials or network aren't available.

### Legacy aliases
`/predict_emotion`, `/generate_response`, `/chat`, `/reset_session`, `/health`
(pre-`/api/` paths) kept working, not removed.

## What changed in v13 vs v12

See `evaluation/v12_vs_v13.md` for the full table with evidence. Headline:
a real conversation-memory bug found and fixed (pending-clarification
resolution — "نفسي" was being reclassified from zero instead of answering
the previous question), Quran audio endpoint corrected to the brief's exact
working pattern, 1,798 genuine multi-turn conversations built and audited
(vs. 5 genuine ones hiding in v12's 3,214-record file), COMFORT_BANK grown
from 8 to 15 lines/emotion, nanoGPT retrained (same architecture/corpus —
acceptance rate unchanged within noise, ~22-23%, confirming the bottleneck
is the from-scratch 1.8M-param model itself, not this data), a 64-test
automated suite added, and stale evaluation outputs archived under
`evaluation/archive/` with `v13_run_metadata.json` pinning what produced the
current numbers.

**Classifier and comfort-bank targets are unchanged from v12 by design**
(brief section 3: the classifier wasn't the bottleneck) — see Limitations.

## Known limitations (still true in v13, read before deploying)

See `evaluation/v12_vs_v13.md` "Remaining limitations" for the full list
with reasoning. Headline:

1. **Quran audio is still genuinely untested.** No network path to
   `quran.foundation` exists in this environment. Endpoint pattern is now
   correct per the brief's own specification, but has never been executed
   against the live API.
2. **nanoGPT generation acceptance rate is ~22-23% (n=100), largely
   near-verbatim reproduction of curated lines, not novel generalization.**
   Retraining from scratch this session (same architecture/corpus) produced
   the same rate within noise — this is strong evidence the ceiling is the
   1.8M-parameter, ~500K-character, from-scratch char-level model itself,
   not the data or the quality gate. A pretrained base model is the real
   fix and remains blocked by network access in this sandbox. Tokenizer
   (char vs. subword) and architecture-size experiments (brief sections
   13-14) were **not run this pass** — flagged honestly rather than
   claimed.
3. **`conversations_dataset_v13.json`'s 1,798 conversations are
   programmatically assembled**, not organically collected: real single-turn
   examples for turn 1, hand-written reviewed follow-ups for turn 2 (see
   `data/multiturn_banks.py` docstring for the exact method and the honest
   repetition rate this implies for turn-2 lines).
4. **Classifier is unchanged from v12** (91% accuracy / 0.92 macro F1,
   `evaluation/classifier_report.txt` still says "v10" in its header from
   an earlier pass — label is stale, the model behind it is what v12/v13
   both actually use, unretrained since).
5. **Safety trigger list is a small, disclosed-incomplete keyword list**,
   not a real crisis-detection model.
6. **Frontend needs a locally-running backend** — no hosted server exists
   behind this prototype.
7. **Grouped train/val splitting by near-duplicate family (brief section
   15) was not implemented this pass** — the existing random split by row
   remains in `model/prepare_corpus.py`. Flagged, not silently skipped.
