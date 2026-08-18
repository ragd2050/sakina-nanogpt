# SAKINA v12 — What actually changed, with evidence

Every claim below was tested during this session; the test is named so it
can be re-run. See `evaluation/v12_evaluation_report.md` for the formal
A-E evaluation and `evaluation/comparison_report.md` for the v10→v11
history this builds on.

## Before (v11) vs After (v12)

| | v11 | v12 |
|---|---|---|
| Frontend | fake demo (`chatDemoOrder`, `setTimeout`, canned responses) | real `fetch()` to `/api/chat`, no demo logic left (verified: `grep` for `chatDemoOrder`/`CLARIFYING_TRIGGERS` in frontend/index.html returns nothing) |
| Emotions | 11 | 12 (شعور بالظلم restored — 200 *real* examples recovered from the original dataset, not rewritten synthetically) |
| Quran audio | none | `/api/reciters`, `/api/audio` implemented to spec — **untested against the real API, no network access to it here; degrades to a clear error instead of crashing** (verified via Flask test client) |
| Ambiguous short words | fixed keyword list only | lexical-anchor layer (deterministic, checked first) + confidence/margin clarification, thresholds tuned against a validation sweep |
| Quran verse accuracy | verses shown as complete without checking | audited all 19 verses by hand; **10 of 19 were excerpts, now correctly flagged** `is_excerpt` and labeled in both API and UI |
| Quality gate | caught repeats/control-tokens/length | + repeated whole words, standalone stray letters (before *and* before punctuation — found a real regex gap while testing), incomplete-generation detection, vocabulary-validity check, cross-emotion contamination check. Verified against the exact "ب بابداية" example from the brief: now rejected. |
| Safety | none | dedicated self-harm/crisis detection, bypasses the entire comfort/verse pipeline, tested |
| COMFORT_BANK | 4 lines/emotion | 8 lines/emotion (not the requested 10-20 — see Limitations) |
| Dataset | 2,755 examples, 11 classes | 3,209 examples, 12 classes (not the requested 8,000-10,000 — see Limitations) |
| API | `/api/chat` returning v11-shape JSON | matches the exact requested schema: `success`, `needs_clarification`, nested `verse` object with `is_excerpt`, etc. (verified via Flask test client, all three response branches) |
| Response structure | always 5 sections | short inputs get a shorter response; reflection is dropped for very short messages rather than padded in |

## Files modified this pass
- `data/knowledge.py` — full rewrite: 12th emotion, expanded COMFORT_BANK, lexical anchors, safety triggers
- `data/normalize.py` — taa marbuta normalization now opt-in (was silently on in v11), punctuation normalization added
- `data/quran_verses.json` — full rewrite with audited `is_excerpt` flags
- `backend/emotion_classifier.py` — char n-grams (2,4)→(3,5), `class_weight="balanced"` re-enabled and re-verified safe
- `backend/response_composer.py` — lexical anchor check, safety check, margin-tuned clarification, excerpt-aware verse output, variable-length response composition
- `backend/app.py` — full rewrite to the new `/api/*` schema
- `model/inference.py` — quality gate hardened (4 new checks), checkpoint path → v12
- `model/train.py` — checkpoint names → v12, budget → 2200 iterations
- `frontend/index.html` — all demo chat logic removed, real backend integration, real (not fake) audio player component, `is_excerpt` display

## Files added this pass
- `backend/quran_api.py` — Quran Foundation OAuth2 + audio lookup (untested, see file docstring)
- `data/build_dataset_v12.py` — restores real injustice data, targeted weak-class augmentation
- `evaluation/evaluate_v12.py`, `evaluation/v12_evaluation_report.md` — sections A-E
- `requirements.txt`

## Sample API output (real, from Flask test client this session)
```json
POST /api/chat {"message": "أنا خايفة من الاختبار", "session_id": "s1"}
{
  "success": true, "session_id": "s1", "needs_clarification": false,
  "emotion": "خوف", "confidence": 0.862,
  "response": "خطوة صغيرة رغم الخوف أفضل من الجمود الكامل.",
  "verse": {"id": "20:46", "text": "لَا تَخَافَا إِنَّنِي مَعَكُمَا أَسْمَعُ وَأَرَىٰ",
            "surah_name": "طه", "ayah_number": 46, "is_excerpt": false},
  "reflection": "المؤمن لا يُطلب منه ألا يخاف...",
  "follow_up_question": "ما الذي تخاف منه تحديداً هذه الأيام؟"
}
```
```json
POST /api/chat {"message": "طفشان"}
{"success": true, "needs_clarification": true,
 "question": "هل هو مجرد ملل، أم أنك تشعر بثقل أو ضيق هذه الفترة؟"}
```

## Remaining limitations (read before deploying)

1. **Quran audio is code-complete but unverified.** No network path to any
   Quran API exists in this sandbox. Set real `QF_CLIENT_ID`/`QF_CLIENT_SECRET`
   and test `backend/quran_api.py` against the live API before trusting it —
   field names in the response parsing are a best guess from documentation.
2. **The frontend needs a running backend to do anything real.** Open
   `frontend/index.html`, run `python3 backend/app.py` locally, and the chat
   header shows a live 🟢/🔴 status. There is no hosted server behind this.
3. **Dataset is 3,209 examples, not 8,000-10,000.** Grew it honestly (real
   restored data + targeted augmentation for weak classes) rather than pad
   with more combinatorial templates, which the v11 report already showed
   teaches a small classifier less per example past a point. Real growth
   needs real data collection.
4. **nanoGPT generation acceptance rate is 18% (n=100), most of which is
   near-verbatim reproduction of the 8 curated lines per emotion, not new
   generalization.** This is unchanged in kind from v11 — more data helped
   the *classifier* a lot; it has not made the generator meaningfully more
   creative, because the underlying constraint (1.8M params, trained from
   scratch, ~500K characters) hasn't changed. A pretrained base model is
   the real fix and remains blocked by network access in this sandbox.
5. **Quran verse excerpt audit was done from the model's own knowledge, not
   a live-verified source** (same network constraint). Recommended before
   production: cross-check all 19 entries in `data/quran_verses.json`
   against Tanzil.net or the Quran Foundation API directly.
6. **Safety trigger list is a small, disclosed-incomplete keyword list**,
   not a real crisis-detection model. Treat it as a minimum floor, not a
   substitute for a proper safety review before any real deployment.
7. **The 400-example threshold-tuning validation set was drawn from the
   training distribution** (only 3/400 misclassified), so the
   confidence/margin thresholds are informed more by the manual
   sanity-check suite (B/C sections above) than by that sweep — disclosed
   in `backend/response_composer.py` comments.
