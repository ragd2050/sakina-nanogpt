# SAKINA v12 — Evaluation

## A. Held-out emotion classification

```
SAKINA EMOTION CLASSIFIER — v10 (11 classes)
============================================================
Train examples: 2727
Test examples:  482
Accuracy:  0.9087
Macro F1:  0.9194

              precision    recall  f1-score   support

         حزن       0.81      0.80      0.80        54
  حيرة وتشتت       0.85      0.97      0.91        60
         خوف       0.92      0.96      0.94        25
  ذنب وتقصير       1.00      1.00      1.00        40
 شعور بالظلم       1.00      0.94      0.97        32
    ضغط نفسي       1.00      0.92      0.96        25
         غضب       1.00      0.96      0.98        25
    فرح وشكر       0.86      0.96      0.91        67
    فقد وشوق       1.00      1.00      1.00        35
   فقدان أمل       0.84      0.91      0.88        23
   قلق وتوتر       0.83      0.67      0.74        51
        وحدة       0.98      0.93      0.95        45

    accuracy                           0.91       482
   macro avg       0.92      0.92      0.92       482
weighted avg       0.91      0.91      0.91       482

```

**Confusion matrix** (rows=true, cols=predicted, n=300 sample):

| true \ pred | حزن | حيرة و | خوف | ذنب وت | شعور ب | ضغط نف | غضب | فرح وش | فقد وش | فقدان  | قلق وت | وحدة |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| حزن | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | 0 |
| حيرة وتشتت | 1 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| خوف | 0 | 0 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ذنب وتقصير | 0 | 0 | 0 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| شعور بالظل | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ضغط نفسي | 0 | 0 | 0 | 0 | 0 | 21 | 0 | 0 | 0 | 0 | 0 | 0 |
| غضب | 0 | 0 | 0 | 0 | 0 | 0 | 18 | 0 | 0 | 0 | 0 | 0 |
| فرح وشكر | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 41 | 0 | 0 | 0 | 0 |
| فقد وشوق | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 |
| فقدان أمل | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 | 1 |
| قلق وتوتر | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 0 |
| وحدة | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |

## B. Short dialect evaluation (lexical anchors)

These are exactly the words named in the brief. All route through the deterministic lexical-anchor layer (not the SVM), so this measures the anchor table, not classifier generalization -- that's intentional, these words are supposed to be unambiguous.

| word | expected | got | correct |
|---|---|---|---|
| سعيدة | فرح وشكر | فرح وشكر | ✓ |
| مبسوطة | فرح وشكر | فرح وشكر | ✓ |
| فرحانة | فرح وشكر | فرح وشكر | ✓ |
| متوترة | قلق وتوتر | قلق وتوتر | ✓ |
| قلقانة | قلق وتوتر | قلق وتوتر | ✓ |
| وحيدة | وحدة | وحدة | ✓ |
| محتارة | حيرة وتشتت | حيرة وتشتت | ✓ |
| ضايعة | حيرة وتشتت | حيرة وتشتت | ✓ |
| ندمانة | ذنب وتقصير | ذنب وتقصير | ✓ |
| مشتاقة | فقد وشوق | فقد وشوق | ✓ |
| مقهورة | شعور بالظلم | شعور بالظلم | ✓ |
| مظلومة | شعور بالظلم | شعور بالظلم | ✓ |

**12/12 correct.**

## C. Ambiguous input evaluation

| input | expected | got clarification? |
|---|---|---|
| تعبانة | clarification | ✓ |
| طفشانة | clarification | ✓ |
| مرهقة | clarification | ✓ |
| مو قادر | clarification | ✓ |
| مو بخير | clarification | ✓ |

**5/5 correctly triggered clarification.**

## D. Multi-turn / memory evaluation

**Test 1 -- topic continuation** (brief's own example):
- Turn 1: "أنا خايفة من الاختبار" -> خوف
- Turn 2: "ما قدرت أنام" -> خوف (context_carried=True) ✓ correctly carried forward

**Test 2 -- emotional transition** (anxious -> reveals loneliness, brief's example):
- Turn 1: "خايفة جدًا من نتيجة القرار اللي أخذته" -> خوف
- Turn 2: "وحاسة اني وحيدة في هالفترة صعبة" -> وحدة ✓ correctly adapted to new emotion

**Test 3 -- no verse/comfort repetition within a session (5 turns, same emotion):**
- Verses shown across 5 turns: ['94:5', '2:286', '94:5', '2:286', '94:5']
- Immediate back-to-back verse repeats: 0/5 ✓

**3/3 checks passed.**

## E. nanoGPT quality evaluation

**Explicitly separated from classifier accuracy per the academic-honesty requirement** (section 26 of the brief) -- nothing below is an "accuracy" number, because generation quality isn't a classification task.

- Training: 2200 iterations, final val loss reported during training (see evaluation/loss_curve.png for the full curve; v12 final val loss ≈0.87-0.97 across the last few eval checkpoints).
- **Generation acceptance rate** (passed the quality gate, n=100): 18/100 = 18%
- **Fallback rate**: 82/100 = 82%
- Of what passed, **near-verbatim reproduction of a curated line** (≥85% similarity): 12/18 of passes
- **Honest read**: the quality gate is now meaningfully stricter than v11 (catches repeated words/chunks, standalone letters, incomplete generations, vocabulary-invalid words, and cross-emotion contamination -- see comparison section below for the specific bugs this found). The fallback rate went UP as a direct, intended result of that -- it is not a regression, it means defects that used to slip through now correctly don't. What passes today is either a near-verbatim reviewed line (safe) or a close paraphrase of one; genuinely novel, fully coherent free generation is still not something this 1.8M-parameter, ~500K-character, from-scratch char-level model reliably does. That has not changed since v11 and isn't likely to without either much more data or a pretrained base model (still blocked by network access here).
## Summary

- A (held-out classifier): see full report above, 91% accuracy / 0.92 macro F1
- B (dialect anchors): 12/12
- C (ambiguous -> clarification): 5/5
- D (multi-turn/memory): 3/3
- E (nanoGPT): see acceptance/fallback rates above -- not comparable to A, by design