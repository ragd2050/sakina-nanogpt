# -*- coding: utf-8 -*-
"""
Builds v12 of the flat classifier dataset on top of v11's
emotions_dataset.json, which already contains the v10 base + v11
augmentation for 11 emotions. This script adds:

1. The 200 REAL شعور بالظلم examples from the original project data,
   which v10 had dropped (DROP_EMOTIONS) -- restoring real, previously
   human/dataset-sourced examples is much better than writing 200 more
   synthetic ones from scratch, so that's what this does.
2. Explicit example sentences using the requested injustice phrases
   ("انظلمت", "ظلموني", "أخذ حقي", "حسبي الله عليه", etc.) not already
   covered by #1.
3. Additional combinatorial augmentation targeted specifically at the
   weakest classes per the brief (فقدان أمل / خوف / غضب / ضغط نفسي), using
   a wider template set than v11 to reduce the risk of the model
   memorizing the small comfort-line set (this was the honest finding
   from the v11 report) -- more input variety per fixed target, not more
   duplication of the same shape.
"""
import json
import itertools
import random
from pathlib import Path

from knowledge import EMOTION_VERSES, COMFORT_BANK

ROOT = Path(__file__).resolve().parent
random.seed(43)

TAGS = {
    "قلق وتوتر": "ANX", "حزن": "SAD", "فقدان أمل": "HOPELESS", "وحدة": "LONE",
    "حيرة وتشتت": "CONF", "ذنب وتقصير": "GUILT", "فقد وشوق": "LOSS",
    "فرح وشكر": "THANK", "خوف": "FEAR", "غضب": "ANGER", "ضغط نفسي": "STRESS",
    "شعور بالظلم": "INJUSTICE",
}

INJUSTICE_ANCHORS = [
    "انظلمت بشكل ما توقعته", "ظلموني وأنا ساكت", "أخذ حقي ولا قدرت أسوي شي",
    "حسبي الله عليه ونعم الوكيل", "اتهموني بشي ما سويته", "هذا ظلم واضح ومو طبيعي",
    "ما أنصفوني بالقرار اللي أخذوه", "حقي ضاع وماحد وقف معي", "عاملوني بظلم قدام الكل",
    "أشعر إن اللي صار معي ظلم بكل معنى الكلمة", "حسيت بغبن كبير من القرار اللي أخذوه بحقي",
    "اتهموني بشي أنا بريء منه تماماً", "أخذوا حقي بدون أي مبرر", "ظلمتني الظروف والناس مع بعض",
]

# Wider template variety for the weakest 4 classes (more distinct sentence
# shapes, not just synonym-swaps within the same shape).
EXTRA_TEMPLATES = [
    "بصراحة {feeling} {reason}، ومو عارف اتصرف كيف",
    "كل ما أحاول أتجاوز الموضوع، أرجع {feeling} {reason}",
    "اليوم بالذات حسيت اني {feeling} أكثر من العادة، بسبب {reason}",
    "ودي أحد يفهم اني {feeling} {reason} بس محد يدري",
    "من الصبح وأنا {feeling}، السبب {reason}",
]

FEELINGS = {
    "فقدان أمل": ["يائس", "يائسة", "محبط", "محبطة", "منهزم", "منهزمة"],
    "خوف": ["خايف", "خايفة", "مرعوب", "مرعوبة", "متوجس", "متوجسة"],
    "غضب": ["غضبان", "غضبانة", "متضايق", "متضايقة", "مستاء", "مستاءة"],
    "ضغط نفسي": ["مضغوط", "مضغوطة", "مرهق", "مرهقة", "منهك", "منهكة"],
}
REASONS = {
    "فقدان أمل": ["إن الأمور بتتغير", "من المحاولة المتكررة بلا نتيجة", "إني أشوف نتيجة لتعبي", "من نفس الدائرة المكررة"],
    "خوف": ["من قرار مصيري قدامي", "من فقدان شيء مهم", "من المستقبل الغامض", "من رد فعل الناس"],
    "غضب": ["من موقف مو عادل", "من تجاهل مستمر لي", "من وعد ما انوفى فيه", "من طريقة تعاملهم معي"],
    "ضغط نفسي": ["من تراكم الالتزامات", "من قلة الوقت للراحة", "من توقعات الكل مني", "من جدول ما فيه فراغ"],
}


def augment_weak_classes():
    rows = []
    for emo in FEELINGS:
        combos = list(itertools.product(EXTRA_TEMPLATES, FEELINGS[emo], REASONS[emo]))
        random.shuffle(combos)
        for template, feeling, reason in combos[:60]:
            text = template.format(feeling=feeling, reason=reason)
            rows.append((text, emo))
    seen, unique = set(), []
    for text, emo in rows:
        if text not in seen:
            seen.add(text)
            unique.append((text, emo))
    return unique


def main():
    with open(ROOT / "emotions_dataset.json", encoding="utf-8") as f:
        existing = json.load(f)
    with open(ROOT / "existing_labeled_examples.json", encoding="utf-8-sig") as f:
        original_full = json.load(f)

    counters = {e: sum(1 for r in existing if r["detected_emotion"] == e) for e in TAGS}
    counters["شعور بالظلم"] = 0
    new_rows = []

    # 1. restore the 200 real injustice examples
    verses, comforts = EMOTION_VERSES["شعور بالظلم"], COMFORT_BANK["شعور بالظلم"]
    for row in original_full:
        if row.get("detected_emotion") != "شعور بالظلم":
            continue
        idx = counters["شعور بالظلم"]
        counters["شعور بالظلم"] += 1
        new_rows.append({
            "user_input": row["user_input"], "detected_emotion": "شعور بالظلم", "emotion_tag": "INJUSTICE",
            "verse_id": verses[idx % len(verses)], "comfort_message": comforts[idx % len(comforts)],
            "source": "Sakina_v10_restored_original",
        })

    # 2. explicit injustice phrase anchors
    for text in INJUSTICE_ANCHORS:
        idx = counters["شعور بالظلم"]
        counters["شعور بالظلم"] += 1
        new_rows.append({
            "user_input": text, "detected_emotion": "شعور بالظلم", "emotion_tag": "INJUSTICE",
            "verse_id": verses[idx % len(verses)], "comfort_message": comforts[idx % len(comforts)],
            "source": "Sakina_v12_injustice_anchor",
        })

    # 3. wider-template augmentation for the 4 weakest classes
    for text, emo in augment_weak_classes():
        v, c = EMOTION_VERSES[emo], COMFORT_BANK[emo]
        idx = counters[emo]
        counters[emo] += 1
        new_rows.append({
            "user_input": text, "detected_emotion": emo, "emotion_tag": TAGS[emo],
            "verse_id": v[idx % len(v)], "comfort_message": c[idx % len(c)],
            "source": "Sakina_v12_augmented",
        })

    combined = existing + new_rows
    with open(ROOT / "emotions_dataset.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print(f"v11 dataset: {len(existing)} examples")
    print(f"v12 additions: {len(new_rows)} "
          f"(200 restored real injustice + {len(INJUSTICE_ANCHORS)} injustice anchors + "
          f"{len(new_rows) - 200 - len(INJUSTICE_ANCHORS)} weak-class augmentation)")
    print(f"v12 total: {len(combined)} examples")
    print(Counter(r["detected_emotion"] for r in combined))


if __name__ == "__main__":
    main()
