# -*- coding: utf-8 -*-
"""
Builds v11 of the dataset on top of v10's emotions_dataset.json.

Two honest additions (not a claim of 10,000 hand-labeled examples -- see
the printed report at the end and evaluation/comparison_report.md for the
real number and why):

1. Explicit coverage of the dialect/spelling-variant words requested
   (قلقانة، متوترة، زعلانة، مبسوطة، وحيدة، مقهورة، محتارة، ضايعة، ندمانة،
   مشتاقة, etc.) as clean anchor sentences.

2. Template x feeling-word x reason-clause combinatorial augmentation:
   every generated sentence is a grammatically real, distinct Arabic
   sentence (not a duplicate/paraphrase-by-synonym-swap of one seed) --
   this is a legitimate augmentation technique for small NLU datasets,
   not padding. Counts are deduplicated and reported honestly.

Also emits data/conversations_dataset.json in the requested
{"conversation":[...], "emotion":..., "verse_id":...} shape, including a
modest set of genuinely-authored multi-turn conversations (a full 10K
multi-turn corpus is not something one pass can honestly hand-author --
see the report for what's real here vs. the original ask).
"""
import json
import itertools
import random
from pathlib import Path

from knowledge import EMOTION_VERSES, COMFORT_BANK, CONTINUATION_BANK

ROOT = Path(__file__).resolve().parent
random.seed(42)

TAGS = {
    "قلق وتوتر": "ANX", "حزن": "SAD", "فقدان أمل": "HOPELESS", "وحدة": "LONE",
    "حيرة وتشتت": "CONF", "ذنب وتقصير": "GUILT", "فقد وشوق": "LOSS",
    "فرح وشكر": "THANK", "خوف": "FEAR", "غضب": "ANGER", "ضغط نفسي": "STRESS",
}

# ---------------------------------------------------------------------
# 1. Explicit dialect/spelling-variant anchor sentences
# ---------------------------------------------------------------------
SYNONYM_ANCHORS = {
    "قلق وتوتر": ["قلقانة من بكرة", "قلقان مرة هالأيام", "متوترة من كل شي قدامي", "متوتر ومو قادر اهدأ"],
    "حزن": ["زعلانة من اللي صار", "زعلان بشكل ما قدرت اوصفه", "مقهورة من الموقف", "مقهور من اللي حصل"],
    "فرح وشكر": ["مبسوطة اليوم بشكل حلو", "مبسوط من خبر وصلني", "سعيدة والحمدلله على كل شي"],
    "وحدة": ["وحيدة من فترة طويلة", "وحيد حتى لو حولي ناس كثير"],
    "حيرة وتشتت": ["محتارة بين خيارين صعبين", "محتار وش اختار", "ضايعة مو عارفة اتجه لوين", "ضايع بين قرارات كثيرة"],
    "ذنب وتقصير": ["ندمانة على قرار أخذته", "ندمان على كلام قلته"],
    "فقد وشوق": ["مشتاقة لشخص غالي علي", "مشتاق لأيام فاتت"],
}

# ---------------------------------------------------------------------
# 2. Combinatorial augmentation
# ---------------------------------------------------------------------
FEELINGS = {
    "قلق وتوتر": ["قلقان", "قلقانة", "متوتر", "متوترة"],
    "حزن": ["حزين", "حزينة", "مكسور", "مكسورة"],
    "فقدان أمل": ["يائس", "يائسة", "محبط", "محبطة"],
    "وحدة": ["وحيد", "وحيدة", "معزول", "معزولة"],
    "حيرة وتشتت": ["محتار", "محتارة", "مشتت", "مشتتة"],
    "ذنب وتقصير": ["نادم", "نادمة", "ندمان", "ندمانة"],
    "فقد وشوق": ["مشتاق", "مشتاقة"],
    "فرح وشكر": ["مبسوط", "مبسوطة", "ممتن", "ممتنة"],
    "خوف": ["خايف", "خايفة", "مرعوب", "مرعوبة"],
    "غضب": ["غضبان", "غضبانة", "متضايق", "متضايقة"],
    "ضغط نفسي": ["مضغوط", "مضغوطة", "مرهق", "مرهقة"],
}

REASONS = {
    "قلق وتوتر": ["من الاختبار الجاي", "من المقابلة بكرة", "من نتيجة قريبة", "من كل اللي في بالي", "بدون سبب واضح"],
    "حزن": ["من موقف صار معي", "من كلام جرحني", "من فترة طويلة", "ومو عارف ليش بالضبط"],
    "فقدان أمل": ["ومافي فايدة من المحاولة", "إن الوضع بيتحسن", "من كثر المحاولة بلا نتيجة", "إني أكمل زي كل مرة"],
    "وحدة": ["حتى وسط الناس", "من فترة طويلة", "ومافي حد يفهمني", "أغلب وقتي"],
    "حيرة وتشتت": ["بين قرارين صعبين", "من كثر الخيارات قدامي", "في حياتي هالفترة", "بدون اتجاه واضح"],
    "ذنب وتقصير": ["على قرار أخذته", "مع أهلي هالفترة", "على كلام قلته", "ومو عارف كيف أصلحه"],
    "فقد وشوق": ["لشخص عزيز فقدته", "لأيام ماضية", "من زمان", "كل ما تذكرته"],
    "فرح وشكر": ["بخبر حلو اليوم", "بنعمة كبيرة في حياتي", "بشي كنت متمنيه", "بلحظة صغيرة بس غالية"],
    "خوف": ["من نتيجة بكرة", "من المجهول قدامي", "أخسر حد أحبه", "بدون سبب واضح"],
    "غضب": ["من تصرف صار معي", "من كلام جرحني", "ومو قادر اسيطر عليه", "من ظلم شفته"],
    "ضغط نفسي": ["من الشغل والدراسة", "من كثر المسؤوليات", "هالفترة", "من المواعيد المتراكمة"],
}

TEMPLATES = [
    "أنا {feeling} {reason}",
    "صراحة حاسس اني {feeling} {reason}",
    "من فترة وأنا {feeling} {reason}",
    "بصراحة {feeling} {reason} ومو عارف اسوي شنو",
]


def augment():
    rows = []
    for emo in FEELINGS:
        combos = list(itertools.product(TEMPLATES, FEELINGS[emo], REASONS[emo]))
        random.shuffle(combos)
        for template, feeling, reason in combos[:90]:   # cap per emotion, deduped below
            text = template.format(feeling=feeling, reason=reason)
            rows.append((text, emo))
    # dedupe while preserving order
    seen = set()
    unique = []
    for text, emo in rows:
        if text not in seen:
            seen.add(text)
            unique.append((text, emo))
    return unique


# ---------------------------------------------------------------------
# 3. A modest, genuinely-authored set of multi-turn conversations
#    (per the requested {"conversation": [...]} shape)
# ---------------------------------------------------------------------
MULTI_TURN = [
    {
        "emotion": "قلق وتوتر", "verse_id": "13:28",
        "conversation": [
            {"role": "user", "text": "أنا متوترة من الاختبار الجاي"},
            {"role": "assistant", "text": "أفهم أن القلق قبل الاختبارات متعب، خصوصاً لما يصير تفكيرك كله في النتيجة قبل ما توصلها."},
            {"role": "user", "text": "ما قدرت أنام الليلة من كثر التفكير"},
            {"role": "assistant", "text": "يبدو أن قلة النوم امتداد لنفس التوتر اللي ذكرتيه، جسمك وعقلك متعبين من نفس الحمل."},
        ],
    },
    {
        "emotion": "وحدة", "verse_id": "2:186",
        "conversation": [
            {"role": "user", "text": "حاسة اني وحيدة من فترة"},
            {"role": "assistant", "text": "أفهم ثقل هذا الشعور، خصوصاً لما يطول. مشاعرك تستحق أن تُسمع."},
            {"role": "user", "text": "حتى لما أكون بين أهلي حاسة إني لحالي"},
            {"role": "assistant", "text": "هذا النوع من الوحدة أصعب لأنه مو مرتبط بغياب الناس، بل بشعور إنك مو مفهومة حتى وأنت بينهم."},
        ],
    },
    {
        "emotion": "ضغط نفسي", "verse_id": "2:286",
        "conversation": [
            {"role": "user", "text": "عندي شغل وامتحانات بنفس الأسبوع"},
            {"role": "assistant", "text": "هذا فعلاً حمل كبير يتجمع في وقت واحد. لست مضطراً لإنجاز كل شيء اليوم."},
            {"role": "user", "text": "وحاسس اني بدأت اتعب جسدياً كمان"},
            {"role": "assistant", "text": "الضغط النفسي يترجم لتعب جسدي غالباً، جسدك يحاول يخبرك إنك تحتاج مساحة صغيرة تتنفس فيها."},
        ],
    },
    {
        "emotion": "حزن", "verse_id": "9:40",
        "conversation": [
            {"role": "user", "text": "حاسس بحزن من فترة وما عارف ليش بالضبط"},
            {"role": "assistant", "text": "أفهم أن قلبك متعب. أحياناً الحزن ما يحتاج سبب واضح حتى يكون حقيقي."},
            {"role": "user", "text": "بس مزعجني اني ما اعرف مصدره"},
            {"role": "assistant", "text": "ما تحتاجين تعرفي المصدر الآن حتى تسمحي لنفسك تحسي به. الفهم ممكن يجي بعدين."},
        ],
    },
    {
        "emotion": "خوف", "verse_id": "20:46",
        "conversation": [
            {"role": "user", "text": "خايفة من نتيجة القرار اللي أخذته"},
            {"role": "assistant", "text": "خذي نفساً، وذكّري نفسك أن ما تخشينه أكبر في خيالك غالباً مما سيكون في الواقع."},
            {"role": "user", "text": "قلبي يرتجف كل ما فكرت فيه"},
            {"role": "assistant", "text": "هذا رد فعل جسدي طبيعي للخوف، مو دليل إن القرار غلط. الله معك في هذه اللحظة أكثر مما تتخيلين."},
        ],
    },
]


def build_conversation_dataset():
    convs = []
    for item in MULTI_TURN:
        convs.append(item)
    # wrap every flat single-turn example (existing + augmented) as a
    # 1-turn "conversation" too, so the whole dataset is queryable in one
    # consistent shape.
    with open(ROOT / "emotions_dataset.json", encoding="utf-8") as f:
        flat = json.load(f)
    for row in flat:
        convs.append({
            "emotion": row["detected_emotion"],
            "verse_id": row["verse_id"],
            "conversation": [
                {"role": "user", "text": row["user_input"]},
                {"role": "assistant", "text": row["comfort_message"]},
            ],
        })
    return convs


def main():
    with open(ROOT / "emotions_dataset.json", encoding="utf-8") as f:
        existing = json.load(f)

    counters = {e: sum(1 for r in existing if r["detected_emotion"] == e) for e in TAGS}
    new_rows = []

    for emo, sentences in SYNONYM_ANCHORS.items():
        verses = EMOTION_VERSES[emo]
        comforts = COMFORT_BANK[emo]
        for text in sentences:
            idx = counters[emo]
            counters[emo] += 1
            new_rows.append({
                "user_input": text, "detected_emotion": emo, "emotion_tag": TAGS[emo],
                "verse_id": verses[idx % len(verses)], "comfort_message": comforts[idx % len(comforts)],
                "source": "Sakina_v11_synonym_anchor",
            })

    for text, emo in augment():
        verses = EMOTION_VERSES[emo]
        comforts = COMFORT_BANK[emo]
        idx = counters[emo]
        counters[emo] += 1
        new_rows.append({
            "user_input": text, "detected_emotion": emo, "emotion_tag": TAGS[emo],
            "verse_id": verses[idx % len(verses)], "comfort_message": comforts[idx % len(comforts)],
            "source": "Sakina_v11_augmented",
        })

    combined = existing + new_rows
    with open(ROOT / "emotions_dataset.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    convs = build_conversation_dataset()
    with open(ROOT / "conversations_dataset.json", "w", encoding="utf-8") as f:
        json.dump(convs, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print(f"v10 dataset: {len(existing)} examples")
    print(f"v11 additions: {len(new_rows)} examples "
          f"({len(SYNONYM_ANCHORS and [s for v in SYNONYM_ANCHORS.values() for s in v])} synonym anchors + "
          f"{len(new_rows) - sum(len(v) for v in SYNONYM_ANCHORS.values())} combinatorial augmented)")
    print(f"v11 total flat dataset: {len(combined)} examples")
    print(Counter(r["detected_emotion"] for r in combined))
    print(f"Conversation-format dataset: {len(convs)} entries "
          f"({len(MULTI_TURN)} genuinely multi-turn, {len(convs) - len(MULTI_TURN)} single-turn wrapped)")


if __name__ == "__main__":
    main()
