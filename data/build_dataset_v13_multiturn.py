# -*- coding: utf-8 -*-
"""
Builds data/conversations_dataset_v13.json: genuine User->Assistant->
User->Assistant conversations (see evaluation/analyze_conversation_dataset.py
for the definition and for the audit showing v12's conversations_dataset.json
was 99.8% single-turn despite its shape).

Method (also documented in data/multiturn_banks.py and
evaluation/v12_vs_v13.md): turn 1 is a REAL single-turn example from
data/emotions_dataset.json, used unmodified. Turn 2 is a hand-written,
reviewed follow-up (data/multiturn_banks.py) chosen either as a same-emotion
continuation or a transition into a second plausible emotion, paired against
many distinct real turn-1 examples.

This is honestly disclosed as programmatic assembly of reviewed material,
not organically collected dialogue and not model-generated paraphrasing.
"""
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiturn_banks import STABLE_FOLLOWUPS, TRANSITIONS, TRANSITION_FOLLOWUPS

ROOT = Path(__file__).resolve().parent.parent
SINGLE_TURN_FILE = ROOT / "data" / "emotions_dataset.json"
OUT_FILE = ROOT / "data" / "conversations_dataset_v13.json"

random.seed(13)

# Roughly this fraction of eligible first turns become a "transition"
# conversation instead of a same-emotion "stable" one.
TRANSITION_RATIO = 0.4

# The brief asks for >=1,000, preferably 1,500-2,000 genuine conversations
# "without poor-quality duplication" rather than maximizing count. All
# 3,209 single-turn source rows are technically eligible (every emotion has
# a follow-up bank), so we stratified-sample down to TARGET_SIZE per emotion
# rather than keep all of them, to stay in the requested range instead of
# over-multiplying a small hand-written follow-up pool.
TARGET_SIZE = 1800


def main():
    with open(SINGLE_TURN_FILE, encoding="utf-8") as f:
        single_turns = json.load(f)

    random.shuffle(single_turns)
    from collections import Counter
    counts = Counter(r["detected_emotion"] for r in single_turns)
    total = len(single_turns)
    # proportional per-emotion quota, at least 1 per emotion present
    quota = {e: max(1, round(TARGET_SIZE * c / total)) for e, c in counts.items()}
    taken = Counter()
    sampled = []
    for row in single_turns:
        e = row["detected_emotion"]
        if taken[e] < quota[e]:
            sampled.append(row)
            taken[e] += 1
    single_turns = sampled

    conversations = []
    skipped_no_bank = 0

    for row in single_turns:
        emotion = row["detected_emotion"]
        stable_pool = STABLE_FOLLOWUPS.get(emotion, [])
        transitions = TRANSITIONS.get(emotion, [])

        use_transition = (
            transitions
            and random.random() < TRANSITION_RATIO
            and any(TRANSITION_FOLLOWUPS.get((emotion, t)) for t in transitions)
        )

        if use_transition:
            eligible = [t for t in transitions if TRANSITION_FOLLOWUPS.get((emotion, t))]
            target_emotion = random.choice(eligible)
            pool = TRANSITION_FOLLOWUPS[(emotion, target_emotion)]
            user2, asst2 = random.choice(pool)
            final_emotion = target_emotion
            conv_type = "transition"
        elif stable_pool:
            user2, asst2 = random.choice(stable_pool)
            final_emotion = emotion
            conv_type = "stable"
        else:
            skipped_no_bank += 1
            continue

        conversations.append({
            "emotion": emotion,
            "final_emotion": final_emotion,
            "conversation_type": conv_type,
            "verse_id": row.get("verse_id"),
            "source": row.get("source", "unspecified") + "+multiturn_bank_v13",
            "conversation": [
                {"role": "user", "text": row["user_input"]},
                {"role": "assistant", "text": row["comfort_message"]},
                {"role": "user", "text": user2},
                {"role": "assistant", "text": asst2},
            ],
        })

    # De-duplicate exact (turn1, turn2) combinations just in case the same
    # first turn appears twice in the source data.
    seen = set()
    deduped = []
    for c in conversations:
        key = tuple(m["text"] for m in c["conversation"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    OUT_FILE.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")

    n_stable = sum(1 for c in deduped if c["conversation_type"] == "stable")
    n_transition = sum(1 for c in deduped if c["conversation_type"] == "transition")
    print(f"Built {len(deduped)} genuine multi-turn conversations "
          f"({n_stable} stable, {n_transition} transition). "
          f"Skipped {skipped_no_bank} (no bank for that emotion).")
    print(f"Wrote {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
