# -*- coding: utf-8 -*-
"""
Audits data/conversations_dataset.json and reports honest statistics about
how many records are GENUINE multi-turn conversations, as opposed to a
single User->Assistant pair that happens to be stored in the same shape.

Definition used here (per the v13 brief): a genuine multi-turn conversation
must contain at least the pattern
    User -> Assistant -> User -> Assistant
i.e. at least 4 messages, alternating, starting with the user, containing
at least two user turns and two assistant turns.

Run:
    python3 evaluation/analyze_conversation_dataset.py

Writes:
    evaluation/conversation_dataset_report.json
"""
import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def is_genuine_multiturn(conv) -> bool:
    """User -> Assistant -> User -> Assistant, at minimum."""
    if not conv or len(conv) < 4:
        return False
    if conv[0].get("role") != "user":
        return False
    roles = [m.get("role") for m in conv]
    # must strictly alternate user/assistant
    for i, r in enumerate(roles):
        expected = "user" if i % 2 == 0 else "assistant"
        if r != expected:
            return False
    user_turns = roles.count("user")
    assistant_turns = roles.count("assistant")
    return user_turns >= 2 and assistant_turns >= 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/conversations_dataset.json",
                         help="Path (relative to project root) of the conversations JSON to audit")
    parser.add_argument("--out", default=None,
                         help="Output report path (relative to project root). Defaults next to --file's name.")
    args = parser.parse_args()

    data_file = ROOT / args.file
    out_file = ROOT / args.out if args.out else ROOT / "evaluation" / f"conversation_dataset_report__{data_file.stem}.json"

    with open(data_file, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    turn_counts = []
    bucket = Counter()          # 1-turn, 2-turn, 3-turn, 4+-turn (message-pair based)
    genuine_count = 0
    emotion_dist = Counter()
    source_dist = Counter()
    seen_conversations = Counter()
    all_user_prompts = []

    for rec in records:
        conv = rec.get("conversation", [])
        n_messages = len(conv)
        turn_counts.append(n_messages)

        # bucket by number of user turns (a "turn" here = one user message)
        n_user_turns = sum(1 for m in conv if m.get("role") == "user")
        if n_user_turns <= 1:
            bucket["1-turn"] += 1
        elif n_user_turns == 2:
            bucket["2-turn"] += 1
        elif n_user_turns == 3:
            bucket["3-turn"] += 1
        else:
            bucket["4+-turn"] += 1

        if is_genuine_multiturn(conv):
            genuine_count += 1

        emotion_dist[rec.get("emotion", "UNKNOWN")] += 1
        source_dist[rec.get("source", "unspecified")] += 1

        key = tuple((m.get("role"), m.get("text", "").strip()) for m in conv)
        seen_conversations[key] += 1

        for m in conv:
            if m.get("role") == "user":
                all_user_prompts.append(m.get("text", "").strip())

    duplicate_conversations = sum(c - 1 for c in seen_conversations.values() if c > 1)
    unique_user_prompts = len(set(all_user_prompts))

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(data_file.relative_to(ROOT)),
            "definition_of_genuine_multiturn": "User->Assistant->User->Assistant (>=2 user turns, >=2 assistant turns, strictly alternating, starts with user)",
        },
        "total_conversation_records": total,
        "genuine_multiturn_count": genuine_count,
        "genuine_multiturn_ratio": round(genuine_count / total, 4) if total else 0.0,
        "turn_buckets_by_user_turns": dict(bucket),
        "average_messages_per_conversation": round(statistics.mean(turn_counts), 3) if turn_counts else 0,
        "median_messages_per_conversation": statistics.median(turn_counts) if turn_counts else 0,
        "unique_user_prompts": unique_user_prompts,
        "total_user_prompts": len(all_user_prompts),
        "duplicate_conversation_records": duplicate_conversations,
        "emotion_distribution": dict(sorted(emotion_dist.items(), key=lambda x: -x[1])),
        "source_distribution": dict(sorted(source_dist.items(), key=lambda x: -x[1])),
    }

    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total records:               {total}")
    print(f"Genuine multi-turn (>=U-A-U-A): {genuine_count} ({report['genuine_multiturn_ratio']*100:.1f}%)")
    print(f"Turn buckets (by # user turns): {dict(bucket)}")
    print(f"Avg messages/conversation:   {report['average_messages_per_conversation']}")
    print(f"Median messages/conversation:{report['median_messages_per_conversation']}")
    print(f"Unique user prompts:         {unique_user_prompts} / {len(all_user_prompts)} total")
    print(f"Duplicate conversation records: {duplicate_conversations}")
    print(f"Emotion distribution:        {dict(sorted(emotion_dist.items(), key=lambda x: -x[1]))}")
    print(f"Source distribution:         {dict(sorted(source_dist.items(), key=lambda x: -x[1]))}")
    print(f"\nWrote {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
