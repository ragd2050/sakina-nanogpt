# -*- coding: utf-8 -*-
"""
Turns data/emotions_dataset.json into the <USER>/<EMOTION>/<COMFORT> corpus
the comfort-generation model trains on, and builds the char-level tokenizer
meta.pkl from that corpus.

Kept in the same "chat-cell" format the v9 model already used, so the
prompting convention in inference.py stays consistent.
"""
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "emotions_dataset.json"
CKPT_DIR = ROOT / "checkpoints"

VAL_RATIO = 0.1
random.seed(42)


def build_cell(user_input: str, emotion: str, comfort: str) -> str:
    return (
        "<USER>\n" + user_input.strip() +
        "\n\n<EMOTION>\n" + emotion.strip() +
        "\n\n<COMFORT>\n" + comfort.strip() +
        "\n<END>\n\n"
    )


def main():
    with open(DATA_FILE, encoding="utf-8") as f:
        rows = json.load(f)

    random.shuffle(rows)
    n_val = int(len(rows) * VAL_RATIO)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    train_text = "".join(build_cell(r["user_input"], r["detected_emotion"], r["comfort_message"]) for r in train_rows)
    val_text = "".join(build_cell(r["user_input"], r["detected_emotion"], r["comfort_message"]) for r in val_rows)

    CKPT_DIR.mkdir(exist_ok=True, parents=True)
    (CKPT_DIR / "train_corpus.txt").write_text(train_text, encoding="utf-8")
    (CKPT_DIR / "val_corpus.txt").write_text(val_text, encoding="utf-8")

    tokenizer = CharTokenizer.build_from_text(train_text + val_text)
    tokenizer.save(CKPT_DIR / "meta.pkl")

    print(f"train chars: {len(train_text):,}  val chars: {len(val_text):,}  vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
