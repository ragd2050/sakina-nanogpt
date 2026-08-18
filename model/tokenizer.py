# -*- coding: utf-8 -*-
"""
Character-level tokenizer for the Sakina comfort-generation model.

Char-level (not word/subword) is a deliberate choice given the corpus size
(a few thousand short Arabic examples): a word-level vocab would leave most
words seen only a handful of times, which is worse for a from-scratch model
this small. Char-level keeps the vocab tiny (~100-200 symbols) and lets the
model at least spell Arabic correctly, even though it caps how "creative"
generation can be -- see checkpoints/README for the honest tradeoff.
"""
import pickle
from pathlib import Path


class CharTokenizer:
    def __init__(self, stoi: dict, itos: dict):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def build_from_text(cls, text: str):
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    @classmethod
    def load(cls, meta_path: Path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return cls(meta["stoi"], meta["itos"])

    def save(self, meta_path: Path):
        with open(meta_path, "wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos, "vocab_size": self.vocab_size}, f)

    def encode(self, text: str):
        space_id = self.stoi.get(" ", 0)
        return [self.stoi.get(ch, space_id) for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)
