# -*- coding: utf-8 -*-
"""
Arabic normalization used before both classifier training/inference and
nanoGPT tokenization-adjacent steps (kept separate from the char-level
tokenizer itself, which must stay lossless -- normalization is applied to
the classifier's view of the text only, not to what the generator sees,
so the generator still learns to reproduce real spelling).

v12 change: taa marbuta (ة) is NO LONGER normalized to ه globally, per
explicit review request -- it was applied in v11 without being evaluated
in isolation, which risks conflating "genuinely helped" with "happened to
help alongside other changes." Character n-grams already give a lot of
tolerance for the ة/ه spelling swap on their own (they share all but the
last character), so dropping the aggressive global normalization costs
little robustness while removing an untested assumption.

Handles:
  - alef forms (أ إ آ ٱ) -> ا
  - alef maqsura (ى) -> ي   (optional-but-on: extremely common, low risk --
    ى almost never contrasts meaningfully with ي in informal chat text)
  - tatweel/kashida (ـ) removed
  - diacritics (tashkeel) removed
  - repeated-letter elongation ("طويييل") collapsed to max 2 repeats
  - punctuation normalized (multiple ! or ؟ collapsed, etc.)
  - whitespace collapsed
"""
import re

_ALEF = re.compile(r"[إأآٱ]")
_ALEF_MAQSURA = re.compile(r"ى")
_TATWEEL = re.compile(r"ـ+")
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")
_ELONGATION = re.compile(r"(.)\1{2,}")
_PUNCT_RUN = re.compile(r"([!؟?.,،])\1+")
_WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str, normalize_taa_marbuta: bool = False) -> str:
    """normalize_taa_marbuta kept as an explicit opt-in flag (default OFF,
    see module docstring) rather than removed outright, so it can still be
    A/B evaluated deliberately if someone wants to test it in isolation."""
    if not text:
        return text
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _ALEF.sub("ا", text)
    text = _ALEF_MAQSURA.sub("ي", text)
    if normalize_taa_marbuta:
        text = text.replace("ة", "ه")
    text = _ELONGATION.sub(r"\1\1", text)
    text = _PUNCT_RUN.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


class ArabicNormalizer:
    """Sklearn-compatible transformer, importable from a stable path
    (data.normalize.ArabicNormalizer) so pickled classifier pipelines
    unpickle correctly regardless of which script loads them."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize_arabic(x) for x in X]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
