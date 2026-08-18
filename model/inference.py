# -*- coding: utf-8 -*-
"""
Loads the trained comfort-generation model and exposes generate_comfort().

Includes the quality gate (adapted from the original project's
sakina_pipeline.py -- that logic was sound, it's kept and extended to
11 emotions) because a tiny from-scratch char model WILL sometimes produce
garbage, and shipping that unfiltered would be worse than a good fallback.
"""
import re
import pickle
import sys
from difflib import SequenceMatcher
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nano_gpt import GPT, GPTConfig
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
CKPT_FILE = ROOT / "checkpoints" / "sakina_v12.pt"
META_FILE = ROOT / "checkpoints" / "meta.pkl"

sys.path.insert(0, str(ROOT / "data"))
from knowledge import COMFORT_BANK  # noqa: E402

# Vocabulary whitelist built from every word that appears in the curated
# COMFORT_BANK (all reviewed, real Arabic) plus common function words that
# appear in the training corpus but not necessarily in COMFORT_BANK itself
# (pronouns, particles). Used to catch word-internal corruption ("تمتنح",
# "أينت") that doesn't repeat characters/chunks and isn't a standalone
# letter, so the earlier checks miss it -- a real gap found in v12 testing.
_FUNCTION_WORDS = {
    "من", "في", "على", "الى", "إلى", "أن", "ان", "أنت", "انت", "أنتِ", "انتِ",
    "هو", "هي", "هذا", "هذه", "ذلك", "التي", "الذي", "لا", "لم", "لن", "ما",
    "و", "أو", "او", "ثم", "لكن", "بل", "قد", "كل", "بعض", "كان", "يكون",
    "لك", "لكِ", "بك", "بكِ", "معك", "معكِ", "له", "لها", "به", "بها",
    "هذي", "كما", "حتى", "إن", "ان", "إذا", "اذا", "عن", "مع", "بين",
}
_VOCAB = set(_FUNCTION_WORDS)
for _lines in COMFORT_BANK.values():
    for _line in _lines:
        _VOCAB.update(re.sub(r"[^\w\s\u0600-\u06FF]", "", _line).split())

DEVICE = "cpu"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    _tokenizer = CharTokenizer.load(META_FILE)
    checkpoint = torch.load(CKPT_FILE, map_location=DEVICE)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(DEVICE)
    _model = model


@torch.no_grad()
def generate_comfort(user_text: str, emotion: str, temperature: float = 0.7, top_k: int = 12):
    """Returns (text, reached_natural_stop: bool). reached_natural_stop is
    False if generation hit the max_new_tokens cap without producing
    <END>/<USER>/<EMOTION> -- a strong signal the sentence was cut off
    mid-thought rather than actually finished."""
    _load()
    prompt = "<USER>\n" + user_text.strip() + "\n\n<EMOTION>\n" + emotion.strip() + "\n\n<COMFORT>\n"
    ids = _tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    generated = _model.generate(x, max_new_tokens=110, temperature=temperature, top_k=top_k)
    full_text = _tokenizer.decode(generated[0].tolist())
    continuation = full_text[len(prompt):]

    reached_stop = False
    for stop_token in ["<END>", "<USER>", "<EMOTION>"]:
        if stop_token in continuation:
            continuation = continuation.split(stop_token, 1)[0]
            reached_stop = True
            break

    return continuation.strip(), reached_stop


# ---------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------
def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"[^\w\s\u0600-\u06FF]", "", text).strip()


def _similarity(a: str, b: str) -> float:
    a, b = _normalize(a), _normalize(b)
    return SequenceMatcher(None, a, b).ratio()


def _word_overlap(a: str, b: str) -> float:
    wa, wb = set(_normalize(a).split()), set(_normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


_SENTENCE_END = re.compile(r"[.!؟]\s*$")
_STANDALONE_LETTER = re.compile(r"(?:^|\s)([\u0621-\u063A\u0641-\u064A])(?:\s|$|[.،,؟!])")


def is_bad_generation(text: str, emotion: str, reached_stop: bool = True) -> bool:
    if not text:
        return True
    text = text.strip()

    if len(text) < 8 or len(text) > 260:
        return True
    if re.search(r"(.)\1{2,}", text):          # aaaa / يييي / "الللله"
        return True
    if re.search(r"(.{2,4})\1{2,}", text):     # repeated char-chunk
        return True
    if re.search(r"\b(\S+)\s+\1\b", text):     # repeated whole word ("كل كل" / "أنا أنا أشعر أشعر")
        return True
    if any(tok in text for tok in ("<USER>", "<EMOTION>", "<COMFORT>", "<END>", "<", ">")):
        return True
    if len(re.findall(r"[\u0600-\u06FF]", text)) < 8:
        return True
    # Standalone single Arabic letters ("ب", "ل" on their own, not attached
    # to a word) are almost never valid tokens in written Arabic -- "و" is
    # the one common exception (stands alone as "and"). This is exactly
    # the class of defect that slipped through in v11 ("ب بابداية" instead
    # of "باب البداية") without tripping the repeated-word/chunk checks.
    for m in _STANDALONE_LETTER.finditer(text):
        if m.group(1) != "و":
            return True
    # A generation that hit the token cap without reaching <END> was very
    # likely cut off mid-sentence -- unless it happens to still end on
    # clean punctuation, treat it as incomplete.
    if not reached_stop and not _SENTENCE_END.search(text):
        return True

    # Vocabulary check: a real Arabic sentence should be built almost
    # entirely from real words. Word-internal corruption ("تمتنح", "أينت")
    # doesn't trip the repeat/standalone-letter checks above, so this
    # catches it directly -- allow some slack (2 unknown tokens or 20%,
    # whichever is larger) since names, rarer words, and the small
    # function-word list aren't exhaustive.
    clean = re.sub(r"[،؛؟۔ۖۗۘۙۚۛ]", " ", text)                    # Arabic punctuation isn't in \w, but IS inside
    clean = re.sub(r"[^\w\s]", " ", clean)                          # the \u0600-\u06FF block, so strip it explicitly first
    tokens = clean.split()
    if tokens:
        unknown = [t for t in tokens if t not in _VOCAB]
        if len(unknown) >= max(2, len(tokens) * 0.2):
            return True

    approved = COMFORT_BANK.get(emotion, [])
    if not approved:
        return True

    best_sim = max((_similarity(text, a) for a in approved), default=0.0)
    best_overlap = max((_word_overlap(text, a) for a in approved), default=0.0)

    # Accept text that's a plausible variation of a reviewed line; reject
    # anything too far from the reviewed register (this is what keeps
    # theologically/emotionally risky improvisation out of the product).
    if not (best_sim >= 0.55 or best_overlap >= 0.35):
        return True

    # Emotion-consistency check: reject if the generation actually reads
    # closer to a DIFFERENT emotion's curated bank than the target
    # emotion's -- catches cross-contamination like a فرح وشكر line
    # surfacing for a خوف input (an issue seen directly in v11 testing).
    for other_emotion, other_bank in COMFORT_BANK.items():
        if other_emotion == emotion:
            continue
        other_best = max((_similarity(text, a) for a in other_bank), default=0.0)
        if other_best > best_sim + 0.05:   # small margin so near-ties don't false-positive
            return True

    return False


def generate_with_fallback(user_text: str, emotion: str, fallback: str):
    """Returns (text, used_fallback: bool, raw_generation: str)."""
    raw, reached_stop = generate_comfort(user_text, emotion)
    if is_bad_generation(raw, emotion, reached_stop):
        return fallback, True, raw
    return raw, False, raw
