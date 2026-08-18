# -*- coding: utf-8 -*-
"""
Sakina nanoGPT inference module.

Loads the trained comfort-generation model once and exposes:
    generate_comfort()
    generate_with_fallback()

Deployment notes:
- The local version can use a longer generation length.
- On Render/free CPU environments, generation is intentionally shorter
  to reduce latency and memory/CPU pressure.
- Any failed or low-quality generation falls back to a reviewed
  COMFORT_BANK response instead of exposing unsafe/garbled output.
"""

import os
import re
import pickle
import sys
from difflib import SequenceMatcher
from pathlib import Path

import torch


# ============================================================
# Project imports
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parent

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(ROOT / "data"))

from nano_gpt import GPT, GPTConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from knowledge import COMFORT_BANK  # noqa: E402


# ============================================================
# Model files
# ============================================================

CKPT_FILE = ROOT / "checkpoints" / "sakina_v12.pt"
META_FILE = ROOT / "checkpoints" / "meta.pkl"

DEVICE = "cpu"


# ============================================================
# Deployment configuration
# ============================================================

# Render sets environment variables in production.
IS_RENDER = bool(os.environ.get("RENDER"))

# Local development keeps the original generation length.
# Render uses a shorter sequence because the free instance
# has limited CPU/RAM and long autoregressive generation can timeout.
DEFAULT_MAX_NEW_TOKENS = 48 if IS_RENDER else 110

MAX_NEW_TOKENS = int(
    os.environ.get(
        "SAKINA_MAX_NEW_TOKENS",
        DEFAULT_MAX_NEW_TOKENS
    )
)


# Keep PyTorch CPU usage predictable on small cloud instances.
try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


# ============================================================
# Vocabulary whitelist
# ============================================================

_FUNCTION_WORDS = {
    "من", "في", "على", "الى", "إلى", "أن", "ان", "أنت", "انت",
    "أنتِ", "انتِ", "هو", "هي", "هذا", "هذه", "ذلك", "التي",
    "الذي", "لا", "لم", "لن", "ما", "و", "أو", "او", "ثم",
    "لكن", "بل", "قد", "كل", "بعض", "كان", "يكون", "لك",
    "لكِ", "بك", "بكِ", "معك", "معكِ", "له", "لها", "به",
    "بها", "هذي", "كما", "حتى", "إن", "إذا", "اذا", "عن",
    "مع", "بين",
}

_VOCAB = set(_FUNCTION_WORDS)

for _lines in COMFORT_BANK.values():
    for _line in _lines:
        cleaned_line = re.sub(
            r"[^\w\s\u0600-\u06FF]",
            "",
            _line
        )
        _VOCAB.update(cleaned_line.split())


# ============================================================
# Lazy-loaded model
# ============================================================

_tokenizer = None
_model = None


def _load():
    """
    Load tokenizer and nanoGPT model once per worker.

    The model stays in memory after the first request so subsequent
    requests do not repeatedly reload the checkpoint.
    """

    global _tokenizer, _model

    if _model is not None:
        return

    if not CKPT_FILE.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CKPT_FILE}"
        )

    if not META_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer metadata not found: {META_FILE}"
        )

    _tokenizer = CharTokenizer.load(META_FILE)

    checkpoint = torch.load(
        CKPT_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    config = GPTConfig(
        **checkpoint["model_args"]
    )

    model = GPT(config)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()
    model.to(DEVICE)

    # Gradients are never needed during web inference.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    _model = model

    print(
        f"[Sakina] nanoGPT loaded on {DEVICE}. "
        f"max_new_tokens={MAX_NEW_TOKENS}, "
        f"render={IS_RENDER}"
    )


# ============================================================
# nanoGPT generation
# ============================================================

@torch.inference_mode()
def generate_comfort(
    user_text: str,
    emotion: str,
    temperature: float = 0.7,
    top_k: int = 12,
):
    """
    Generate a comfort sentence using Sakina nanoGPT.

    Returns:
        (generated_text, reached_natural_stop)

    reached_natural_stop=False means generation reached the token
    limit before emitting <END>, <USER>, or <EMOTION>.
    """

    _load()

    user_text = str(user_text or "").strip()
    emotion = str(emotion or "").strip()

    prompt = (
        "<USER>\n"
        + user_text
        + "\n\n<EMOTION>\n"
        + emotion
        + "\n\n<COMFORT>\n"
    )

    ids = _tokenizer.encode(prompt)

    x = torch.tensor(
        ids,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    generated = _model.generate(
        x,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temperature,
        top_k=top_k,
    )

    generated_ids = generated[0].tolist()

    full_text = _tokenizer.decode(
        generated_ids
    )

    continuation = full_text[len(prompt):]

    reached_stop = False

    for stop_token in [
        "<END>",
        "<USER>",
        "<EMOTION>",
    ]:
        if stop_token in continuation:
            continuation = continuation.split(
                stop_token,
                1
            )[0]

            reached_stop = True
            break

    continuation = continuation.strip()

    # Release temporary references as soon as possible.
    del x
    del generated

    return continuation, reached_stop


# ============================================================
# Quality Gate
# ============================================================

def _normalize(text: str) -> str:
    text = re.sub(
        r"\s+",
        " ",
        text.strip()
    )

    return re.sub(
        r"[^\w\s\u0600-\u06FF]",
        "",
        text
    ).strip()


def _similarity(a: str, b: str) -> float:
    a = _normalize(a)
    b = _normalize(b)

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def _word_overlap(a: str, b: str) -> float:
    wa = set(
        _normalize(a).split()
    )

    wb = set(
        _normalize(b).split()
    )

    if not wa or not wb:
        return 0.0

    return len(wa & wb) / len(wa | wb)


_SENTENCE_END = re.compile(
    r"[.!؟]\s*$"
)

_STANDALONE_LETTER = re.compile(
    r"(?:^|\s)"
    r"([\u0621-\u063A\u0641-\u064A])"
    r"(?:\s|$|[.،,؟!])"
)


def is_bad_generation(
    text: str,
    emotion: str,
    reached_stop: bool = True
) -> bool:
    """
    Return True when a nanoGPT generation should not be shown
    directly to the user.
    """

    if not text:
        return True

    text = text.strip()


    # --------------------------------------------------------
    # Basic length validation
    # --------------------------------------------------------

    if len(text) < 8:
        return True

    if len(text) > 260:
        return True


    # --------------------------------------------------------
    # Repetition / corruption checks
    # --------------------------------------------------------

    if re.search(
        r"(.)\1{2,}",
        text
    ):
        return True

    if re.search(
        r"(.{2,4})\1{2,}",
        text
    ):
        return True

    if re.search(
        r"\b(\S+)\s+\1\b",
        text
    ):
        return True


    # --------------------------------------------------------
    # Special-token leakage
    # --------------------------------------------------------

    if any(
        token in text
        for token in (
            "<USER>",
            "<EMOTION>",
            "<COMFORT>",
            "<END>",
            "<",
            ">",
        )
    ):
        return True


    # --------------------------------------------------------
    # Arabic-content check
    # --------------------------------------------------------

    arabic_chars = re.findall(
        r"[\u0600-\u06FF]",
        text
    )

    if len(arabic_chars) < 8:
        return True


    # --------------------------------------------------------
    # Standalone Arabic letter corruption
    # --------------------------------------------------------

    for match in _STANDALONE_LETTER.finditer(text):

        if match.group(1) != "و":
            return True


    # --------------------------------------------------------
    # Incomplete generation check
    # --------------------------------------------------------

    if (
        not reached_stop
        and not _SENTENCE_END.search(text)
    ):
        return True


    # --------------------------------------------------------
    # Vocabulary quality check
    # --------------------------------------------------------

    clean = re.sub(
        r"[،؛؟۔ۖۗۘۙۚۛ]",
        " ",
        text
    )

    clean = re.sub(
        r"[^\w\s]",
        " ",
        clean
    )

    tokens = clean.split()

    if tokens:

        unknown = [
            token
            for token in tokens
            if token not in _VOCAB
        ]

        allowed_unknown = max(
            2,
            int(len(tokens) * 0.2)
        )

        if len(unknown) >= allowed_unknown:
            return True


    # --------------------------------------------------------
    # Emotion bank validation
    # --------------------------------------------------------

    approved = COMFORT_BANK.get(
        emotion,
        []
    )

    if not approved:
        return True


    best_sim = max(
        (
            _similarity(
                text,
                approved_line
            )
            for approved_line in approved
        ),
        default=0.0,
    )


    best_overlap = max(
        (
            _word_overlap(
                text,
                approved_line
            )
            for approved_line in approved
        ),
        default=0.0,
    )


    if not (
        best_sim >= 0.55
        or best_overlap >= 0.35
    ):
        return True


    # --------------------------------------------------------
    # Cross-emotion contamination check
    # --------------------------------------------------------

    for (
        other_emotion,
        other_bank
    ) in COMFORT_BANK.items():

        if other_emotion == emotion:
            continue

        other_best = max(
            (
                _similarity(
                    text,
                    approved_line
                )
                for approved_line in other_bank
            ),
            default=0.0,
        )

        if other_best > best_sim + 0.05:
            return True


    return False


# ============================================================
# Safe generation + fallback
# ============================================================

def generate_with_fallback(
    user_text: str,
    emotion: str,
    fallback: str,
):
    """
    Attempt nanoGPT generation.

    Returns:
        text
        used_fallback
        raw_generation

    If nanoGPT raises a normal Python exception, Sakina returns
    the reviewed fallback instead of returning HTTP 500.
    """

    try:

        raw, reached_stop = generate_comfort(
            user_text,
            emotion
        )

    except Exception as exc:

        print(
            "[Sakina] nanoGPT generation failed; "
            f"using fallback. Error: "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            fallback,
            True,
            ""
        )


    if is_bad_generation(
        raw,
        emotion,
        reached_stop
    ):

        return (
            fallback,
            True,
            raw
        )


    return (
        raw,
        False,
        raw
    )