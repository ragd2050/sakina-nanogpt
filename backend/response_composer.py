# -*- coding: utf-8 -*-

"""
The Sakina response pipeline.

    user text
        --> safety check
        --> pending clarification resolution
        --> direct lexical anchor
        --> curated clarification if genuinely ambiguous
        --> emotion classification (+ conversation context)
        --> confidence-based clarification if needed
        --> verse selection (rotated, session-aware)
        --> nanoGPT acknowledgment (quality-gated, curated fallback)
        --> reflection (curated)
        --> natural continuation question (curated, rotated)
        --> composed response
"""

import json
import pickle
import random
import sys

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT / "data"),
)

sys.path.insert(
    0,
    str(ROOT / "model"),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from knowledge import (  # noqa: E402
    EMOTION_VERSES,
    COMFORT_BANK,
    REFLECTION_BANK,
    CONTINUATION_BANK,
    CLARIFYING_TRIGGERS,
    EMOTION_TAGS,
    LEXICAL_ANCHORS,
    SAFETY_TRIGGERS,
    SAFETY_RESPONSE,
    CLARIFICATION_ANSWER_MAP,
)

from normalize import normalize_arabic  # noqa: E402

from inference import generate_with_fallback  # noqa: E402

import memory  # noqa: E402


# ============================================================
# FILES
# ============================================================

CLASSIFIER_FILE = (
    ROOT
    / "checkpoints"
    / "sakina_emotion_classifier.pkl"
)

QURAN_FILE = (
    ROOT
    / "data"
    / "quran_verses.json"
)


# ============================================================
# CLASSIFIER CONFIDENCE SETTINGS
# ============================================================

LOW_CONFIDENCE_THRESHOLD = 0.40

LOW_MARGIN_THRESHOLD = 0.10

SHORT_INPUT_CHARS = 18


# Context carry:

CONTEXT_CARRY_CONFIDENCE_THRESHOLD = 0.65

CONTEXT_CARRY_MAX_CHARS = 20


# ============================================================
# LOAD QURAN DATABASE
# ============================================================

with open(
    QURAN_FILE,
    encoding="utf-8",
) as f:

    QURAN_DB = json.load(f)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

with open(
    CLASSIFIER_FILE,
    "rb",
) as f:

    _CLASSIFIER = pickle.load(f)


# ============================================================
# 0. SAFETY CHECK
# ============================================================

def check_safety(text: str):
    """
    Safety is checked before the normal emotional pipeline.

    Returns True if one of the curated high-risk triggers
    appears in the normalized user message.
    """

    normalized = normalize_arabic(text)


    for trigger in SAFETY_TRIGGERS:

        normalized_trigger = normalize_arabic(
            trigger
        )

        if normalized_trigger in normalized:
            return True


    return False


# ============================================================
# 1. LEXICAL ANCHORS
# ============================================================

def find_lexical_anchor(text: str):
    """
    Detect very clear emotion words or short phrases inside
    the user's message.

    Lexical anchors are intentionally narrow and deterministic.

    Examples:

        "معصبة"
            -> غضب

        "أنا مره معصبة"
            -> غضب

        "اليوم أنا سعيدة"
            -> فرح وشكر

        "أحس إني وحيدة"
            -> وحدة

        "انظلمت وأخذوا حقي"
            -> شعور بالظلم


    IMPORTANT:

    The previous implementation mainly matched when the anchor
    was the entire input or appeared at the beginning.

    That meant:

        "معصبة"

    could work, while:

        "أنا مره معصبة"

    could incorrectly fall through to the SVM.

    The new implementation matches complete normalized words
    anywhere inside short user messages while avoiding accidental
    substring matches.
    """

    if not text:
        return None


    normalized_text = normalize_arabic(
        text.strip()
    )


    if not normalized_text:
        return None


    # Lexical anchors are designed for clear short/medium
    # expressions, not long multi-emotion stories.
    #
    # A slightly larger limit than before allows natural phrases
    # such as:
    #
    # "أنا مره معصبة"
    # "اليوم أنا سعيدة"
    # "أحس إني وحيدة"
    #
    if len(normalized_text) > 35:
        return None


    # Padding lets us match complete single words:
    #
    # " معصبة "
    #
    # rather than accidentally matching part of another word.
    padded_text = (
        f" {normalized_text} "
    )


    for emotion, words in LEXICAL_ANCHORS.items():

        for word in words:

            normalized_word = normalize_arabic(
                word.strip()
            )


            if not normalized_word:
                continue


            # ------------------------------------------------
            # MULTI-WORD ANCHOR
            # ------------------------------------------------
            #
            # Examples:
            #
            # "حقي ضاع"
            # "ما أنصفوني"
            #
            if " " in normalized_word:

                if normalized_word in normalized_text:

                    return emotion


            # ------------------------------------------------
            # SINGLE-WORD ANCHOR
            # ------------------------------------------------
            #
            # Examples:
            #
            # "معصبة"
            # "سعيدة"
            # "وحيدة"
            #
            else:

                padded_word = (
                    f" {normalized_word} "
                )


                if padded_word in padded_text:

                    return emotion


    return None


# ============================================================
# 2. CURATED AMBIGUITY CHECK
# ============================================================

def find_curated_clarifying_trigger(
    text: str,
):
    """
    Return the matched trigger key.

    We return the trigger itself rather than only the question
    because session memory needs to remember which clarification
    is pending for the next user turn.
    """

    stripped = text.strip()


    if len(stripped) > 12:
        return None


    for trigger in CLARIFYING_TRIGGERS:

        if trigger in stripped:
            return trigger


    return None


# ============================================================
# 3. RESOLVE PENDING CLARIFICATION
# ============================================================

def resolve_pending_clarification(
    trigger: str,
    reply_text: str,
):
    """
    Resolve the user's answer to a previous clarification question.

    Example:

        User:
            تعبانة

        Sakina:
            هل تقصدين تعبًا جسديًا أم نفسيًا؟

        User:
            نفسي

        Result:
            ضغط نفسي


    Also handles generic classifier clarification markers:

        GENERIC:خوف|قلق وتوتر
    """

    stripped = normalize_arabic(
        reply_text.strip()
    )


    # --------------------------------------------------------
    # GENERIC CLARIFICATION
    # --------------------------------------------------------

    if trigger.startswith(
        "GENERIC:"
    ):

        candidates = trigger[
            len("GENERIC:"):
        ]


        try:

            e1, e2 = candidates.split(
                "|",
                1,
            )

        except ValueError:

            return None


        if (
            normalize_arabic(e1)
            in stripped
        ):

            return e1


        if (
            normalize_arabic(e2)
            in stripped
        ):

            return e2


        if stripped in (
            "الاول",
            "الأول",
            "اول",
            "أول",
        ):

            return e1


        if stripped in (
            "الثاني",
            "ثاني",
        ):

            return e2


        return None


    # --------------------------------------------------------
    # CURATED CLARIFICATION
    # --------------------------------------------------------

    mappings = (
        CLARIFICATION_ANSWER_MAP
        .get(
            trigger,
            [],
        )
    )


    for keywords, emotion in mappings:

        for keyword in keywords:

            normalized_keyword = (
                normalize_arabic(
                    keyword
                )
            )


            if normalized_keyword in stripped:

                return emotion


    return None


# ============================================================
# 4. CONFIDENCE-BASED CLARIFICATION
# ============================================================

def needs_clarification(
    text: str,
    probs,
    classes,
):
    """
    Ask for clarification only for relatively short inputs.

    Long messages generally contain enough contextual information
    to provide a best-guess supportive response rather than
    repeatedly interrupting the conversation.
    """

    if (
        len(text.strip())
        > SHORT_INPUT_CHARS
    ):

        return False


    order = probs.argsort()[::-1]


    top1 = probs[
        order[0]
    ]

    top2 = probs[
        order[1]
    ]


    margin = (
        top1
        - top2
    )


    return (
        top1
        < LOW_CONFIDENCE_THRESHOLD

        or

        margin
        < LOW_MARGIN_THRESHOLD
    )


# ============================================================
# 5. GENERIC CLARIFYING QUESTION
# ============================================================

def generic_clarifying_question(
    classes,
    probs,
):
    """
    Build a clarification between the classifier's top two
    candidate emotions.

    Returns:

        question_text,
        pending_marker
    """

    order = probs.argsort()[::-1]


    e1 = classes[
        order[0]
    ]

    e2 = classes[
        order[1]
    ]


    question = (
        "حابة أفهمك أكثر، "
        f"هل تقصد إنك تشعر بـ{e1} "
        f"أم بـ{e2}؟"
    )


    marker = (
        f"GENERIC:{e1}|{e2}"
    )


    return (
        question,
        marker,
    )


# ============================================================
# 6. EMOTION CLASSIFICATION
# ============================================================

def classify_emotion(
    text: str,
    session_id: str = None,
):
    """
    Classify user emotion.

    Lexical anchors receive deterministic confidence = 1.0.

    Otherwise:
        TF-IDF + calibrated SVM is used.

    A light session-context mechanism can preserve the previous
    emotion for short uncertain follow-up messages.
    """

    # --------------------------------------------------------
    # DIRECT LEXICAL ANCHOR
    # --------------------------------------------------------

    anchor = find_lexical_anchor(
        text
    )


    if anchor:

        return (
            anchor,
            1.0,
            False,
        )


    # --------------------------------------------------------
    # SVM CLASSIFICATION
    # --------------------------------------------------------

    probs = (
        _CLASSIFIER
        .predict_proba(
            [text]
        )[0]
    )


    classes = (
        _CLASSIFIER
        .classes_
    )


    best_idx = (
        probs.argmax()
    )


    emotion = classes[
        best_idx
    ]


    confidence = float(
        probs[
            best_idx
        ]
    )


    # --------------------------------------------------------
    # LIGHT CONTEXT CARRY
    # --------------------------------------------------------

    if session_id:

        session = (
            memory
            .get_session(
                session_id
            )
        )


        previous_emotion = (
            session
            .last_emotion()
        )


        if (
            previous_emotion

            and

            confidence
            < CONTEXT_CARRY_CONFIDENCE_THRESHOLD

            and

            len(
                text.strip()
            )
            <= CONTEXT_CARRY_MAX_CHARS
        ):

            return (
                previous_emotion,
                confidence,
                True,
            )


    return (
        emotion,
        confidence,
        False,
    )


# ============================================================
# 7. VERSE SELECTION
# ============================================================

def select_verse(
    emotion: str,
    session_id: str = None,
):
    """
    Select a Quran verse from the curated pool.

    Avoid repeating the immediately previous verse when possible.

    Quran text is ALWAYS retrieved from quran_verses.json.
    nanoGPT is never used to generate Quran text.
    """

    pool = EMOTION_VERSES[
        emotion
    ]


    last_verse = None


    if session_id:

        last_verse = (
            memory
            .get_session(
                session_id
            )
            .last_verse_id
        )


    candidates = [
        verse_id
        for verse_id in pool
        if verse_id != last_verse
    ]


    if not candidates:

        candidates = pool


    verse_id = random.choice(
        candidates
    )


    verse = QURAN_DB[
        verse_id
    ]


    display = (
        f'سورة {verse["surah_name"]}، '
        f'الآية {verse["ayah_number"]}'
    )


    return (
        verse_id,
        verse["text"],
        display,
        verse.get(
            "is_excerpt",
            False,
        ),
    )


# ============================================================
# 8. CONTINUATION QUESTION
# ============================================================

def select_continuation(
    emotion: str,
    session_id: str = None,
):
    """
    Select a natural follow-up question.

    Avoid repeating questions within the same session when possible.
    """

    pool = CONTINUATION_BANK[
        emotion
    ]


    if not session_id:

        return random.choice(
            pool
        )


    session = (
        memory
        .get_session(
            session_id
        )
    )


    fresh = [
        question

        for question in pool

        if question
        not in session.used_continuations
    ]


    if fresh:

        choice = random.choice(
            fresh
        )

    else:

        choice = random.choice(
            pool
        )


    session.used_continuations.add(
        choice
    )


    return choice


# ============================================================
# 9. MAIN SAKINA PIPELINE
# ============================================================

def sakina_response(
    user_text: str,
    session_id: str = "default",
):
    """
    Main Sakina response pipeline.

    Order:

        1. Safety
        2. Pending clarification resolution
        3. Lexical anchor
        4. Curated ambiguity clarification
        5. Generic classifier uncertainty clarification
        6. Emotion classification/context
        7. Full response composition
    """


    # ========================================================
    # 1. SAFETY FIRST
    # ========================================================

    if check_safety(
        user_text
    ):

        memory.record_turn(
            session_id,
            user_text,
            "safety",
            None,
            None,
        )


        return {

            "type":
                "safety_response",

            "status":
                "safety",

            "user_input":
                user_text,

            "response":
                SAFETY_RESPONSE,

            "emotion":
                None,

            "verse":
                None,

            "verse_reference":
                None,

            "reflection":
                None,
        }


    # ========================================================
    # 2. RESOLVE PREVIOUS CLARIFICATION
    # ========================================================

    pending_trigger = None


    if session_id:

        pending_trigger = (
            memory
            .pop_pending_clarification(
                session_id
            )
        )


    if pending_trigger:

        resolved_emotion = (
            resolve_pending_clarification(
                pending_trigger,
                user_text,
            )
        )


        if resolved_emotion:

            return _compose_full_response(

                user_text,

                resolved_emotion,

                confidence=1.0,

                context_carried=True,

                anchor_emotion=True,

                session_id=session_id,
            )


        # If the response did not match a known answer,
        # continue through the normal pipeline.


    # ========================================================
    # 3. DIRECT LEXICAL ANCHOR
    # ========================================================
    #
    # IMPORTANT:
    #
    # This now happens before curated ambiguity and before
    # classifier uncertainty.
    #
    # Example:
    #
    #   أنا مره معصبة
    #
    # must become:
    #
    #   غضب
    #
    # rather than:
    #
    #   حيرة وتشتت
    #   or a generic clarification.
    #
    # ========================================================

    anchor_emotion = (
        find_lexical_anchor(
            user_text
        )
    )


    if anchor_emotion:

        return _compose_full_response(

            user_text,

            anchor_emotion,

            confidence=1.0,

            context_carried=False,

            anchor_emotion=True,

            session_id=session_id,
        )


    # ========================================================
    # 4. CURATED AMBIGUOUS INPUTS
    # ========================================================

    curated_trigger = (
        find_curated_clarifying_trigger(
            user_text
        )
    )


    if curated_trigger:

        if session_id:

            memory.set_pending_clarification(
                session_id,
                curated_trigger,
            )


        question = (
            CLARIFYING_TRIGGERS[
                curated_trigger
            ]
        )


        return {

            "type":
                "clarifying_question",

            "status":
                "clarification_needed",

            "user_input":
                user_text,

            "question":
                question,

            "response":
                question,

            "emotion":
                None,

            "verse":
                None,

            "verse_reference":
                None,

            "reflection":
                None,
        }


    # ========================================================
    # 5. GENERIC CLASSIFIER UNCERTAINTY
    # ========================================================

    probs = (
        _CLASSIFIER
        .predict_proba(
            [user_text]
        )[0]
    )


    classes = (
        _CLASSIFIER
        .classes_
    )


    if needs_clarification(
        user_text,
        probs,
        classes,
    ):

        (
            question,
            generic_marker,
        ) = generic_clarifying_question(
            classes,
            probs,
        )


        if session_id:

            memory.set_pending_clarification(
                session_id,
                generic_marker,
            )


        return {

            "type":
                "clarifying_question",

            "status":
                "clarification_needed",

            "user_input":
                user_text,

            "question":
                question,

            "response":
                question,

            "emotion":
                None,

            "verse":
                None,

            "verse_reference":
                None,

            "reflection":
                None,
        }


    # ========================================================
    # 6. NORMAL CLASSIFICATION
    # ========================================================

    (
        emotion,
        confidence,
        context_carried,
    ) = classify_emotion(
        user_text,
        session_id,
    )


    # ========================================================
    # 7. FULL RESPONSE
    # ========================================================

    return _compose_full_response(

        user_text,

        emotion,

        confidence=confidence,

        context_carried=context_carried,

        anchor_emotion=False,

        session_id=session_id,
    )


# ============================================================
# 10. FULL RESPONSE COMPOSER
# ============================================================

def _compose_full_response(
    user_text,
    emotion,
    confidence,
    context_carried,
    anchor_emotion,
    session_id,
):
    """
    Compose a complete grounded Sakina response.
    """

    tag = EMOTION_TAGS[
        emotion
    ]


    # --------------------------------------------------------
    # Quran grounding
    # --------------------------------------------------------

    (
        verse_id,
        verse_text,
        verse_display,
        is_excerpt,
    ) = select_verse(
        emotion,
        session_id,
    )


    # --------------------------------------------------------
    # Curated fallback
    # --------------------------------------------------------

    fallback_line = random.choice(
        COMFORT_BANK[
            emotion
        ]
    )


    # --------------------------------------------------------
    # nanoGPT generation + quality gate
    # --------------------------------------------------------

    (
        comfort,
        used_fallback,
        raw_generation,
    ) = generate_with_fallback(
        user_text,
        emotion,
        fallback_line,
    )


    # --------------------------------------------------------
    # Curated reflection
    # --------------------------------------------------------

    reflection = (
        REFLECTION_BANK[
            emotion
        ]
    )


    # --------------------------------------------------------
    # Follow-up
    # --------------------------------------------------------

    continuation = (
        select_continuation(
            emotion,
            session_id,
        )
    )


    # --------------------------------------------------------
    # Conversation bridge
    # --------------------------------------------------------

    bridge = ""


    if session_id:

        session = (
            memory
            .get_session(
                session_id
            )
        )


        if context_carried:

            bridge = (
                "يبدو أن هذا امتداد لما ذكرته "
                f"قبل قليل عن {emotion}. "
            )


        elif session.emotion_changed(
            emotion
        ):

            bridge = (
                "ألاحظ أن شعورك تغيّر قليلاً "
                "منذ حديثنا قبل قليل، وهذا طبيعي. "
            )


    # --------------------------------------------------------
    # Excerpt label
    # --------------------------------------------------------

    excerpt_label = (
        "\n(مقتطف من الآية)"
        if is_excerpt
        else ""
    )


    # --------------------------------------------------------
    # Response depth
    # --------------------------------------------------------

    is_short_input = (
        len(
            user_text.strip()
        )
        <= 20
    )


    if is_short_input:

        full_text = (

            f"{bridge}"
            f"{comfort}\n\n"

            f"﴿ {verse_text} ﴾\n"

            f"{verse_display}"
            f"{excerpt_label}\n\n"

            f"{continuation}"
        )


    else:

        full_text = (

            f"{bridge}"
            f"{comfort}\n\n"

            f"﴿ {verse_text} ﴾\n"

            f"{verse_display}"
            f"{excerpt_label}\n\n"

            f"{reflection}\n\n"

            f"{continuation}"
        )


    # --------------------------------------------------------
    # Session memory
    # --------------------------------------------------------

    memory.record_turn(
        session_id,
        user_text,
        emotion,
        verse_id,
        comfort,
    )


    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return {

        "type":
            "full_response",

        "status":
            "ok",

        "user_input":
            user_text,

        "emotion":
            emotion,

        "emotion_tag":
            tag,

        "confidence":
            round(
                confidence,
                3,
            ),

        "context_carried":
            context_carried,

        "lexical_anchor":
            bool(
                anchor_emotion
            ),

        "verse_id":
            verse_id,

        "verse":
            verse_text,

        "verse_reference":
            verse_display,

        "is_excerpt":
            is_excerpt,

        "comfort_message":
            comfort,

        "used_fallback":
            used_fallback,

        "raw_generation":
            raw_generation,

        "reflection":
            reflection,

        "continuation":
            continuation,

        "follow_up_question":
            continuation,

        "response":
            full_text,
    }