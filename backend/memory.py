# -*- coding: utf-8 -*-
"""
Lightweight in-process conversation memory, keyed by session id.

Deliberately simple (no DB) since this is a prototype-scale deliverable --
swap _SESSIONS for Redis/a DB table in production without touching the
composer's interface.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

MAX_TURNS = 6


@dataclass
class Turn:
    user_text: str
    emotion: str


@dataclass
class SessionState:
    turns: deque = field(default_factory=lambda: deque(maxlen=MAX_TURNS))
    last_verse_id: Optional[str] = None
    last_comfort: Optional[str] = None
    used_continuations: set = field(default_factory=set)
    # v13: the clarifying-question trigger word we're waiting on an answer
    # for, e.g. "تعبانة" (from CLARIFYING_TRIGGERS). None when no question
    # is outstanding. See response_composer.py's CLARIFICATION_ANSWER_MAP --
    # this is what lets a short reply like "نفسي" be understood as an
    # answer to "تعباً جسدياً أم نفسياً؟" instead of being reclassified from
    # zero as an unrelated new message (the exact failure mode named in the
    # v13 brief section 26).
    pending_clarification: Optional[str] = None

    def last_emotion(self) -> Optional[str]:
        return self.turns[-1].emotion if self.turns else None

    def emotion_changed(self, new_emotion: str) -> bool:
        prev = self.last_emotion()
        return prev is not None and prev != new_emotion


_SESSIONS: dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionState()
    return _SESSIONS[session_id]


def record_turn(session_id: str, user_text: str, emotion: str, verse_id: str, comfort: str):
    s = get_session(session_id)
    s.turns.append(Turn(user_text=user_text, emotion=emotion))
    s.last_verse_id = verse_id
    s.last_comfort = comfort


def set_pending_clarification(session_id: str, trigger: Optional[str]):
    get_session(session_id).pending_clarification = trigger


def pop_pending_clarification(session_id: str) -> Optional[str]:
    """Read and clear in one step, so a resolved (or unresolved-but-consumed)
    clarification never lingers into a third, unrelated turn."""
    s = get_session(session_id)
    trigger = s.pending_clarification
    s.pending_clarification = None
    return trigger


def reset_session(session_id: str):
    _SESSIONS.pop(session_id, None)
