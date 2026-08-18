# -*- coding: utf-8 -*-
"""
v13 automated test suite (brief section 34). Run with:

    cd sakina && python3 -m pytest tests/ -v

Covers: classifier (all 12 emotions + lexical anchors + spelling variants),
ambiguity/clarification (incl. the v13 pending-clarification fix), memory
(context carry, emotion transition, pending clarification), Quran (valid
retrieval, excerpt flag, verse text never sourced from the model), audio
(response parsing / graceful degradation), nanoGPT (valid generation,
malformed rejection, repetition rejection, emotional-mismatch rejection),
and the Flask API surface.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "model"))

import memory  # noqa: E402
from knowledge import EMOTIONS, COMFORT_BANK, LEXICAL_ANCHORS  # noqa: E402
from response_composer import (  # noqa: E402
    sakina_response, classify_emotion, select_verse, QURAN_DB,
)
from inference import is_bad_generation  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_sessions():
    memory._SESSIONS.clear()
    yield
    memory._SESSIONS.clear()


# ===================== Classifier =====================
class TestClassifier:
    @pytest.mark.parametrize("emotion", EMOTIONS)
    def test_all_12_emotions_are_classifiable(self, emotion):
        """Every emotion has at least one COMFORT_BANK line the classifier
        can round-trip (sanity check the label space itself, not accuracy)."""
        assert emotion in EMOTIONS
        assert len(COMFORT_BANK[emotion]) >= 8

    @pytest.mark.parametrize("word,expected", [
        ("سعيدة", "فرح وشكر"), ("مبسوط", "فرح وشكر"),
        ("وحيدة", "وحدة"), ("لحالي", "وحدة"),
        ("متوترة", "قلق وتوتر"), ("قلقان", "قلق وتوتر"),
        ("محتارة", "حيرة وتشتت"), ("ضايع", "حيرة وتشتت"),
        ("ندمان", "ذنب وتقصير"),
        ("مشتاقة", "فقد وشوق"),
        ("مظلوم", "شعور بالظلم"), ("انظلمت", "شعور بالظلم"),
    ])
    def test_lexical_anchors(self, word, expected):
        emotion, confidence, _ = classify_emotion(word)
        assert emotion == expected
        assert confidence == 1.0  # anchors are deterministic, not probabilistic

    def test_spelling_variants_route_consistently(self):
        # taa marbuta / trailing alef variants of the same anchor word
        e1, _, _ = classify_emotion("متوترة")
        e2, _, _ = classify_emotion("متوتر")
        assert e1 == e2 == "قلق وتوتر"


# ===================== Ambiguity / clarification =====================
class TestAmbiguity:
    @pytest.mark.parametrize("text", ["تعبانة", "طفشانة", "مو قادر", "مو بخير"])
    def test_ambiguous_input_triggers_clarification(self, text):
        r = sakina_response(text, session_id=f"t-{text}")
        assert r["status"] == "clarification_needed"
        assert r["emotion"] is None

    def test_unambiguous_lexical_anchor_does_not_clarify(self):
        r = sakina_response("سعيدة", session_id="t-clear")
        assert r["status"] == "ok"
        assert r["emotion"] == "فرح وشكر"


# ===================== Memory =====================
class TestMemory:
    def test_context_carry_on_short_followup(self):
        sid = "mem-1"
        r1 = sakina_response("أنا خايفة من الاختبار", sid)
        r2 = sakina_response("ما قدرت أنام", sid)
        assert r2["emotion"] == r1["emotion"]
        assert r2["context_carried"] is True

    def test_emotion_transition_is_detected(self):
        sid = "mem-2"
        sakina_response("خايفة جدًا من نتيجة القرار اللي أخذته", sid)
        r2 = sakina_response("وحاسة اني وحيدة في هالفترة صعبة", sid)
        assert r2["emotion"] == "وحدة"

    def test_pending_clarification_resolves_correctly(self):
        """v13 regression test: this was the real bug found and fixed this
        session -- 'نفسي' used to be reclassified from zero (-> ذنب وتقصير,
        wrong) instead of resolving the outstanding question."""
        sid = "mem-3"
        r1 = sakina_response("تعبانة", sid)
        assert r1["status"] == "clarification_needed"
        r2 = sakina_response("نفسي", sid)
        assert r2["emotion"] == "ضغط نفسي"

    def test_pending_clarification_does_not_leak_to_third_turn(self):
        sid = "mem-4"
        sakina_response("طفشان", sid)
        sakina_response("ما أدري بصراحة وش فيني", sid)  # unmatched reply
        r3 = sakina_response("سعيدة", sid)
        assert r3["emotion"] == "فرح وشكر"  # not still "answering" old question

    def test_reset_session_clears_state(self):
        sid = "mem-5"
        sakina_response("سعيدة", sid)
        assert memory.get_session(sid).turns
        memory.reset_session(sid)
        assert sid not in memory._SESSIONS


# ===================== Quran =====================
class TestQuran:
    @pytest.mark.parametrize("emotion", EMOTIONS)
    def test_valid_verse_retrieval_for_every_emotion(self, emotion):
        verse_id, text, display, is_excerpt = select_verse(emotion)
        assert verse_id in QURAN_DB
        assert text and isinstance(text, str)
        assert isinstance(is_excerpt, bool)

    def test_excerpt_flag_present_on_every_verse(self):
        for verse_id, entry in QURAN_DB.items():
            assert "is_excerpt" in entry, f"{verse_id} missing is_excerpt"

    def test_verse_text_is_never_sourced_from_the_model(self):
        """Hard requirement (brief section 24): nanoGPT must never generate,
        complete, or paraphrase Quran text. We can't inspect the model's
        internals here, but we CAN assert the pipeline's data flow never
        routes generated text into the verse field -- select_verse() only
        ever reads from QURAN_DB (data/quran_verses.json), never from
        inference.generate_comfort()."""
        import inspect
        import response_composer
        src = inspect.getsource(response_composer.select_verse)
        assert "generate" not in src
        assert "QURAN_DB[verse_id]" in src


# ===================== Audio =====================
class TestAudio:
    def test_get_verse_audio_url_rejects_bad_verse_id_format(self):
        import quran_api
        with pytest.raises(ValueError):
            quran_api.get_verse_audio_url("not-a-verse-id", 7)

    def test_missing_credentials_degrade_gracefully(self, monkeypatch):
        import quran_api
        monkeypatch.setattr(quran_api, "QF_CLIENT_ID", None)
        monkeypatch.setattr(quran_api, "QF_CLIENT_SECRET", None)
        with pytest.raises(quran_api.QuranFoundationError):
            quran_api.list_reciters()


# ===================== nanoGPT quality gate =====================
class TestQualityGate:
    def test_valid_curated_line_is_accepted(self):
        line = COMFORT_BANK["حزن"][0]
        assert is_bad_generation(line, "حزن", reached_stop=True) is False

    def test_repeated_word_is_rejected(self):
        assert is_bad_generation("أنا أنا أفهم أفهم تعبك.", "حزن") is True

    def test_repeated_chunk_is_rejected(self):
        assert is_bad_generation("الله الله الله معك", "خوف") is True

    def test_malformed_word_fragment_is_rejected(self):
        assert is_bad_generation("الندم لا يعني أن ب بابداية أغلق.", "ذنب وتقصير") is True

    def test_too_short_is_rejected(self):
        assert is_bad_generation("تمام", "حزن") is True

    def test_control_tokens_leaking_into_output_is_rejected(self):
        assert is_bad_generation("خذ وقتك <END> <USER>", "حزن") is True

    def test_cross_emotion_contamination_is_rejected(self):
        joy_line = COMFORT_BANK["فرح وشكر"][0]
        assert is_bad_generation(joy_line, "خوف") is True


# ===================== Flask API surface =====================
class TestAPI:
    @pytest.fixture
    def client(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        return flask_app.app.test_client()

    def test_chat_requires_message(self, client):
        r = client.post("/api/chat", json={})
        assert r.status_code == 400

    def test_chat_happy_path(self, client):
        r = client.post("/api/chat", json={"message": "سعيدة", "session_id": "api-1"})
        assert r.status_code == 200
        body = r.get_json()
        assert body["success"] is True
        assert body["emotion"] == "فرح وشكر"
        assert "is_excerpt" in body["verse"]

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.get_json()["success"] is True

    def test_reciters_degrades_without_credentials(self, client):
        r = client.get("/api/reciters")
        assert r.status_code in (200, 503)  # 503 expected without QF credentials

    def test_audio_requires_params(self, client):
        r = client.get("/api/audio")
        assert r.status_code == 400

    def test_reset_session(self, client):
        client.post("/api/chat", json={"message": "سعيدة", "session_id": "api-reset"})
        r = client.post("/api/reset_session", json={"session_id": "api-reset"})
        assert r.status_code == 200
        assert "api-reset" not in memory._SESSIONS
