# -*- coding: utf-8 -*-
"""
Sakina backend API (v12).

  POST /api/chat              -- main endpoint, exact schema in the brief
  POST /api/predict_emotion
  POST /api/generate_response
  POST /api/reset_session
  GET  /api/reciters           -- see backend/quran_api.py: written to spec,
  GET  /api/audio                  UNTESTED against the real Quran Foundation
                                    API (no network access to it in this
                                    sandbox). Both degrade to a clear JSON
                                    error rather than crashing if credentials
                                    or network access aren't available.
  GET  /api/health

  Legacy v10/v11 paths (/predict_emotion, /generate_response, /chat,
  /reset_session, /health) kept as aliases, not removed.

Run: python3 app.py   (serves on http://localhost:5000)
"""
import sys
import uuid
import random
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "model"))

from response_composer import sakina_response, classify_emotion, select_verse  # noqa: E402
from knowledge import COMFORT_BANK, REFLECTION_BANK  # noqa: E402
from inference import generate_with_fallback  # noqa: E402
import memory  # noqa: E402
import quran_api  # noqa: E402

app = Flask(__name__)
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"


@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def serve_frontend_files(path):
    file_path = FRONTEND_DIR / path

    if file_path.exists() and file_path.is_file():
        return send_from_directory(FRONTEND_DIR, path)

    return jsonify({
        "success": False,
        "error": "Not found"
    }), 404
CORS(app)


# ---------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"success": False, "error": "message is required"}), 400

    session_id = data.get("session_id") or str(uuid.uuid4())
    result = sakina_response(message, session_id)

    if result["status"] == "safety":
        return jsonify({
            "success": True,
            "session_id": session_id,
            "needs_clarification": False,
            "is_safety_response": True,
            "response": result["response"],
        })

    if result["status"] == "clarification_needed":
        return jsonify({
            "success": True,
            "session_id": session_id,
            "needs_clarification": True,
            "question": result["question"],
        })

    from response_composer import QURAN_DB
    entry = QURAN_DB[result["verse_id"]]
    return jsonify({
        "success": True,
        "session_id": session_id,
        "needs_clarification": False,
        "emotion": result["emotion"],
        "confidence": result["confidence"],
        "response": result["comfort_message"],
        "verse": {
            "id": result["verse_id"],
            "text": result["verse"],
            "surah_name": entry["surah_name"],
            "ayah_number": entry["ayah_number"],
            "is_excerpt": result["is_excerpt"],
        },
        "reflection": result["reflection"],
        "follow_up_question": result["follow_up_question"],
        # extra debug fields, not part of the minimum spec, safe to ignore
        "_debug": {
            "context_carried": result["context_carried"],
            "lexical_anchor": result["lexical_anchor"],
            "used_fallback": result["used_fallback"],
        },
    })


@app.route("/api/predict_emotion", methods=["POST"])
def api_predict_emotion():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"success": False, "error": "message is required"}), 400
    session_id = data.get("session_id")
    emotion, confidence, context_carried = classify_emotion(message, session_id)
    return jsonify({"success": True, "emotion": emotion, "confidence": round(confidence, 3)})


@app.route("/api/generate_response", methods=["POST"])
def api_generate_response():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"success": False, "error": "message is required"}), 400
    emotion = data.get("emotion")
    if not emotion:
        emotion, _, _ = classify_emotion(message)
    verse_id, verse_text, verse_display, is_excerpt = select_verse(emotion)
    fallback_line = random.choice(COMFORT_BANK[emotion])
    comfort, used_fallback, _ = generate_with_fallback(message, emotion, fallback_line)
    reflection = REFLECTION_BANK[emotion]
    return jsonify({
        "success": True,
        "response": comfort,
        "verse": verse_text,
        "verse_reference": verse_display,
        "reflection": reflection,
        "emotion": emotion,
    })


@app.route("/api/reset_session", methods=["POST"])
def api_reset_session():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id")
    if session_id:
        memory.reset_session(session_id)
    return jsonify({"success": True})


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"success": True, "status": "ok", "service": "sakina-backend-v13"})


# ---------------------------------------------------------------------
# Quran Foundation audio -- see backend/quran_api.py docstring: written to
# spec, never executed against the real API in this environment.
# ---------------------------------------------------------------------
@app.route("/api/reciters", methods=["GET"])
def api_reciters():
    try:
        reciters = quran_api.list_reciters()
        return jsonify({"success": True, "reciters": reciters})
    except quran_api.QuranFoundationError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:  # network/API errors -- degrade, don't crash
        return jsonify({"success": False, "error": f"Quran Foundation API unreachable: {e}"}), 503


@app.route("/api/audio", methods=["GET"])
def api_audio():
    verse_id = request.args.get("verse_id")
    recitation_id = request.args.get("recitation_id")
    if not verse_id or not recitation_id:
        return jsonify({"success": False, "error": "verse_id and recitation_id are required"}), 400
    try:
        recitation_id_int = int(recitation_id)
        audio_url = quran_api.get_verse_audio_url(verse_id, recitation_id_int)
        return jsonify({
            "success": True,
            "verse_id": verse_id,
            "recitation_id": recitation_id_int,
            "audio_url": audio_url,
        })
    except quran_api.QuranFoundationError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"Quran Foundation API unreachable: {e}"}), 503


# ---------------------------------------------------------------------
# Legacy aliases (v10/v11 paths), kept working, not removed.
# ---------------------------------------------------------------------
@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    return api_predict_emotion()


@app.route("/generate_response", methods=["POST"])
def generate_response():
    return api_generate_response()


@app.route("/chat", methods=["POST"])
def chat():
    return api_chat()


@app.route("/reset_session", methods=["POST"])
def reset_session():
    return api_reset_session()


@app.route("/health", methods=["GET"])
def health():
    return api_health()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
