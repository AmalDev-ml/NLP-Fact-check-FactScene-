"""
app.py — Live Web Evidence Verification Flask Server & SSE Streaming
---------------------------------------------------------------------
Phase 3 Features:
Routes:
  GET  /               → Web interface dashboard (index.html)
  POST /analyze        → Standard JSON verification endpoint
  GET  /analyze_stream → Server-Sent Events (SSE) streaming progress, early verdict, and explanation
  GET  /health         → Server & provider readiness check (includes degraded status)
"""

import json
import queue
import logging
import threading
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify, Response, stream_with_context

from config import get_active_providers
from detector import detect, get_embed_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

_model_ready = False
_warmup_event = threading.Event()

# Log active providers at startup
active_providers = get_active_providers()
logger.info(f"FactSense Phase 3 Startup — Active Providers: {active_providers}")


def _warmup():
    global _model_ready
    logger.info("Initializing embedding models in background warmup...")
    try:
        get_embed_model()
        logger.info("Sentence embedding model successfully warmed up.")
    except Exception as e:
        logger.error(f"Warmup notice: {e}")
    finally:
        _model_ready = True
        _warmup_event.set()


threading.Thread(target=_warmup, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    providers = get_active_providers()
    return jsonify({
        "status": "ok",
        "model_ready": _model_ready,
        "providers": providers,
        "degraded": providers.get("degraded", True)
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """Standard JSON endpoint."""
    data = request.get_json(force=True)
    article = data.get("article", "").strip()

    if not article:
        return jsonify({"error": "No article text provided."}), 400

    if not _model_ready:
        if not _warmup_event.wait(timeout=30):
            return jsonify({"error": "System is still initializing. Please retry in a moment."}), 503

    try:
        result = detect(article)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error during detection")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500


@app.route("/analyze_stream", methods=["GET", "POST"])
def analyze_stream():
    """
    Server-Sent Events (SSE) streaming endpoint:
    Streams real-time pipeline stages:
      - extracting_claims
      - searching (live search & page reading)
      - reading_pages
      - judging (LLM / NLI factual judgment & framing)
      - verdict (early emit of verdict & confidence before explanation)
      - explaining (synthesizing grounded factual explanation)
      - complete (with full final payload)
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        article = data.get("article", "").strip()
    else:
        article = request.args.get("article", "").strip()

    if not article:
        return jsonify({"error": "No article text provided."}), 400

    if not _model_ready:
        _warmup_event.wait(timeout=30)

    def event_stream():
        evt_queue = queue.Queue()

        def progress_callback(stage: str, message: str):
            evt_queue.put({"type": "stage", "stage": stage, "message": message})

        def event_callback(evt_payload: Dict[str, Any]):
            evt_queue.put(evt_payload)

        def worker():
            try:
                res = detect(article, progress_callback=progress_callback, event_callback=event_callback)
                evt_queue.put({"type": "complete", "result": res})
            except Exception as e:
                logger.exception("Stream execution failed")
                evt_queue.put({"type": "error", "error": str(e)})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            try:
                msg = evt_queue.get(timeout=35.0)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("complete", "error"):
                    break
            except queue.Empty:
                yield f": heartbeat\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
