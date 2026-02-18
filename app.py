"""
Flask application — serves the login page, chat UI, and API endpoints.
Authentication is handled by ERPNext: users log in with their ERPNext
credentials, and the session cookies are stored in the Flask session.
"""

import uuid
import pickle
import base64
import logging
import requests
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from config import Config
from agent import Agent
from erpnext_client import ERPNextClient

# ── Logging setup ────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s\033[0m %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai-erpnext")

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

agent = Agent()


# ── Helpers ──────────────────────────────────────────────────────────


def login_required(f):
    """Decorator: redirect to login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "erp_cookies" not in session:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "Not authenticated"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def _get_erp_client() -> ERPNextClient:
    """Reconstruct an ERPNextClient from the session-stored cookies."""
    sess = requests.Session()
    cookies = pickle.loads(base64.b64decode(session["erp_cookies"]))
    sess.cookies = cookies
    return ERPNextClient(session=sess)


def _save_erp_cookies(erp: ERPNextClient) -> None:
    """Persist the ERPNext session cookies into the Flask session."""
    session["erp_cookies"] = base64.b64encode(
        pickle.dumps(erp.session.cookies)
    ).decode("ascii")


# ── Auth routes ──────────────────────────────────────────────────────


@app.route("/login")
def login_page():
    if "erp_cookies" in session:
        return redirect(url_for("index"))
    return render_template("login.html", erpnext_url=Config.ERPNEXT_URL)


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400

    try:
        erp = ERPNextClient.login(username, password)
        full_user = erp.get_logged_in_user()

        _save_erp_cookies(erp)
        session["erp_user"] = full_user or username
        session["session_id"] = str(uuid.uuid4())

        logger.info("\033[32m[LOGIN]\033[0m %s authenticated successfully", session["erp_user"])
        return jsonify({"success": True, "user": session["erp_user"]})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 401
    except Exception as e:
        return jsonify({"success": False, "error": f"Connection error: {e}"}), 500


@app.route("/api/logout", methods=["POST"])
def api_logout():
    sid = session.get("session_id", "")
    if sid:
        agent.clear_session(sid)

    # Try to log out from ERPNext too
    try:
        erp = _get_erp_client()
        erp.session.get(f"{Config.ERPNEXT_URL}/api/method/logout")
    except Exception:
        pass

    session.clear()
    return jsonify({"success": True})


# ── Main page ────────────────────────────────────────────────────────


@app.route("/")
@login_required
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html", user=session.get("erp_user", ""))


# ── Chat API ─────────────────────────────────────────────────────────


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))
    session["session_id"] = session_id

    try:
        erp = _get_erp_client()
        reply = agent.chat(session_id, user_message, erp_client=erp)
        # Update cookies in case ERPNext rotated the session
        _save_erp_cookies(erp)
        return jsonify({"reply": reply})
    except Exception as e:
        # If auth expired, force re-login
        if "403" in str(e) or "401" in str(e) or "Forbidden" in str(e):
            session.clear()
            return jsonify({"error": "Session expired. Please log in again."}), 401
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
@login_required
def history():
    session_id = session.get("session_id", "")
    if not session_id:
        return jsonify({"messages": []})
    msgs = agent.get_history(session_id)
    return jsonify(
        {
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in msgs
                if m["role"] in ("user", "assistant")
            ]
        }
    )


@app.route("/api/clear", methods=["POST"])
@login_required
def clear():
    session_id = session.get("session_id", "")
    if session_id:
        agent.clear_session(session_id)
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=True)
