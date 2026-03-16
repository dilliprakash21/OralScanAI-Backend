import os
import io
import time
import json
from uuid import uuid4
from datetime import datetime, timedelta, timezone

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from PIL import Image

import mysql.connector
from mysql.connector import pooling
from mysql.connector import Error as MySQLError

import bcrypt
import jwt

# Torch/torchvision are optional. If they fail to import (common on some Windows setups),
# the server will still run and /analyze will fall back to a deterministic mock.
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
except Exception as _e:
    torch = None
    nn = None
    transforms = None
    models = None
    _TORCH_IMPORT_ERROR = str(_e)
else:
    _TORCH_IMPORT_ERROR = ""

"""
Flask backend (FULL): Auth + MySQL DB + AI /analyze

This file is designed to RUN WITHOUT CRASHING even if:
- the model file is missing (it will fall back to a deterministic mock)
- Anthropic isn't installed or no API key is set (it will return safe fallback text)

ENV (backend/.env recommended)
--------------------------------
# MySQL
DB_HOST=localhost
DB_USER=root
DB_PASS=your_password
DB_NAME=oralscan_db

# JWT
JWT_SECRET=change_me_to_a_long_random_string

# AI (optional)
MODEL_PATH=/path/to/oralscan_classification.pth
ANTHROPIC_API_KEY=your_key   # optional

# Server (optional)
PORT=8000

# SMTP (required for OTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
SMTP_FROM=OralScan AI <your_email@gmail.com>
"""

# Load .env from backend folder if present
load_dotenv()

# =====================================================
# 1) Flask setup
# =====================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global DB error handler
@app.errorhandler(MySQLError)
def handle_mysql_error(e):
    return jsonify({"error": f"Database error: {str(e)}"}), 500

# =====================================================
# 2) Config
# =====================================================
PORT = int(os.getenv("PORT", "8000"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "1234")
DB_NAME = os.getenv("DB_NAME", "oralscan_db")

JWT_SECRET = os.getenv("JWT_SECRET", "dev_only_change_me")
JWT_ALG = "HS256"

MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\vdill\Downloads\oralscan_classification.pth")
DEVICE = (torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch else None)

# Optional Anthropic
try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "") # optional
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if (Anthropic and ANTHROPIC_API_KEY) else None

# SMTP Config
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "oralscanai@gmail.com")
SMTP_PASS = os.getenv("SMTP_PASS", "ovkp orqs lkfo xopn")
SMTP_FROM = os.getenv("SMTP_FROM", f"OralScan AI <{SMTP_USER}>")

# =====================================================
# 3) MySQL pool + helpers
# =====================================================
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="oralscan_pool",
            pool_size=5,
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
        )
    return _pool

def db_query(sql, params=None, *, fetchone=False, fetchall=False, commit=False):
    cnx = get_pool().get_connection()
    try:
        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, params or ())
        if commit:
            cnx.commit()
        if fetchone:
            return cur.fetchone()
        if fetchall:
            return cur.fetchall()
        return None
    finally:
        cur.close()
        cnx.close()

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def send_email(to_email, subject, body):
    if not SMTP_USER or not SMTP_PASS:
        print(f"⚠️ Email not sent to {to_email} (SMTP credentials missing)")
        return False
    
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"✅ OTP Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {str(e)}")
        return False

# =====================================================
# 4) Auth helpers
# =====================================================
def sign_token(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 24 * 7,  # 7 days
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def require_auth():
    h = request.headers.get("Authorization", "")
    if not h.startswith("Bearer "):
        return None
    token = h.split(" ", 1)[1]
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        return None

# =====================================================
# 5) AI model (optional)
# =====================================================
# =====================================================
# 5) AI model (optional)
# =====================================================
if torch and nn and models and transforms:
    class OralScanClassifier(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.backbone = models.resnet18(weights=None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.fc_plaque = nn.Linear(in_features, num_classes)
            self.fc_gingival = nn.Linear(in_features, num_classes)

        def forward(self, x):
            f = self.backbone(x)
            return self.fc_plaque(f), self.fc_gingival(f)

    def load_model(path):
        if not path or not os.path.exists(path):
            return None
        m = OralScanClassifier()
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        print("✅ Model loaded")
        return m

MODEL = load_model(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
LABELS = ["Low","Moderate","High","Severe"]

# =====================================================
# 6) Static uploads
# =====================================================
@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# =====================================================
# 7) Health
# =====================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "time": now_utc_iso()})

# =====================================================
# 8) AUTH ROUTES
# =====================================================
@app.route("/auth/signup", methods=["POST"])
def auth_signup():
    try:
        body = request.get_json(silent=True) or {}
        email = (body.get("email") or "").strip().lower()
        password = body.get("password") or ""
        name = body.get("name", "").strip()
        phone = (body.get("phone") or "").strip()
        role = body.get("role", "").strip()
        license_no = (body.get("license_no") or "").strip()
        clinic = (body.get("clinic") or "").strip()
        location = (body.get("location") or "").strip()

        if not email or not password or not name or not role:
            return jsonify({"error": "Email, password, name, and role required"}), 400

        # 1. Check if user already exists in main table (email AND phone)
        exists_email = db_query("SELECT id FROM users WHERE email=%s LIMIT 1", (email,), fetchone=True)
        if exists_email:
            return jsonify({"error": "Email already registered. Please login."}), 409
            
        if phone:
            exists_phone = db_query("SELECT id FROM users WHERE phone=%s LIMIT 1", (phone,), fetchone=True)
            if exists_phone:
                return jsonify({"error": "Phone number already registered. Please use another."}), 409

        # 2. Generate OTP
        import random
        otp_code = f"{random.randint(100000, 999999)}"
        
        # 3. Store in pending_users
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        payload = {
            "name": name,
            "password_hash": pw_hash,
            "phone": phone,
            "role": role,
            "license_no": license_no,
            "clinic": clinic,
            "location": location
        }
        
        db_query(
            "REPLACE INTO pending_users (email, payload, otp_code, expires_at) VALUES (%s, %s, %s, DATE_ADD(NOW(), INTERVAL 15 MINUTE))",
            (email, json.dumps(payload), otp_code),
            commit=True
        )

        # 4. Send Email
        sent = send_email(
            email,
            "Verify your OralScan AI account",
            f"Your 6-digit verification code is: {otp_code}\n\nIt expires in 15 minutes."
        )
        
        if not sent:
            return jsonify({"error": "Failed to send OTP email. Check server config."}), 500

        return jsonify({"ok": True, "message": "OTP sent to email."})
    except Exception as e:
        print(f"🔥 SIGNUP CRASH: {str(e)}")
        return jsonify({"error": "Internal server error during signup"}), 500




@app.route("/auth/verify-signup", methods=["POST"])
def auth_verify_signup():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    code = (body.get("code") or "").strip()
    
    if not email or not code:
        return jsonify({"error": "Email and code required"}), 400

    # 1. Find in pending
    pending = db_query("SELECT * FROM pending_users WHERE email=%s AND expires_at > NOW() LIMIT 1", (email,), fetchone=True)
    if not pending:
        return jsonify({"error": "OTP expired or invalid. Please sign up again."}), 400
        
    if pending["otp_code"] != code:
        return jsonify({"error": "Invalid verification code"}), 400

    # 2. Extract payload and move to main tables
    data = json.loads(pending["payload"])
    user_id = str(uuid4())
    
    # Insert User
    db_query(
        "INSERT INTO users (id, email, password_hash, phone, full_name, role, email_verified, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())",
        (user_id, email, data["password_hash"], data["phone"], data["name"], data["role"], 1),
        commit=True
    )
    
    # Insert Profile
    db_query(
        "INSERT INTO profiles (id, user_id, name, email, phone, role, license_no, clinic, location, created_at, updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())",
        (str(uuid4()), user_id, data["name"], email, data["phone"], data["role"], data["license_no"], data["clinic"], data["location"]),
        commit=True
    )
    
    # 3. Clean up
    db_query("DELETE FROM pending_users WHERE email=%s", (email,), commit=True)
    
    # 4. Return initial tokens
    token = sign_token(user_id, email)
    return jsonify({"ok": True, "token": token, "user": {"id": user_id, "email": email}})

@app.route("/auth/login", methods=["POST"])
def auth_login():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    user = db_query("SELECT id, email, password_hash FROM users WHERE email=%s LIMIT 1", (email,), fetchone=True)
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    ok = bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8"))
    if not ok:
        return jsonify({"error": "Invalid credentials"}), 401

    token = sign_token(user["id"], user["email"])
    
    refresh_token = str(uuid4())
    db_query("INSERT INTO refresh_tokens (id, user_id, token, expires_at) VALUES (%s,%s,%s,DATE_ADD(NOW(), INTERVAL 7 DAY))",
             (str(uuid4()), user["id"], refresh_token), commit=True)
             
    return jsonify({"token": token, "refresh_token": refresh_token, "user": {"id": user["id"], "email": user["email"]}})

@app.route("/auth/me", methods=["GET"])
def auth_me():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = payload["user_id"]
    profile = db_query("SELECT * FROM profiles WHERE user_id=%s LIMIT 1", (user_id,), fetchone=True)
    return jsonify({"user": {"id": user_id, "email": payload.get("email")}, "profile": profile})

@app.route("/auth/verify-password", methods=["POST"])
def auth_verify_password():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    password = body.get("password") or ""
    if not password:
        return jsonify({"error": "Password required"}), 400

    user_id = payload["user_id"]
    user = db_query("SELECT password_hash FROM users WHERE id=%s LIMIT 1", (user_id,), fetchone=True)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    ok = bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8"))
    return jsonify({"ok": bool(ok)})

@app.route("/auth/verify", methods=["POST"])
def auth_verify_otp():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    code = (body.get("code") or "").strip()
    if not email or not code:
        return jsonify({"error": "Email and code required"}), 400
    
    otp_row = db_query("SELECT code FROM otp_codes WHERE email=%s AND expires_at > NOW() ORDER BY created_at DESC LIMIT 1", (email,), fetchone=True)
    if not otp_row:
        return jsonify({"error": "Invalid or expired OTP"}), 400
        
    if otp_row["code"] != code:
        return jsonify({"error": "Invalid OTP"}), 400
        
    db_query("UPDATE users SET email_verified=1 WHERE email=%s", (email,), commit=True)
    return jsonify({"ok": True})

@app.route("/auth/forgot-password", methods=["POST"])
def auth_forgot_password():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Email required"}), 400

    # 1. Check if user exists
    user = db_query("SELECT id FROM users WHERE email=%s LIMIT 1", (email,), fetchone=True)
    if not user:
        return jsonify({"error": "No account found with this email. Please register first."}), 404

    # 2. Generate OTP
    import random
    otp_code = f"{random.randint(100000, 999999)}"
    db_query(
        "REPLACE INTO otp_codes (id, email, code, type, expires_at) VALUES (%s,%s,%s,'forgot_password',DATE_ADD(NOW(), INTERVAL 15 MINUTE))",
        (str(uuid4()), email, otp_code),
        commit=True
    )

    # 3. Send Email
    sent = send_email(
        email,
        "OralScan AI - Reset Password",
        f"Your password reset code is: {otp_code}\n\nIt expires in 15 minutes."
    )
    
    if not sent:
        return jsonify({"error": "Failed to send reset email"}), 500

    return jsonify({"ok": True, "message": "Reset code sent."})

@app.route("/auth/reset-password", methods=["POST"])
def auth_reset_password():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    code = (body.get("code") or "").strip()
    new_password = body.get("password", "")
    
    if not email or not code or not new_password:
        return jsonify({"error": "Email, code, and new password required"}), 400

    # 1. Verify OTP
    otp_row = db_query(
        "SELECT code FROM otp_codes WHERE email=%s AND type='forgot_password' AND expires_at > NOW() ORDER BY created_at DESC LIMIT 1",
        (email,),
        fetchone=True
    )
    if not otp_row or otp_row["code"] != code:
        return jsonify({"error": "Invalid or expired reset code"}), 400

    # 2. Update Password
    pw_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    db_query("UPDATE users SET password_hash=%s WHERE email=%s", (pw_hash, email), commit=True)
    
    # 3. Clean up
    db_query("DELETE FROM otp_codes WHERE email=%s AND type='forgot_password'", (email,), commit=True)

    return jsonify({"ok": True, "message": "Password updated successfully."})

@app.route("/auth/change-password", methods=["POST"])
def auth_change_password():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    
    body = request.get_json(silent=True) or {}
    current_password = body.get("current_password")
    new_password = body.get("new_password")
    
    if not current_password or not new_password:
        return jsonify({"error": "Current and new password required"}), 400
        
    user_id = payload["user_id"]
    user = db_query("SELECT password_hash FROM users WHERE id=%s LIMIT 1", (user_id,), fetchone=True)
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    # Verify current password
    if not bcrypt.checkpw(current_password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return jsonify({"error": "Incorrect current password"}), 400
        
    # Update to new password
    pw_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    db_query("UPDATE users SET password_hash=%s WHERE id=%s", (pw_hash, user_id), commit=True)
    
    return jsonify({"ok": True, "message": "Password changed successfully"})

@app.route("/auth/refresh", methods=["POST"])
def auth_refresh():
    body = request.get_json(silent=True) or {}
    r_token = body.get("refresh_token")
    if not r_token:
        return jsonify({"error": "Missing refresh_token"}), 400
        
    rt = db_query("SELECT user_id FROM refresh_tokens WHERE token=%s AND expires_at > NOW() LIMIT 1", (r_token,), fetchone=True)
    if not rt:
        return jsonify({"error": "Invalid or expired refresh token"}), 401
    
    user_id = rt["user_id"]
    user = db_query("SELECT email FROM users WHERE id=%s LIMIT 1", (user_id,), fetchone=True)
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    new_token = sign_token(user_id, user["email"])
    return jsonify({"token": new_token})

# =====================================================
# 9) PROFILE ROUTES
# =====================================================
@app.route("/profiles/me", methods=["PUT"])
def profile_update():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = payload["user_id"]

    body = request.get_json(silent=True) or {}
    allowed = ["name","phone","role","license_no","clinic","location","hospital","avatar_url"]
    sets = []
    params = []
    for k in allowed:
        if k in body:
            sets.append(f"{k}=%s")
            params.append(body.get(k))
    if not sets:
        return jsonify({"error": "No fields to update"}), 400

    params.append(user_id)
    db_query(f"UPDATE profiles SET {', '.join(sets)}, updated_at=NOW() WHERE user_id=%s", tuple(params), commit=True)
    profile = db_query("SELECT * FROM profiles WHERE user_id=%s LIMIT 1", (user_id,), fetchone=True)
    return jsonify({"profile": profile})

@app.route("/profiles/me/avatar", methods=["POST"])
def profile_upload_avatar():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = payload["user_id"]

    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400
    f = request.files["file"]
    
    # Check file size (5MB limit)
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > 5 * 1024 * 1024:
        return jsonify({"error": "File size exceeds 5MB limit"}), 413
        
    if not f.filename:
        return jsonify({"error": "Invalid file"}), 400

    ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
    filename = f"avatar_{user_id}_{uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)

    base_url = request.host_url.rstrip("/")
    avatar_url = f"{base_url}/uploads/{filename}"

    db_query("UPDATE profiles SET avatar_url=%s, updated_at=NOW() WHERE user_id=%s", (avatar_url, user_id), commit=True)
    return jsonify({"avatar_url": avatar_url})

# =====================================================
# 10) SCREENINGS ROUTES
# =====================================================
@app.route("/screenings", methods=["POST"])
def screenings_create():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = payload["user_id"]

    b = request.get_json(silent=True) or {}
    if not b.get("patient_id"):
        return jsonify({"error": "patient_id required"}), 400

    sid = str(uuid4())

    db_query(
        """INSERT INTO screenings (
          id,user_id,patient_id,patient_name,phone,age,dob,gender,location,mode,consent_given,
          image_url,heatmap_url,risk_level,ai_confidence,plaque_index,gingival_index,overridden,
          override_reason,referral_clinic,notes,created_at
        ) VALUES (
          %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
          %s,%s,%s,%s,%s,%s,%s,
          %s,%s,%s,NOW()
        )""",
        (
            sid, user_id,
            b.get("patient_id"),
            b.get("patient_name"),
            b.get("phone"),
            b.get("age"),
            b.get("dob"),
            b.get("gender"),
            b.get("location"),
            b.get("mode") or "screening",
            1 if b.get("consent_given") else 0,
            b.get("image_url"),
            b.get("heatmap_url"),
            (b.get("risk_level") or "").lower() if b.get("risk_level") else None,
            b.get("ai_confidence"),
            b.get("plaque_index"),
            b.get("gingival_index"),
            1 if b.get("overridden") else 0,
            b.get("override_reason"),
            b.get("referral_clinic"),
            b.get("notes"),
        ),
        commit=True
    )

    row = db_query("SELECT * FROM screenings WHERE id=%s AND user_id=%s", (sid, user_id), fetchone=True)
    return jsonify({"screening": row})

@app.route("/screenings", methods=["GET"])
def screenings_list():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = payload["user_id"]

    limit = int(request.args.get("limit", "50"))
    q = (request.args.get("q") or "").strip()
    date_from = (request.args.get("from") or "").strip()
    date_to = (request.args.get("to") or "").strip()

    is_admin = payload.get("role") == "admin"
    where = []
    params = []

    if not is_admin:
        where.append("user_id=%s")
        params.append(user_id)

    if q:
        where.append("(patient_name LIKE %s OR patient_id LIKE %s OR phone LIKE %s)")
        like = f"%{q}%"
        params.extend([like, like, like])

    if date_from:
        where.append("created_at >= %s")
        params.append(date_from)

    if date_to:
        where.append("created_at <= %s")
        params.append(date_to)

    sql = f"SELECT * FROM screenings WHERE {' AND '.join(where)} ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    rows = db_query(sql, tuple(params), fetchall=True)
    return jsonify({"screenings": rows})

@app.route("/screenings/<sid>", methods=["GET"])
def screenings_get(sid):
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = payload["user_id"]

    row = db_query("SELECT * FROM screenings WHERE id=%s AND user_id=%s", (sid, user_id), fetchone=True)
    if not row:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"screening": row})

@app.route("/screenings/stats", methods=["GET"])
def screenings_stats():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = payload["user_id"] # Keep user_id for non-admin queries
    is_admin = payload.get("role") == "admin"
    role_filter = "" if is_admin else "AND user_id=%s"
    stats_params = [] if is_admin else [user_id]

    today_start = datetime.now().strftime("%Y-%m-%d 00:00:00")
    today = db_query(
        f"SELECT COUNT(*) AS c FROM screenings WHERE created_at >= %s {role_filter}",
        (today_start, *stats_params),
        fetchone=True
    )["c"]

    since = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    week = db_query(
        f"SELECT COUNT(*) AS c FROM screenings WHERE created_at >= %s {role_filter}",
        (since, *stats_params),
        fetchone=True
    )["c"]

    # For total, if not admin, the WHERE clause needs to start with 'WHERE user_id=%s'
    # If admin, there is no WHERE clause.
    total_where_clause = ""
    if not is_admin:
        total_where_clause = "WHERE user_id=%s"
    
    total = db_query(
        f"SELECT COUNT(*) AS c FROM screenings {total_where_clause}",
        tuple(stats_params),
        fetchone=True
    )["c"]

    low = db_query(
        f"SELECT COUNT(*) AS c FROM screenings WHERE LOWER(risk_level)='low' {role_filter}",
        tuple(stats_params),
        fetchone=True
    )["c"]

    lowRiskPercent = round((low / total) * 100, 2) if total else 0

    return jsonify({
        "today": int(today),
        "week": int(week),
        "lowRiskPercent": float(lowRiskPercent)
    })

# =====================================================
# 11) NOTIFICATIONS
# =====================================================
@app.route("/notifications", methods=["GET"])
def notifications_list():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = payload["user_id"]

    rows = db_query(
        "SELECT * FROM notifications WHERE user_id=%s ORDER BY created_at DESC LIMIT 50",
        (user_id,),
        fetchall=True
    )
    return jsonify({"notifications": rows})

@app.route("/notifications/mark-all-read", methods=["POST"])
def notifications_mark_all_read():
    payload = require_auth()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = payload["user_id"]

    db_query("UPDATE notifications SET `read`=1 WHERE user_id=%s", (user_id,), commit=True)
    return jsonify({"ok": True})

# =====================================================
# 12) ANALYZE
# =====================================================

def analyze_image_bytes(image_bytes):
    if not MODEL or not transform or not torch:
        return {
          "risk": "low",
          "plaqueScore": 20,
          "gingivalScore": 15,
          "heatmapUrl": None,
          "notes": "Mock result - model not loaded"
        }

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    filename = f"{uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    img.save(path)

    base_url = request.host_url.rstrip("/")
    image_url = f"{base_url}/uploads/{filename}"

    if MODEL and transform and torch:
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            plaque_pred, gingival_pred = MODEL(x)

            plaque_index = torch.argmax(plaque_pred,1).item()
            gingival_index = torch.argmax(gingival_pred,1).item()

            plaque_conf = torch.softmax(plaque_pred,dim=1)[0,plaque_index].item()
            gingival_conf = torch.softmax(gingival_pred,dim=1)[0,gingival_index].item()

        model_conf = float((plaque_conf + gingival_conf)/2)

    else:
        plaque_index = 0
        gingival_index = 0
        model_conf = 0.5

    ai_confidence = model_conf

    if anthropic_client:
        try:
            prompt = f"""
            Plaque index: {plaque_index}
            Gingival index: {gingival_index}
            Give overall confidence 0-1 number only
            """

            msg = anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role":"user","content":prompt}]
            )

            ai_confidence = float(msg.content[0].text.strip())
        except Exception:
            pass

    return {
        "ok": True,
        "image_url": image_url,
        "plaqueIndex": plaque_index,
        "plaque_label": LABELS[plaque_index],
        "gingivalIndex": gingival_index,
        "gingival_label": LABELS[gingival_index],
        "aiConfidence": round(float(ai_confidence) * 100, 2)
    }
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error":"No file"}),400

    image_bytes = request.files["file"].read()
    if not image_bytes:
        return jsonify({"error":"Empty"}),400
        
    if len(image_bytes) > 5 * 1024 * 1024:
        return jsonify({"error": "File size exceeds 5MB limit"}), 413

    return jsonify(analyze_image_bytes(image_bytes))

if __name__ == "__main__":
    app_debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=PORT, debug=True)
