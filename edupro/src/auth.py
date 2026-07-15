"""
EdUPro Learner Intelligence Platform
File: src/auth.py
Handles user authentication and session management
"""

import hashlib
import json
import os
import pandas as pd
from datetime import datetime

AUTH_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'outputs', 'users_auth.json'
)

LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'outputs', 'login_log.csv'
)

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _hash(password: str) -> str:
    """SHA-256 hash a password."""
    return hashlib.sha256(password.encode()).hexdigest()


# ─────────────────────────────────────────────
# ROLE PERMISSIONS
# ─────────────────────────────────────────────

ROLE_PAGES = {
    "admin": [
        "📊 Platform Overview",
        "👤 Professional Explorer",
        "🔮 Recommendations",
        "🔵 Cluster Map",
        "📈 Segment Insights",
        "🔍 EDA & Analytics",
        "📣 Feedback Analytics",
        "👥 User Management",
    ],
    "viewer": [
        "📊 Platform Overview",
        "👤 Professional Explorer",
        "🔮 Recommendations",
        "🔵 Cluster Map",
        "📈 Segment Insights",
        "🔍 EDA & Analytics",
    ],
    "learner": [
        "👤 Professional Explorer",
        "🔮 Recommendations",
    ]
}


# ─────────────────────────────────────────────
# DEFAULT USERS
# ─────────────────────────────────────────────

def _get_default_users() -> dict:
    return {
        "admin": {
            "password": _hash("admin123"),
            "role":     "admin",
            "name":     "EdUPro Admin",
            "email":    "admin@edupro-online.com",
            "created":  datetime.now().strftime("%Y-%m-%d")
        },
        "marene": {
            "password": _hash("edupro2026"),
            "role":     "admin",
            "name":     "Marene Jooste",
            "email":    "marene@edupro-online.com",
            "created":  datetime.now().strftime("%Y-%m-%d")
        },
        "viewer": {
            "password": _hash("view123"),
            "role":     "viewer",
            "name":     "Guest Viewer",
            "email":    "viewer@edupro-online.com",
            "created":  datetime.now().strftime("%Y-%m-%d")
        },
        "demo": {
            "password": _hash("demo123"),
            "role":     "learner",
            "name":     "Demo Professional",
            "email":    "demo@edupro-online.com",
            "created":  datetime.now().strftime("%Y-%m-%d")
        }
    }


# ─────────────────────────────────────────────
# USER STORAGE
# ─────────────────────────────────────────────

def _load_users() -> dict:
    os.makedirs(os.path.dirname(AUTH_FILE), exist_ok=True)
    if not os.path.exists(AUTH_FILE):
        users = _get_default_users()
        _save_users(users)
        return users
    try:
        with open(AUTH_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        users = _get_default_users()
        _save_users(users)
        return users


def _save_users(users: dict) -> None:
    os.makedirs(os.path.dirname(AUTH_FILE), exist_ok=True)
    with open(AUTH_FILE, 'w') as f:
        json.dump(users, f, indent=2)


# ─────────────────────────────────────────────
# AUTH FUNCTIONS
# ─────────────────────────────────────────────

def authenticate(username: str, password: str):
    """
    Verify credentials.
    Returns user dict on success, None on failure.
    """
    users    = _load_users()
    username = username.strip().lower()
    if username not in users:
        _log_attempt(username, success=False)
        return None
    user = users[username]
    if user['password'] == _hash(password):
        _log_attempt(username, success=True)
        return {
            'username': username,
            'name':     user['name'],
            'role':     user['role'],
            'email':    user['email'],
        }
    _log_attempt(username, success=False)
    return None


def get_allowed_pages(role: str) -> list:
    """Return pages this role can access."""
    return ROLE_PAGES.get(role, ROLE_PAGES['learner'])


def _log_attempt(username: str, success: bool) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        row = pd.DataFrame([{
            'Username':  username,
            'Success':   success,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        if os.path.exists(LOG_FILE):
            existing = pd.read_csv(LOG_FILE)
            row = pd.concat([existing, row], ignore_index=True)
        row.to_csv(LOG_FILE, index=False)
    except Exception:
        pass


def load_login_log() -> pd.DataFrame:
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=['Username', 'Success', 'Timestamp'])


def add_user(username: str, password: str,
             role: str, name: str, email: str) -> bool:
    """Add new user. Returns True if successful."""
    users = _load_users()
    if username.lower() in users:
        return False
    users[username.lower()] = {
        'password': _hash(password),
        'role':     role,
        'name':     name,
        'email':    email,
        'created':  datetime.now().strftime('%Y-%m-%d')
    }
    _save_users(users)
    return True


def change_password(username: str,
                    old_password: str,
                    new_password: str) -> bool:
    """Change password. Returns True if successful."""
    users = _load_users()
    username = username.lower()
    if username not in users:
        return False
    if users[username]['password'] != _hash(old_password):
        return False
    users[username]['password'] = _hash(new_password)
    _save_users(users)
    return True


def get_all_users() -> pd.DataFrame:
    """Return all users as DataFrame without passwords."""
    users = _load_users()
    rows = []
    for uname, info in users.items():
        rows.append({
            'Username': uname,
            'Name':     info.get('name', ''),
            'Email':    info.get('email', ''),
            'Role':     info.get('role', ''),
            'Created':  info.get('created', ''),
        })
    return pd.DataFrame(rows)
