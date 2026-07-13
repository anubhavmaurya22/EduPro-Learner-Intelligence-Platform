"""
EdUPro Learner Intelligence Platform
File: src/feedback.py
Handles collection and analysis of recommendation feedback
"""

import pandas as pd
import os
from datetime import datetime

FEEDBACK_FILE = 'data/outputs/feedback.csv'

def save_feedback(user_id: str,
                  course_id: str,
                  liked: bool) -> None:
    """Save a thumbs up or down for a recommendation."""
    os.makedirs('data/outputs', exist_ok=True)

    new_row = pd.DataFrame([{
        'UserID':    user_id,
        'CourseID':  course_id,
        'Liked':     liked,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])

    if os.path.exists(FEEDBACK_FILE):
        existing = pd.read_csv(FEEDBACK_FILE)
        # Remove old feedback for same user+course
        existing = existing[
            ~((existing['UserID'] == user_id) &
              (existing['CourseID'] == course_id))
        ]
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(FEEDBACK_FILE, index=False)


def load_feedback() -> pd.DataFrame:
    """Load all feedback. Returns empty DataFrame if none exists."""
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=['UserID','CourseID','Liked','Timestamp'])


def get_feedback_stats() -> dict:
    """Returns summary stats for the dashboard."""
    fb = load_feedback()
    if fb.empty:
        return {'total': 0, 'liked': 0, 'disliked': 0, 'rate': 0.0}

    return {
        'total':    len(fb),
        'liked':    fb['Liked'].sum(),
        'disliked': (~fb['Liked']).sum(),
        'rate':     round(fb['Liked'].mean() * 100, 1)
    }


def get_liked_courses(user_id: str) -> list:
    """Get list of CourseIDs this user thumbs-upped."""
    fb = load_feedback()
    if fb.empty:
        return []
    return fb[
        (fb['UserID'] == user_id) & (fb['Liked'] == True)
    ]['CourseID'].tolist()


def get_disliked_courses(user_id: str) -> list:
    """Get list of CourseIDs this user thumbs-downed."""
    fb = load_feedback()
    if fb.empty:
        return []
    return fb[
        (fb['UserID'] == user_id) & (fb['Liked'] == False)
    ]['CourseID'].tolist()


def compute_feedback_weights(base_weights: dict) -> dict:
    """
    Adjusts recommendation weights based on feedback patterns.
    If high-rated courses get more thumbs up → increase rating weight.
    If peer-popular courses get thumbs down → decrease popularity weight.
    Returns adjusted weights that sum to 1.0.
    """
    fb = load_feedback()
    if len(fb) < 10:
        return base_weights   # not enough data yet

    # Join with course ratings if available
    weights = base_weights.copy()

    liked_pct = fb['Liked'].mean()

    # Simple heuristic adjustment
    if liked_pct > 0.75:
        pass    # model is working well, keep weights
    elif liked_pct < 0.50:
        # Users unhappy — boost rating quality weight
        weights['rating']     = min(0.40, weights['rating'] + 0.05)
        weights['popularity'] = max(0.20, weights['popularity'] - 0.05)
        # Renormalize
        total = sum(weights.values())
        weights = {k: round(v/total, 2) for k, v in weights.items()}

    return weights