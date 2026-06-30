"""
EduPro Learner Intelligence Platform
File: src/utils.py

Shared constants and small helper functions used across
data_pipeline.py, features.py, clustering.py, recommender.py
"""

import numpy as np

# ─────────────────────────────────────────────
# CATEGORICAL CONSTANTS
# ─────────────────────────────────────────────

CATEGORIES = [
    'Technology', 'Data Science', 'Business',
    'Design', 'Marketing', 'Finance',
    'Health & Wellness', 'Arts & Creativity'
]

LEVELS       = ['Beginner', 'Intermediate', 'Advanced']
COURSE_TYPES = ['Video', 'Live', 'Hybrid', 'Self-Paced']
GENDERS      = ['Male', 'Female', 'Non-Binary']


# ─────────────────────────────────────────────
# SEGMENT METADATA
# ─────────────────────────────────────────────

SEGMENT_NAMES = {
    0: '🔭 Tech Explorer',
    1: '🚀 Career Climber',
    2: '📚 Deep Specialist',
    3: '🌿 Casual Browser'
}

SEGMENT_COLORS = {
    0: '#4361ee',
    1: '#f72585',
    2: '#7209b7',
    3: '#4cc9f0'
}

SEGMENT_DESC = {
    0: 'Diverse beginner-to-intermediate learners, heavy in Technology & Data Science, high course volume.',
    1: 'Career-driven, certificate-seekers across Business, Finance & Tech. High spenders.',
    2: 'Deep focus in 1–2 domains at Advanced level. Highest spend-per-course.',
    3: 'Low-engagement browsers with minimal spending and diverse but shallow interest.'
}


# ─────────────────────────────────────────────
# ML CONSTANTS
# ─────────────────────────────────────────────

CLUSTERING_FEATURES = [
    'total_courses', 'avg_spending', 'avg_course_rating',
    'diversity_score', 'learning_depth_index', 'beginner_ratio',
    'enrollment_frequency', 'category_concentration', 'n_categories',
    'recency_days', 'preferred_category_enc', 'preferred_level_enc'
]

N_USERS     = 2000
N_COURSES   = 400
N_CLUSTERS  = 4
RANDOM_SEED = 42

# Recommendation scoring weights — must sum to 1.0
SCORE_WEIGHTS = {
    'popularity': 0.30,
    'category':   0.25,
    'level':      0.20,
    'rating':     0.25,
}


# ─────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────

def depth_index_to_level(depth_idx: float) -> str:
    """
    Convert a learning_depth_index (0–1) into a target
    course level. Used by the recommender to decide what
    level to push a user toward next.
    """
    if depth_idx < 0.33:
        return 'Beginner'
    elif depth_idx < 0.67:
        return 'Intermediate'
    else:
        return 'Advanced'


def adjacent_level(level: str) -> str:
    """
    Returns the 'next' level for partial-credit scoring
    in the recommender (e.g. Beginner -> Intermediate).
    """
    mapping = {
        'Beginner':     'Intermediate',
        'Intermediate': 'Advanced',
        'Advanced':     'Intermediate'
    }
    return mapping.get(level, '')


def safe_divide(numerator, denominator, fill=0.0):
    """
    Divide two numpy arrays / pandas Series safely,
    replacing divide-by-zero with `fill` instead of inf/NaN.
    """
    denom = np.where(denominator == 0, 1, denominator)
    result = numerator / denom
    return np.where(denominator == 0, fill, result)