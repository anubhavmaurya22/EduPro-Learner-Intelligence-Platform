
"""
EduPro Learner Intelligence Platform
File: src/utils.py
Context: edupro-online.com — CPD platform for pediatric professionals
"""

import numpy as np

# ─────────────────────────────────────────────
# EDUPRO-SPECIFIC CONSTANTS
# ─────────────────────────────────────────────

CATEGORIES = [
    'DCD & Motor Disorders',
    'Fundamental Movement Skills',
    'Classroom Interventions',
    'Sensory Processing',
    'Sport & Leisure',
    'Assessment & Diagnosis'
]

LEVELS            = ['Beginner', 'Intermediate', 'Advanced']
COURSE_TYPES      = ['Webinar', 'Self-Paced Video', 'Live Workshop', 'Mini Course']
GENDERS           = ['Female', 'Male', 'Non-Binary']

PROFESSION_TYPES = [
    'Occupational Therapist',
    'Physiotherapist',
    'Kinderkineticist',
    'Classroom Teacher',
    'School Psychologist',
    'Parent/Caregiver'
]

# ─────────────────────────────────────────────
# SEGMENT METADATA — EdUPro Professional Names
# ─────────────────────────────────────────────

SEGMENT_NAMES = {
    0: '🎓 Eager Learners',
    1: '📜 CPD Collectors',
    2: '🔬 Clinical Experts',
    3: '👀 Curious Browsers'
}

SEGMENT_COLORS = {
    0: '#4361ee',
    1: '#f72585',
    2: '#7209b7',
    3: '#4cc9f0'
}

SEGMENT_DESC = {
    0: 'Newly qualified professionals exploring many CPD areas. High course volume, beginner-intermediate level.',
    1: 'Mid-career professionals actively accumulating HPCSA/SAPIK accreditation points across specialties.',
    2: 'Veteran therapists going deep into specific disorders. Advanced level. Highest spend-per-course.',
    3: 'Teachers, parents and allied professionals occasionally exploring EdUPro content. Re-activation opportunity.'
}

# ─────────────────────────────────────────────
# ML CONSTANTS
# ─────────────────────────────────────────────

CLUSTERING_FEATURES = [
    'total_courses',
    'avg_spending',
    'avg_course_rating',
    'diversity_score',
    'learning_depth_index',
    'beginner_ratio',
    'enrollment_frequency',
    'category_concentration',
    'n_categories',
    'recency_days',
    'preferred_category_enc',
    'preferred_level_enc'
]

N_USERS     = 2000
N_COURSES   = 400
N_CLUSTERS  = 4
RANDOM_SEED = 42

SCORE_WEIGHTS = {
    'popularity': 0.30,
    'category':   0.25,
    'level':      0.20,
    'rating':     0.25,
}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def depth_index_to_level(depth_idx: float) -> str:
    if depth_idx < 0.33:
        return 'Beginner'
    elif depth_idx < 0.67:
        return 'Intermediate'
    else:
        return 'Advanced'


def adjacent_level(level: str) -> str:
    mapping = {
        'Beginner':     'Intermediate',
        'Intermediate': 'Advanced',
        'Advanced':     'Intermediate'
    }
    return mapping.get(level, '')


def profession_to_segment_hint(profession: str) -> int:
    """
    For cold-start new users — gives a segment hint
    based on profession type before any enrollment history.
    0=Eager, 1=CPD Collector, 2=Clinical Expert, 3=Browser
    """
    mapping = {
        'Occupational Therapist': 2,
        'Physiotherapist':        2,
        'Kinderkineticist':       1,
        'Classroom Teacher':      3,
        'School Psychologist':    1,
        'Parent/Caregiver':       3,
    }
    return mapping.get(profession, 0)
