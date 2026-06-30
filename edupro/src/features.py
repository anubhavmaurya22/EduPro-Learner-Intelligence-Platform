"""
EduPro Learner Intelligence Platform
File: src/features.py

Turns raw users / courses / transactions into one row per user
with 12+ engineered features, matching notebooks/02_feature_engineering.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .utils import CATEGORIES, LEVELS


def engineer_features(users: pd.DataFrame,
                      courses: pd.DataFrame,
                      transactions: pd.DataFrame,
                      reference_date: str = '2026-01-01') -> pd.DataFrame:
    """
    Builds the learner_profiles table — one row per user.

    Returns a DataFrame with columns:
      UserID, total_courses, total_spending, avg_spending,
      avg_course_rating, n_categories, preferred_category,
      preferred_level, Beginner/Intermediate/Advanced counts,
      enrollment_frequency, diversity_score, category_concentration,
      learning_depth_index, beginner_ratio, advanced_ratio,
      recency_days, preferred_category_enc, preferred_level_enc,
      Age, Gender
    """
    tx = transactions.copy()
    tx['TransactionDate'] = pd.to_datetime(tx['TransactionDate'])
    tx = tx.merge(courses, on='CourseID', how='left')

    ref_date = pd.to_datetime(reference_date)

    # ── Step 1: Basic aggregation ────────────
    profiles = tx.groupby('UserID').agg(
        total_courses      = ('CourseID',         'nunique'),
        total_transactions = ('CourseID',          'count'),
        total_spending      = ('Amount',            'sum'),
        avg_spending        = ('Amount',            'mean'),
        min_spending        = ('Amount',            'min'),
        max_spending        = ('Amount',            'max'),
        avg_course_rating   = ('CourseRating',      'mean'),
        n_categories        = ('CourseCategory',    'nunique'),
        first_enroll        = ('TransactionDate',   'min'),
        last_enroll         = ('TransactionDate',   'max'),
    ).reset_index()

    # ── Step 2: Preferred category ───────────
    pref_cat = (tx.groupby(['UserID', 'CourseCategory'])
                  .size().reset_index(name='cnt')
                  .sort_values('cnt', ascending=False)
                  .drop_duplicates('UserID')[['UserID', 'CourseCategory']]
                  .rename(columns={'CourseCategory': 'preferred_category'}))

    # ── Step 3: Preferred level ──────────────
    pref_lvl = (tx.groupby(['UserID', 'CourseLevel'])
                  .size().reset_index(name='cnt')
                  .sort_values('cnt', ascending=False)
                  .drop_duplicates('UserID')[['UserID', 'CourseLevel']]
                  .rename(columns={'CourseLevel': 'preferred_level'}))

    # ── Step 4: Level counts (for depth index) ──
    lvl_counts = (tx.groupby(['UserID', 'CourseLevel'])
                    .size().unstack(fill_value=0)
                    .reset_index())
    for lv in LEVELS:
        if lv not in lvl_counts.columns:
            lvl_counts[lv] = 0

    # ── Step 5: Merge everything ─────────────
    profiles = (profiles
                .merge(pref_cat, on='UserID', how='left')
                .merge(pref_lvl, on='UserID', how='left')
                .merge(lvl_counts[['UserID', 'Beginner', 'Intermediate', 'Advanced']],
                       on='UserID', how='left')
                .merge(users[['UserID', 'Age', 'Gender']], on='UserID', how='left'))

    # ── Step 6: Derived features ─────────────
    duration_days = (profiles['last_enroll'] - profiles['first_enroll']).dt.days + 1
    profiles['enrollment_frequency']  = profiles['total_courses'] / duration_days
    profiles['diversity_score']        = profiles['n_categories']
    profiles['category_concentration'] = 1 - (profiles['n_categories'] / len(CATEGORIES))

    total_lvl = (profiles['Beginner'] + profiles['Intermediate'] + profiles['Advanced']).replace(0, 1)
    profiles['beginner_ratio']       = profiles['Beginner'] / total_lvl
    profiles['advanced_ratio']       = profiles['Advanced'] / total_lvl
    profiles['learning_depth_index'] = (
        (profiles['Intermediate'] + 2 * profiles['Advanced']) / (2 * total_lvl)
    )
    profiles['recency_days'] = (ref_date - profiles['last_enroll']).dt.days

    # ── Step 7: Encode categoricals ──────────
    le_cat = LabelEncoder().fit(CATEGORIES)
    le_lvl = LabelEncoder().fit(LEVELS)
    profiles['preferred_category_enc'] = le_cat.transform(
        profiles['preferred_category'].fillna('Technology')
    )
    profiles['preferred_level_enc'] = le_lvl.transform(
        profiles['preferred_level'].fillna('Beginner')
    )

    # ── Step 8: Final cleanup ────────────────
    profiles = profiles.fillna(0)
    profiles = profiles.replace([np.inf, -np.inf], 0)

    return profiles


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == '__main__':
    from data_pipeline import generate_data

    print("Generating sample data...")
    users, courses, transactions = generate_data(n_users=200, n_courses=80)

    print("Engineering features...")
    profiles = engineer_features(users, courses, transactions)

    print(f"  Profiles shape : {profiles.shape}")
    print(f"  Null count     : {profiles.isnull().sum().sum()}")
    print(f"  Columns        : {profiles.columns.tolist()}")
    print("features.py self-test passed ✅")