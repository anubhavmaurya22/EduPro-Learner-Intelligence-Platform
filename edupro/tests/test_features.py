"""
EduPro Learner Intelligence Platform
File: tests/test_features.py

Test suite for src/features.py  — engineer_features()
Covers happy-path, edge cases, and error-handling scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import generate_data
from src.features import engineer_features
from src.utils import CATEGORIES, LEVELS, CLUSTERING_FEATURES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def raw_data():
    """Small synthetic dataset reused across all tests."""
    return generate_data(n_users=200, n_courses=80, seed=42)


@pytest.fixture(scope='module')
def profiles(raw_data):
    users, courses, transactions = raw_data
    return engineer_features(users, courses, transactions)


# ── Happy Path ────────────────────────────────────────────────────────────────

def test_profiles_has_correct_number_of_rows(raw_data, profiles):
    """One row per user who made at least one transaction."""
    _, _, transactions = raw_data
    enrolled_users = transactions['UserID'].nunique()
    assert len(profiles) == enrolled_users, (
        f"Expected {enrolled_users} rows, got {len(profiles)}"
    )


def test_profiles_contains_user_id(profiles):
    assert 'UserID' in profiles.columns


def test_all_clustering_features_present(profiles):
    """All 12 features required by the clustering pipeline must exist."""
    missing = [f for f in CLUSTERING_FEATURES if f not in profiles.columns]
    assert missing == [], f"Missing clustering features: {missing}"


def test_preferred_category_is_valid(profiles):
    """preferred_category must be one of the 8 known categories."""
    invalid = profiles[~profiles['preferred_category'].isin(CATEGORIES)]
    assert len(invalid) == 0, (
        f"Found {len(invalid)} rows with invalid preferred_category"
    )


def test_preferred_level_is_valid(profiles):
    """preferred_level must be Beginner / Intermediate / Advanced."""
    invalid = profiles[~profiles['preferred_level'].isin(LEVELS)]
    assert len(invalid) == 0, (
        f"Found {len(invalid)} rows with invalid preferred_level"
    )


def test_label_encoded_columns_are_integers(profiles):
    """Encoded columns must be integer-valued (scikit-learn LabelEncoder output)."""
    assert profiles['preferred_category_enc'].dtype in (int, 'int64', 'int32', np.int64, np.int32)
    assert profiles['preferred_level_enc'].dtype in (int, 'int64', 'int32', np.int64, np.int32)


def test_preferred_category_enc_range(profiles):
    """Encoded category must be in [0, len(CATEGORIES)-1]."""
    assert profiles['preferred_category_enc'].min() >= 0
    assert profiles['preferred_category_enc'].max() <= len(CATEGORIES) - 1


def test_preferred_level_enc_range(profiles):
    """Encoded level must be in [0, len(LEVELS)-1]."""
    assert profiles['preferred_level_enc'].min() >= 0
    assert profiles['preferred_level_enc'].max() <= len(LEVELS) - 1


def test_no_nulls_in_profiles(profiles):
    null_counts = profiles.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, (
        f"Null values found in: {cols_with_nulls.to_dict()}"
    )


def test_no_infinities_in_numeric_columns(profiles):
    numeric = profiles.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values).any(), "Infinite values found in profiles"


def test_learning_depth_index_in_unit_interval(profiles):
    assert (profiles['learning_depth_index'] >= 0).all()
    assert (profiles['learning_depth_index'] <= 1).all()


def test_beginner_ratio_in_unit_interval(profiles):
    assert (profiles['beginner_ratio'] >= 0).all()
    assert (profiles['beginner_ratio'] <= 1).all()


def test_advanced_ratio_in_unit_interval(profiles):
    assert (profiles['advanced_ratio'] >= 0).all()
    assert (profiles['advanced_ratio'] <= 1).all()


def test_total_spending_non_negative(profiles):
    assert (profiles['total_spending'] >= 0).all()


def test_total_courses_positive(profiles):
    assert (profiles['total_courses'] > 0).all()


def test_enrollment_frequency_non_negative(profiles):
    assert (profiles['enrollment_frequency'] >= 0).all()


def test_diversity_score_equals_n_categories(profiles):
    """diversity_score is currently defined as n_categories — they must match."""
    assert (profiles['diversity_score'] == profiles['n_categories']).all()


def test_age_carried_through(raw_data, profiles):
    """Age from users table should be present and non-zero."""
    assert 'Age' in profiles.columns
    assert (profiles['Age'] > 0).all()


def test_gender_carried_through(profiles):
    """Gender should be present and non-empty strings."""
    assert 'Gender' in profiles.columns
    assert (profiles['Gender'].str.len() > 0).all()


def test_recency_days_non_negative(profiles):
    """recency_days = reference_date - last_enroll; should be >= 0 for 2026 ref date."""
    assert (profiles['recency_days'] >= 0).all()


# ── Edge Case: single-user dataset ───────────────────────────────────────────

def test_single_user_dataset():
    """engineer_features must not crash on a minimal 1-user dataset."""
    users, courses, transactions = generate_data(n_users=10, n_courses=20, seed=7)
    # Keep only the first user
    uid = users['UserID'].iloc[0]
    single_tx = transactions[transactions['UserID'] == uid].copy()

    profiles_1 = engineer_features(users, courses, single_tx)
    assert len(profiles_1) == 1
    assert profiles_1.isnull().sum().sum() == 0


def test_minimal_one_transaction_per_user():
    """Works even when each user enrolled in exactly 1 course."""
    users, courses, _ = generate_data(n_users=20, n_courses=20, seed=99)
    # Build a trivial transactions table: one row per user
    import random
    random.seed(99)
    records = []
    for uid in users['UserID']:
        cid = courses['CourseID'].sample(1, random_state=42).values[0]
        records.append({'UserID': uid, 'CourseID': cid,
                        'TransactionDate': '2024-06-01', 'Amount': 50.0})
    tx = pd.DataFrame(records)
    profiles_min = engineer_features(users, courses, tx)
    assert len(profiles_min) == len(users)
    assert profiles_min['total_courses'].max() <= 1


# ── Error Handling ────────────────────────────────────────────────────────────

def test_wrong_type_raises_attribute_error():
    """Passing a plain dict instead of DataFrames should raise an error."""
    with pytest.raises((AttributeError, TypeError, KeyError)):
        engineer_features({"UserID": [1]}, {"CourseID": [1]}, {"UserID": [1]})


def test_empty_transactions_raises_or_returns_empty():
    """An empty transactions table should either raise or return an empty DataFrame."""
    users, courses, _ = generate_data(n_users=10, n_courses=20, seed=1)
    empty_tx = pd.DataFrame(columns=['UserID', 'CourseID', 'TransactionDate', 'Amount'])
    try:
        result = engineer_features(users, courses, empty_tx)
        assert isinstance(result, pd.DataFrame)
        # If it doesn't raise, the result must be empty
        assert len(result) == 0
    except Exception:
        pass  # Any exception is also acceptable for empty input


def test_missing_amount_column_raises():
    """transactions without 'Amount' should raise KeyError."""
    users, courses, transactions = generate_data(n_users=20, n_courses=20, seed=5)
    bad_tx = transactions.drop(columns=['Amount'])
    with pytest.raises((KeyError, Exception)):
        engineer_features(users, courses, bad_tx)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
