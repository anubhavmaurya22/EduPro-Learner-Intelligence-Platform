


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pytest

from src.data_pipeline import generate_data
from src.features import engineer_features
from src.clustering import run_clustering, attach_cluster_labels
from src.recommender import (
    recommend_courses, recommend_new_user,
    evaluate_recommendation_quality
)
from src.utils import CATEGORIES, LEVELS


# -- Fixtures --------------------------------

@pytest.fixture(scope='module')
def setup_data():
    """Build a small but complete pipeline once for all tests."""
    users, courses, transactions = generate_data(n_users=300, n_courses=100, seed=42)
    profiles = engineer_features(users, courses, transactions)
    cr = run_clustering(profiles)
    profiles = attach_cluster_labels(profiles, cr)
    return profiles, courses, transactions


# -- Basic Recommendation Tests ----------------

def test_recommend_returns_correct_count(setup_data):
    profiles, courses, transactions = setup_data
    uid = profiles['UserID'].iloc[0]
    recs = recommend_courses(uid, profiles, courses, transactions, top_n=8)
    assert len(recs) <= 8
    assert len(recs) > 0


def test_recommend_unknown_user_returns_empty(setup_data):
    profiles, courses, transactions = setup_data
    recs = recommend_courses('U9999_FAKE', profiles, courses, transactions)
    assert recs.empty


def test_recommend_excludes_already_taken(setup_data):
    """A course the user already enrolled in should never be recommended again."""
    profiles, courses, transactions = setup_data
    uid = profiles['UserID'].iloc[0]
    taken = set(transactions[transactions['UserID'] == uid]['CourseID'])

    recs = recommend_courses(uid, profiles, courses, transactions, top_n=8)
    recommended_ids = set(recs['CourseID'])

    assert len(taken & recommended_ids) == 0


def test_recommend_scores_between_0_and_1(setup_data):
    profiles, courses, transactions = setup_data
    uid = profiles['UserID'].iloc[5]
    recs = recommend_courses(uid, profiles, courses, transactions, top_n=8)
    assert (recs['RecommendationScore'] >= 0).all()
    assert (recs['RecommendationScore'] <= 1).all()


def test_recommend_sorted_descending(setup_data):
    """Top recommendation must have the highest score."""
    profiles, courses, transactions = setup_data
    uid = profiles['UserID'].iloc[3]
    recs = recommend_courses(uid, profiles, courses, transactions, top_n=8)
    scores = recs['RecommendationScore'].tolist()
    assert scores == sorted(scores, reverse=True)


def test_recommend_rank_starts_at_1(setup_data):
    profiles, courses, transactions = setup_data
    uid = profiles['UserID'].iloc[0]
    recs = recommend_courses(uid, profiles, courses, transactions, top_n=5)
    assert recs['Rank'].iloc[0] == 1
    assert recs['Rank'].tolist() == list(range(1, len(recs) + 1))


def test_recommend_has_match_reason(setup_data):
    """Every recommendation must have a non-empty explanation."""
    profiles, courses, transactions = setup_data
    uid = profiles['UserID'].iloc[0]
    recs = recommend_courses(uid, profiles, courses, transactions, top_n=5)
    assert (recs['MatchReason'].str.len() > 0).all()


# -- Filter Tests -------------------------------

def test_recommend_category_filter_applied(setup_data):
    profiles, courses, transactions = setup_data
    uid = profiles['UserID'].iloc[0]
    test_category = CATEGORIES[0]
    recs = recommend_courses(
        uid, profiles, courses, transactions,
        top_n=8, filter_category=test_category
    )
    if not recs.empty:
        assert (recs['CourseCategory'] == test_category).all()


def test_recommend_level_filter_applied(setup_data):
    profiles, courses, transactions = setup_data
    uid = profiles['UserID'].iloc[0]
    test_level = 'Beginner'
    recs = recommend_courses(
        uid, profiles, courses, transactions,
        top_n=8, filter_level=test_level
    )
    if not recs.empty:
        assert (recs['CourseLevel'] == test_level).all()


def test_recommend_filter_with_no_matches_returns_empty(setup_data):
    """An impossible filter combo should return empty, not crash."""
    profiles, courses, transactions = setup_data
    uid = profiles['UserID'].iloc[0]

    impossible_courses = courses[
        (courses['CourseCategory'] == 'Technology') &
        (courses['CourseLevel'] == 'Advanced')
    ]
    if impossible_courses.empty:
        pytest.skip("No Technology/Advanced courses generated in this seed")

    fake_tx = pd.concat([
        transactions,
        pd.DataFrame({
            'UserID': [uid] * len(impossible_courses),
            'CourseID': impossible_courses['CourseID'].values,
            'TransactionDate': pd.Timestamp('2025-01-01'),
            'Amount': 50.0
        })
    ])

    recs = recommend_courses(
        uid, profiles, courses, fake_tx,
        top_n=8, filter_category='Technology', filter_level='Advanced'
    )
    assert recs.empty


# -- Cold Start Tests ---------------------------

def test_cold_start_returns_results(setup_data):
    _, courses, _ = setup_data
    recs = recommend_new_user(
        courses, goal='Career', category='Technology',
        level='Beginner', top_n=5
    )
    assert len(recs) > 0
    assert len(recs) <= 5


def test_cold_start_sorted_by_rating(setup_data):
    _, courses, _ = setup_data
    recs = recommend_new_user(
        courses, goal='Skill', category='Business',
        level='Intermediate', top_n=5
    )
    ratings = recs['CourseRating'].tolist()
    assert ratings == sorted(ratings, reverse=True)


def test_cold_start_all_categories_work(setup_data):
    """Every category should produce at least one recommendation."""
    _, courses, _ = setup_data
    for cat in CATEGORIES:
        recs = recommend_new_user(
            courses, goal='Hobby', category=cat,
            level='Beginner', top_n=3
        )
        assert len(recs) > 0, f"No recommendations for category: {cat}"


# -- Quality Evaluation Tests -------------------

def test_evaluate_quality_returns_dataframe(setup_data):
    profiles, courses, transactions = setup_data
    quality = evaluate_recommendation_quality(
        profiles, courses, transactions, sample_size=20, seed=42
    )
    assert isinstance(quality, pd.DataFrame)
    assert len(quality) > 0


def test_evaluate_quality_columns_present(setup_data):
    profiles, courses, transactions = setup_data
    quality = evaluate_recommendation_quality(
        profiles, courses, transactions, sample_size=20, seed=42
    )
    expected_cols = ['UserID', 'AvgScore', 'AvgRating', 'AvgCatMatch', 'AvgLvlMatch', 'AvgPop']
    for col in expected_cols:
        assert col in quality.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
