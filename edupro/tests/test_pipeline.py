
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pytest

from src.data_pipeline import generate_data
from src.features import engineer_features
from src.clustering import run_clustering, get_cluster_profiles, attach_cluster_labels
from src.utils import CLUSTERING_FEATURES, N_CLUSTERS


# -- Fixtures --------------------------------

@pytest.fixture(scope='module')
def raw_data():
    """Generate a small dataset once and reuse across tests (faster)."""
    return generate_data(n_users=300, n_courses=100, seed=42)


@pytest.fixture(scope='module')
def profiles(raw_data):
    users, courses, transactions = raw_data
    return engineer_features(users, courses, transactions)


@pytest.fixture(scope='module')
def cluster_result(profiles):
    return run_clustering(profiles)


# -- Data Generation Tests --------------------

def test_generate_data_shapes(raw_data):
    users, courses, transactions = raw_data
    assert len(users) == 300
    assert len(courses) == 100
    assert len(transactions) > 0


def test_generate_data_no_nulls(raw_data):
    users, courses, transactions = raw_data
    assert users.isnull().sum().sum() == 0
    assert courses.isnull().sum().sum() == 0
    assert transactions.isnull().sum().sum() == 0


def test_generate_data_reproducible():
    """Same seed must produce identical data -- critical for debugging."""
    u1, c1, t1 = generate_data(n_users=50, n_courses=20, seed=42)
    u2, c2, t2 = generate_data(n_users=50, n_courses=20, seed=42)
    pd.testing.assert_frame_equal(u1, u2)
    pd.testing.assert_frame_equal(c1, c2)


def test_transaction_amounts_positive(raw_data):
    _, _, transactions = raw_data
    assert (transactions['Amount'] > 0).all()


def test_transaction_users_exist(raw_data):
    """Every UserID in transactions must exist in users table."""
    users, _, transactions = raw_data
    assert transactions['UserID'].isin(users['UserID']).all()


def test_transaction_courses_exist(raw_data):
    """Every CourseID in transactions must exist in courses table."""
    _, courses, transactions = raw_data
    assert transactions['CourseID'].isin(courses['CourseID']).all()


# -- Feature Engineering Tests -----------------

def test_profiles_one_row_per_user(raw_data, profiles):
    users, _, transactions = raw_data
    enrolled_users = transactions['UserID'].nunique()
    assert len(profiles) == enrolled_users


def test_profiles_no_nulls(profiles):
    assert profiles.isnull().sum().sum() == 0


def test_profiles_no_infinities(profiles):
    import numpy as np
    numeric = profiles.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values).any()


def test_clustering_features_present(profiles):
    for feat in CLUSTERING_FEATURES:
        assert feat in profiles.columns, f"Missing feature: {feat}"


def test_learning_depth_index_bounds(profiles):
    """Depth index should always be between 0 and 1."""
    assert (profiles['learning_depth_index'] >= 0).all()
    assert (profiles['learning_depth_index'] <= 1).all()


def test_beginner_ratio_bounds(profiles):
    assert (profiles['beginner_ratio'] >= 0).all()
    assert (profiles['beginner_ratio'] <= 1).all()


def test_total_courses_positive(profiles):
    assert (profiles['total_courses'] > 0).all()


# -- Clustering Tests ---------------------------

def test_cluster_result_keys(cluster_result):
    required_keys = [
        'scaler', 'km_model', 'labels', 'sil_vals',
        'overall_silhouette', 'X_pca'
    ]
    for key in required_keys:
        assert key in cluster_result


def test_no_nan_in_cluster_labels(cluster_result):
    """Regression test for the NaN-cluster bug from Day 3."""
    labels = cluster_result['labels']
    assert pd.isna(labels).sum() == 0


def test_cluster_labels_in_valid_range(cluster_result):
    labels = cluster_result['labels']
    assert set(labels).issubset({0, 1, 2, 3})


def test_all_four_segments_present(cluster_result):
    """Every one of the 4 named segments should have at least 1 user."""
    labels = cluster_result['labels']
    assert len(set(labels)) == N_CLUSTERS


def test_silhouette_score_reasonable(cluster_result):
    """Silhouette should be positive -- confirms clusters aren't random noise."""
    assert cluster_result['overall_silhouette'] > 0


def test_pca_shape(profiles, cluster_result):
    assert cluster_result['X_pca'].shape == (len(profiles), 2)


def test_cluster_profiles_sums_to_total(profiles, cluster_result):
    cp = get_cluster_profiles(profiles, cluster_result)
    assert cp['n_users'].sum() == len(profiles)


def test_attach_cluster_labels_adds_columns(profiles, cluster_result):
    labeled = attach_cluster_labels(profiles, cluster_result)
    for col in ['Cluster', 'SegmentName', 'PCA_1', 'PCA_2', 'SilhouetteVal']:
        assert col in labeled.columns
    # Original profiles must be untouched (function returns a copy)
    assert 'Cluster' not in profiles.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
