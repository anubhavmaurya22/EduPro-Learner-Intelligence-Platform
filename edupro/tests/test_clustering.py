"""
EduPro Learner Intelligence Platform
File: tests/test_clustering.py

Test suite for src/clustering.py:
  run_clustering(), _map_clusters_to_segments(),
  get_cluster_profiles(), attach_cluster_labels()
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
from src.clustering import (
    run_clustering, get_cluster_profiles, attach_cluster_labels
)
from src.utils import CLUSTERING_FEATURES, N_CLUSTERS, SEGMENT_NAMES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def raw_data():
    return generate_data(n_users=300, n_courses=100, seed=42)


@pytest.fixture(scope='module')
def profiles(raw_data):
    users, courses, transactions = raw_data
    return engineer_features(users, courses, transactions)


@pytest.fixture(scope='module')
def cluster_result(profiles):
    return run_clustering(profiles)


@pytest.fixture(scope='module')
def labeled_profiles(profiles, cluster_result):
    return attach_cluster_labels(profiles, cluster_result)


# ── Happy Path: run_clustering() output structure ─────────────────────────────

def test_cluster_result_is_dict(cluster_result):
    assert isinstance(cluster_result, dict)


def test_cluster_result_has_all_required_keys(cluster_result):
    required = [
        'scaler', 'km_model', 'pca', 'X_scaled', 'X_pca',
        'raw_labels', 'labels', 'cluster_mapping',
        'k_range', 'inertias', 'sil_scores', 'sil_vals',
        'overall_silhouette', 'hier_labels', 'hier_sample_idx'
    ]
    for key in required:
        assert key in cluster_result, f"Missing key: '{key}'"


def test_labels_length_matches_profiles(profiles, cluster_result):
    assert len(cluster_result['labels']) == len(profiles)


def test_raw_labels_length_matches_profiles(profiles, cluster_result):
    assert len(cluster_result['raw_labels']) == len(profiles)


def test_sil_vals_length_matches_profiles(profiles, cluster_result):
    assert len(cluster_result['sil_vals']) == len(profiles)


def test_x_pca_shape(profiles, cluster_result):
    assert cluster_result['X_pca'].shape == (len(profiles), 2)


def test_x_scaled_shape(profiles, cluster_result):
    n_features = len(CLUSTERING_FEATURES)
    assert cluster_result['X_scaled'].shape == (len(profiles), n_features)


# ── Happy Path: label correctness ─────────────────────────────────────────────

def test_no_nan_in_labels(cluster_result):
    """Regression test for the NaN-cluster bug."""
    assert pd.isna(cluster_result['labels']).sum() == 0


def test_labels_are_integers(cluster_result):
    assert cluster_result['labels'].dtype in (int, 'int64', 'int32', np.int64, np.int32)


def test_labels_only_contain_valid_segment_ids(cluster_result):
    valid = set(SEGMENT_NAMES.keys())  # {0, 1, 2, 3}
    actual = set(cluster_result['labels'].tolist())
    assert actual.issubset(valid), f"Invalid segment IDs found: {actual - valid}"


def test_all_four_segments_present(cluster_result):
    """With 300 users all 4 named segments must be populated."""
    assert len(set(cluster_result['labels'])) == N_CLUSTERS


def test_cluster_mapping_covers_all_raw_labels(cluster_result):
    raw = set(cluster_result['raw_labels'].tolist())
    mapped = set(cluster_result['cluster_mapping'].keys())
    assert raw == mapped, f"Mapping is missing raw labels: {raw - mapped}"


# ── Happy Path: diagnostic metrics ───────────────────────────────────────────

def test_overall_silhouette_is_positive(cluster_result):
    """Silhouette > 0 confirms clusters are non-trivial."""
    assert cluster_result['overall_silhouette'] > 0, (
        f"Silhouette = {cluster_result['overall_silhouette']:.4f}; expected > 0"
    )


def test_silhouette_above_submission_threshold(cluster_result):
    """Project requirement: silhouette must exceed 0.2."""
    assert cluster_result['overall_silhouette'] > 0.2, (
        f"Silhouette = {cluster_result['overall_silhouette']:.4f}; required > 0.2"
    )


def test_inertias_decrease_monotonically(cluster_result):
    """KMeans inertia must strictly decrease as K grows."""
    inertias = cluster_result['inertias']
    for i in range(1, len(inertias)):
        assert inertias[i] < inertias[i - 1], (
            f"Inertia did not decrease at k={cluster_result['k_range'][i]}"
        )


def test_k_range_and_inertias_same_length(cluster_result):
    assert len(cluster_result['k_range']) == len(cluster_result['inertias'])
    assert len(cluster_result['k_range']) == len(cluster_result['sil_scores'])


# ── Happy Path: get_cluster_profiles() ───────────────────────────────────────

def test_cluster_profiles_n_users_sums_to_total(profiles, cluster_result):
    cp = get_cluster_profiles(profiles, cluster_result)
    assert cp['n_users'].sum() == len(profiles)


def test_cluster_profiles_has_four_rows(cluster_result, profiles):
    cp = get_cluster_profiles(profiles, cluster_result)
    assert len(cp) == N_CLUSTERS


def test_cluster_profiles_has_segment_name_column(profiles, cluster_result):
    cp = get_cluster_profiles(profiles, cluster_result)
    assert 'SegmentName' in cp.columns


def test_cluster_profiles_all_mean_courses_positive(profiles, cluster_result):
    cp = get_cluster_profiles(profiles, cluster_result)
    assert (cp['total_courses'] > 0).all()


# ── Happy Path: attach_cluster_labels() ──────────────────────────────────────

def test_attach_adds_cluster_column(labeled_profiles):
    assert 'Cluster' in labeled_profiles.columns


def test_attach_adds_segment_name(labeled_profiles):
    assert 'SegmentName' in labeled_profiles.columns


def test_attach_adds_pca_columns(labeled_profiles):
    assert 'PCA_1' in labeled_profiles.columns
    assert 'PCA_2' in labeled_profiles.columns


def test_attach_adds_silhouette_val(labeled_profiles):
    assert 'SilhouetteVal' in labeled_profiles.columns


def test_attach_does_not_mutate_original(profiles, cluster_result):
    """attach_cluster_labels must return a COPY — original should be unmodified."""
    _ = attach_cluster_labels(profiles, cluster_result)
    assert 'Cluster' not in profiles.columns


def test_segment_names_match_utils(labeled_profiles):
    """SegmentName values must be taken from SEGMENT_NAMES dict."""
    expected_names = set(SEGMENT_NAMES.values())
    actual_names   = set(labeled_profiles['SegmentName'].unique())
    assert actual_names.issubset(expected_names)


# ── Edge Case: small dataset ──────────────────────────────────────────────────

def test_clustering_on_small_dataset():
    """run_clustering should work with as few as 50 users."""
    users, courses, transactions = generate_data(n_users=50, n_courses=30, seed=7)
    profiles_small = engineer_features(users, courses, transactions)
    cr = run_clustering(profiles_small)
    assert 'labels' in cr
    assert pd.isna(cr['labels']).sum() == 0


def test_deterministic_with_same_seed(profiles):
    """Same seed must produce identical cluster labels."""
    cr1 = run_clustering(profiles, seed=42)
    cr2 = run_clustering(profiles, seed=42)
    np.testing.assert_array_equal(cr1['labels'], cr2['labels'])


# ── Error Handling ────────────────────────────────────────────────────────────

def test_missing_clustering_feature_raises():
    """Passing profiles without a required feature should raise KeyError."""
    users, courses, transactions = generate_data(n_users=100, n_courses=40, seed=3)
    profiles_bad = engineer_features(users, courses, transactions)
    profiles_bad = profiles_bad.drop(columns=['total_courses'])
    with pytest.raises((KeyError, Exception)):
        run_clustering(profiles_bad)


def test_wrong_input_type_raises():
    """Passing a list instead of DataFrame should raise an AttributeError."""
    with pytest.raises((AttributeError, TypeError, ValueError)):
        run_clustering([1, 2, 3])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
