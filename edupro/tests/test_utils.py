"""
EduPro Learner Intelligence Platform
File: tests/test_utils.py

Test suite for src/utils.py:
  Constants, depth_index_to_level(), adjacent_level(), safe_divide()
Covers happy-path, boundary/edge cases, and error handling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    CATEGORIES, LEVELS, COURSE_TYPES, GENDERS,
    SEGMENT_NAMES, SEGMENT_COLORS, SEGMENT_DESC,
    CLUSTERING_FEATURES, SCORE_WEIGHTS,
    N_USERS, N_COURSES, N_CLUSTERS, RANDOM_SEED,
    depth_index_to_level, adjacent_level, safe_divide
)


# ── Constants sanity checks ───────────────────────────────────────────────────

def test_categories_list_length():
    assert len(CATEGORIES) == 8


def test_levels_list_length():
    assert len(LEVELS) == 3


def test_levels_correct_values():
    assert set(LEVELS) == {'Beginner', 'Intermediate', 'Advanced'}


def test_segment_names_has_four_entries():
    assert len(SEGMENT_NAMES) == 4


def test_segment_names_keys_are_0_to_3():
    assert set(SEGMENT_NAMES.keys()) == {0, 1, 2, 3}


def test_segment_colors_match_segment_names():
    assert set(SEGMENT_COLORS.keys()) == set(SEGMENT_NAMES.keys())


def test_segment_desc_match_segment_names():
    assert set(SEGMENT_DESC.keys()) == set(SEGMENT_NAMES.keys())


def test_clustering_features_count():
    """Should have exactly 12 features for the clustering pipeline."""
    assert len(CLUSTERING_FEATURES) == 12


def test_score_weights_sum_to_one():
    total = sum(SCORE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Score weights sum to {total}, expected 1.0"


def test_score_weights_all_positive():
    assert all(v > 0 for v in SCORE_WEIGHTS.values())


def test_n_clusters_equals_four():
    """N_CLUSTERS must equal the number of named segments."""
    assert N_CLUSTERS == len(SEGMENT_NAMES)


def test_random_seed_is_integer():
    assert isinstance(RANDOM_SEED, int)


def test_n_users_and_n_courses_positive():
    assert N_USERS > 0
    assert N_COURSES > 0


# ── depth_index_to_level() — Happy Path ───────────────────────────────────────

def test_depth_low_returns_beginner():
    assert depth_index_to_level(0.0)  == 'Beginner'
    assert depth_index_to_level(0.10) == 'Beginner'
    assert depth_index_to_level(0.32) == 'Beginner'


def test_depth_mid_returns_intermediate():
    assert depth_index_to_level(0.33) == 'Intermediate'
    assert depth_index_to_level(0.50) == 'Intermediate'
    assert depth_index_to_level(0.66) == 'Intermediate'


def test_depth_high_returns_advanced():
    assert depth_index_to_level(0.67) == 'Advanced'
    assert depth_index_to_level(0.90) == 'Advanced'
    assert depth_index_to_level(1.00) == 'Advanced'


def test_depth_returns_string():
    assert isinstance(depth_index_to_level(0.5), str)


def test_depth_result_always_in_levels():
    test_values = [0.0, 0.1, 0.32, 0.33, 0.5, 0.66, 0.67, 0.99, 1.0]
    for v in test_values:
        result = depth_index_to_level(v)
        assert result in LEVELS, f"depth_index_to_level({v}) = '{result}' not in LEVELS"


# ── depth_index_to_level() — Edge / Boundary Cases ───────────────────────────

def test_depth_exact_boundary_0_33():
    """0.33 is the first value that switches to Intermediate."""
    assert depth_index_to_level(0.33) == 'Intermediate'


def test_depth_exact_boundary_0_67():
    """0.67 is the first value that switches to Advanced."""
    assert depth_index_to_level(0.67) == 'Advanced'


def test_depth_negative_value_returns_beginner():
    """Negative depth index (invalid but defensive) should be treated as Beginner."""
    assert depth_index_to_level(-0.1) == 'Beginner'


def test_depth_value_above_1():
    """Values above 1 (invalid but defensive) should return Advanced."""
    assert depth_index_to_level(1.5) == 'Advanced'


# ── adjacent_level() — Happy Path ────────────────────────────────────────────

def test_adjacent_beginner_is_intermediate():
    assert adjacent_level('Beginner') == 'Intermediate'


def test_adjacent_intermediate_is_advanced():
    assert adjacent_level('Intermediate') == 'Advanced'


def test_adjacent_advanced_is_intermediate():
    """Advanced pushes back to Intermediate (no higher level)."""
    assert adjacent_level('Advanced') == 'Intermediate'


def test_adjacent_level_returns_string():
    for lvl in LEVELS:
        assert isinstance(adjacent_level(lvl), str)


# ── adjacent_level() — Edge Cases ────────────────────────────────────────────

def test_adjacent_unknown_level_returns_empty_string():
    """Unknown level should return '' gracefully (not crash)."""
    result = adjacent_level('Expert')
    assert result == ''


def test_adjacent_empty_string_returns_empty_string():
    result = adjacent_level('')
    assert result == ''


# ── adjacent_level() — Error Handling ────────────────────────────────────────

def test_adjacent_none_input():
    """Passing None should either return '' or raise a TypeError."""
    try:
        result = adjacent_level(None)
        assert result == ''
    except (TypeError, AttributeError):
        pass  # Also acceptable


def test_adjacent_integer_input():
    """Passing an integer should either return '' or raise gracefully."""
    try:
        result = adjacent_level(0)
        assert result == ''
    except (TypeError, AttributeError):
        pass


# ── safe_divide() — Happy Path ────────────────────────────────────────────────

def test_safe_divide_normal_arrays():
    num = np.array([10.0, 20.0, 30.0])
    den = np.array([2.0,  4.0,  5.0])
    result = safe_divide(num, den)
    np.testing.assert_allclose(result, [5.0, 5.0, 6.0])


def test_safe_divide_with_pandas_series():
    num = pd.Series([100.0, 50.0])
    den = pd.Series([10.0,  5.0])
    result = safe_divide(num, den)
    np.testing.assert_allclose(result, [10.0, 10.0])


def test_safe_divide_scalar_denominator():
    num = np.array([6.0, 9.0, 12.0])
    den = np.array([3.0, 3.0,  3.0])
    result = safe_divide(num, den)
    np.testing.assert_allclose(result, [2.0, 3.0, 4.0])


# ── safe_divide() — Edge / Zero-Division Cases ───────────────────────────────

def test_safe_divide_zero_denominator_returns_fill():
    num = np.array([5.0, 10.0])
    den = np.array([0.0,  2.0])
    result = safe_divide(num, den, fill=0.0)
    assert result[0] == 0.0
    np.testing.assert_allclose(result[1], 5.0)


def test_safe_divide_custom_fill_value():
    num = np.array([1.0, 2.0, 3.0])
    den = np.array([0.0, 0.0, 1.0])
    result = safe_divide(num, den, fill=-1.0)
    assert result[0] == -1.0
    assert result[1] == -1.0
    np.testing.assert_allclose(result[2], 3.0)


def test_safe_divide_all_zeros_denominator():
    num = np.array([1.0, 2.0, 3.0])
    den = np.zeros(3)
    result = safe_divide(num, den, fill=0.0)
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])


def test_safe_divide_no_infinities():
    num = np.array([1.0, 2.0, 3.0])
    den = np.array([0.0, 0.0, 0.0])
    result = safe_divide(num, den)
    assert not np.isinf(result).any()


def test_safe_divide_no_nans():
    num = np.array([0.0, 1.0, 2.0])
    den = np.array([0.0, 0.0, 1.0])
    result = safe_divide(num, den)
    assert not np.isnan(result).any()


# ── safe_divide() — Error Handling ───────────────────────────────────────────

def test_safe_divide_mismatched_shapes_raises():
    """Mismatched array lengths should raise an error from numpy."""
    num = np.array([1.0, 2.0, 3.0])
    den = np.array([1.0, 2.0])
    with pytest.raises((ValueError, Exception)):
        safe_divide(num, den)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
