"""
EduPro Learner Intelligence Platform
File: src/recommender.py

Hybrid recommendation engine, matching notebooks/04_recommendations.ipynb.
Combines cluster popularity, category match, level fit, and rating
quality into a single weighted score per course.
"""

import warnings

import pandas as pd

from .utils import (
    SEGMENT_NAMES, SCORE_WEIGHTS,
    depth_index_to_level, adjacent_level
)


def recommend_courses(user_id: str,
                      profiles: pd.DataFrame,
                      courses: pd.DataFrame,
                      transactions: pd.DataFrame,
                      top_n: int = 8,
                      filter_category: str = None,
                      filter_level: str = None) -> pd.DataFrame:
    """
    Returns top_n recommended courses for a user.

    `profiles` MUST already have a 'Cluster' column
    (use clustering.attach_cluster_labels() first).

    Scoring (weights from utils.SCORE_WEIGHTS):
      30% cluster peer popularity
      25% category preference match
      20% level fit (pushes user to next level)
      25% course rating quality
    """
    user_row = profiles[profiles['UserID'] == user_id]
    if user_row.empty:
        return pd.DataFrame()

    user_row     = user_row.iloc[0]
    user_cluster = int(user_row['Cluster'])
    pref_cat     = user_row['preferred_category']
    depth_idx    = float(user_row['learning_depth_index'])
    target_level = depth_index_to_level(depth_idx)

    taken = set(transactions[transactions['UserID'] == user_id]['CourseID'])
    candidates = courses[~courses['CourseID'].isin(taken)].copy()

    if filter_category and filter_category != 'All':
        candidates = candidates[candidates['CourseCategory'] == filter_category]
    if filter_level and filter_level != 'All':
        candidates = candidates[candidates['CourseLevel'] == filter_level]

    if candidates.empty:
        return pd.DataFrame()

    # ── Cluster popularity ───────────────────
    peer_ids = profiles[profiles['Cluster'] == user_cluster]['UserID']
    peer_tx  = transactions[transactions['UserID'].isin(peer_ids)]
    pop = peer_tx.groupby('CourseID').size().reset_index(name='peer_count')

    candidates = candidates.merge(pop, on='CourseID', how='left')
    candidates['peer_count'] = candidates['peer_count'].fillna(0)
    max_pop = candidates['peer_count'].max() or 1
    candidates['pop_score'] = candidates['peer_count'] / max_pop

    # ── Category match ───────────────────────
    candidates['cat_score'] = (candidates['CourseCategory'] == pref_cat).astype(float)

    # ── Level fit (with partial credit) ──────
    candidates['lvl_score'] = (candidates['CourseLevel'] == target_level).astype(float)
    adj = adjacent_level(target_level)
    candidates.loc[candidates['CourseLevel'] == adj, 'lvl_score'] = 0.5

    # ── Rating quality ───────────────────────
    candidates['rating_score'] = (candidates['CourseRating'] - 1) / 4

    # ── Final weighted score ─────────────────
    candidates['RecommendationScore'] = (
        SCORE_WEIGHTS['popularity'] * candidates['pop_score']    +
        SCORE_WEIGHTS['category']   * candidates['cat_score']    +
        SCORE_WEIGHTS['level']      * candidates['lvl_score']    +
        SCORE_WEIGHTS['rating']     * candidates['rating_score']
    )

    top = (candidates.sort_values('RecommendationScore', ascending=False)
                     .head(top_n)
                     .reset_index(drop=True))

    top['Rank']        = top.index + 1
    top['UserID']      = user_id
    top['Segment']     = SEGMENT_NAMES[user_cluster]
    top['TargetLevel'] = target_level
    top['MatchReason'] = top.apply(
        _match_reason, axis=1, args=(pref_cat, target_level)
    )

    return top[[
        'Rank', 'UserID', 'CourseID', 'CourseCategory', 'CourseType',
        'CourseLevel', 'CourseRating', 'RecommendationScore',
        'pop_score', 'cat_score', 'lvl_score', 'rating_score',
        'Segment', 'TargetLevel', 'MatchReason'
    ]]


def _match_reason(row, pref_cat, target_level) -> str:
    """Human-readable explanation for why a course was recommended."""
    reasons = []
    if row['cat_score'] == 1.0:
        reasons.append(f"Matches your {pref_cat} interest")
    if row['lvl_score'] == 1.0:
        reasons.append(f"Perfect {target_level} level fit")
    elif row['lvl_score'] == 0.5:
        reasons.append("Close to your target level")
    if row['pop_score'] > 0.6:
        reasons.append("Popular in your learner group")
    if row['CourseRating'] >= 4.5:
        reasons.append(f"Highly rated ({row['CourseRating']}★)")
    elif row['CourseRating'] >= 4.0:
        reasons.append(f"Well rated ({row['CourseRating']}★)")
    return " · ".join(reasons) if reasons else "Recommended for your segment"


def recommend_new_user(courses: pd.DataFrame,
                       goal: str,
                       category: str,
                       level: str,
                       top_n: int = 8) -> pd.DataFrame:
    """
    Cold-start recommendations for a brand-new user with
    no transaction history. Based on 3 onboarding answers:
      goal     : 'Career' / 'Skill' / 'Hobby'   (kept for context/logging)
      category : one of utils.CATEGORIES
      level    : one of utils.LEVELS
    Ranks purely by course rating within the chosen category/level.
    """
    candidates = courses[
        (courses['CourseCategory'] == category) &
        (courses['CourseLevel']    == level)
    ].copy()

    if len(candidates) < top_n:
        candidates = courses[courses['CourseCategory'] == category].copy()

    candidates = (candidates
                  .sort_values('CourseRating', ascending=False)
                  .head(top_n)
                  .reset_index(drop=True))

    candidates['Rank']        = candidates.index + 1
    candidates['MatchReason'] = 'Top rated in your chosen category'
    candidates['OnboardingGoal'] = goal

    return candidates[[
        'Rank', 'CourseID', 'CourseCategory', 'CourseLevel',
        'CourseType', 'CourseRating', 'MatchReason', 'OnboardingGoal'
    ]]


def evaluate_recommendation_quality(profiles: pd.DataFrame,
                                    courses: pd.DataFrame,
                                    transactions: pd.DataFrame,
                                    sample_size: int = 100,
                                    seed: int = 42) -> pd.DataFrame:
    """
    Samples `sample_size` users, generates recommendations for
    each, and returns a DataFrame of per-user quality metrics
    (avg score, avg rating, category/level match rate, popularity).
    Useful for tracking recommendation health over time.
    """
    sample_users = profiles['UserID'].sample(
        min(sample_size, len(profiles)), random_state=seed
    )

    rows = []
    for uid in sample_users:
        recs = recommend_courses(uid, profiles, courses, transactions, top_n=5)
        if recs.empty:
            continue
        rows.append({
            'UserID':      uid,
            'AvgScore':    recs['RecommendationScore'].mean(),
            'AvgRating':   recs['CourseRating'].mean(),
            'AvgCatMatch': recs['cat_score'].mean(),
            'AvgLvlMatch': recs['lvl_score'].mean(),
            'AvgPop':      recs['pop_score'].mean(),
        })

    n_sampled  = len(sample_users)
    n_covered  = len(rows)
    coverage   = n_covered / n_sampled if n_sampled else 0.0

    if coverage < 0.5:
        warnings.warn(
            f"evaluate_recommendation_quality: only {n_covered}/{n_sampled} "
            f"sampled users ({coverage:.1%}) received non-empty recommendations. "
            f"Quality metrics may be unreliable — check for over-restrictive "
            f"filter_category/filter_level arguments or an empty candidate pool.",
            UserWarning,
            stacklevel=2,
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == '__main__':
    from data_pipeline import generate_data
    from features import engineer_features
    from clustering import run_clustering, attach_cluster_labels

    print("Building pipeline for smoke test...")
    users, courses, transactions = generate_data(n_users=300, n_courses=100)
    profiles = engineer_features(users, courses, transactions)
    cr = run_clustering(profiles)
    profiles = attach_cluster_labels(profiles, cr)

    test_uid = profiles['UserID'].iloc[0]
    recs = recommend_courses(test_uid, profiles, courses, transactions, top_n=5)
    print(f"\nRecommendations for {test_uid}:")
    print(recs[['Rank', 'CourseID', 'CourseCategory', 'CourseLevel',
                'RecommendationScore', 'MatchReason']].to_string(index=False))

    cold = recommend_new_user(courses, goal='Career', category='Technology', level='Beginner', top_n=5)
    print(f"\nCold-start recommendations:")
    print(cold[['Rank', 'CourseID', 'CourseCategory', 'CourseRating']].to_string(index=False))

    print("recommender.py self-test passed ✅")