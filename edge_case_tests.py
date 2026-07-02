"""
EdUPro Edge Case Tests
Run: python edge_case_tests.py
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from src.data_pipeline import generate_data
from src.features import engineer_features
from src.clustering import run_clustering, attach_cluster_labels
from src.recommender import recommend_courses, recommend_new_user
from src.utils import CATEGORIES, LEVELS

print("Edge Case Tests")
print("=" * 50)
passed = failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1; print(f"  PASS  {name}")
    else:
        failed += 1; print(f"  FAIL  {name}")

# Build minimal pipeline
users, courses, transactions = generate_data(n_users=300, n_courses=100)
profiles = engineer_features(users, courses, transactions)
cr = run_clustering(profiles)
profiles = attach_cluster_labels(profiles, cr)

print("\n-- User Edge Cases --")
# Unknown user
recs = recommend_courses("P9999_FAKE", profiles, courses, transactions)
check("Unknown user returns empty", recs.empty)

# User with only 1 course
single_course_user = (
    transactions.groupby('UserID')
    .filter(lambda x: len(x) == 1)
    ['UserID'].iloc[0]
    if len(transactions.groupby('UserID').filter(
        lambda x: len(x) == 1)) > 0
    else profiles['UserID'].iloc[0]
)
recs = recommend_courses(
    single_course_user, profiles, courses, transactions, top_n=5
)
check("Single-course user gets recs", len(recs) > 0)

print("\n-- Filter Edge Cases --")
# Filter that returns results
uid = profiles['UserID'].iloc[0]
recs_cat = recommend_courses(
    uid, profiles, courses, transactions,
    filter_category=CATEGORIES[0]
)
check("Category filter works",
      recs_cat.empty or
      (recs_cat['CourseCategory'] == CATEGORIES[0]).all())

recs_lvl = recommend_courses(
    uid, profiles, courses, transactions,
    filter_level='Advanced'
)
check("Level filter works",
      recs_lvl.empty or
      (recs_lvl['CourseLevel'] == 'Advanced').all())

# Both filters together
recs_both = recommend_courses(
    uid, profiles, courses, transactions,
    filter_category=CATEGORIES[0], filter_level='Beginner'
)
check("Combined filter no crash", True)

print("\n-- Cold Start Edge Cases --")
for cat in CATEGORIES:
    recs = recommend_new_user(
        courses, 'Career', cat, 'Beginner', top_n=3
    )
    check(f"Cold start {cat[:20]}", len(recs) > 0)

for lvl in LEVELS:
    recs = recommend_new_user(
        courses, 'Skill', CATEGORIES[0], lvl, top_n=3
    )
    check(f"Cold start {lvl}", len(recs) > 0)

print("\n-- Score Integrity --")
uid2 = profiles['UserID'].iloc[10]
recs2 = recommend_courses(uid2, profiles, courses, transactions, top_n=8)
if not recs2.empty:
    check("Scores are floats",
          recs2['RecommendationScore'].dtype in ['float64', 'float32'])
    check("Scores ascending rank",
          recs2['RecommendationScore'].tolist() ==
          sorted(recs2['RecommendationScore'].tolist(), reverse=True))
    check("Rank column sequential",
          recs2['Rank'].tolist() == list(range(1, len(recs2)+1)))
    check("MatchReason not empty",
          (recs2['MatchReason'].str.len() > 0).all())

print()
print("=" * 50)
print(f"RESULTS: {passed} passed | {failed} failed")
if failed == 0:
    print("All edge cases handled correctly ✅")
