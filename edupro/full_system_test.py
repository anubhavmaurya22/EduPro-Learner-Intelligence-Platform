"""
EdUPro Full System Test
Run: python full_system_test.py
Tests everything from data generation to recommendations
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

print("=" * 60)
print("   EdUPro Full System Test")
print("=" * 60)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} {detail}")

# ── STEP 1: DATA GENERATION ────────────────────
print("\n[1/6] Data Generation...")
from src.data_pipeline import generate_data
users, courses, transactions = generate_data(
    n_users=500, n_courses=200
)
check("Users generated",        len(users) == 500)
check("Courses generated",      len(courses) == 200)
check("Transactions generated", len(transactions) > 0)
check("No null users",          users.isnull().sum().sum() == 0)
check("No null transactions",   transactions.isnull().sum().sum() == 0)
check("Amounts positive",       (transactions['Amount'] > 0).all())

# Check EdUPro-specific columns
if 'ProfessionType' in users.columns:
    check("ProfessionType present", True)
    check("Valid profession types", users['ProfessionType'].notna().all())
else:
    check("ProfessionType present", False, "← update generate.py")

# ── STEP 2: FEATURE ENGINEERING ────────────────
print("\n[2/6] Feature Engineering...")
from src.features import engineer_features
profiles = engineer_features(users, courses, transactions)

check("One row per user",    len(profiles) == transactions['UserID'].nunique())
check("No nulls in profiles",profiles.isnull().sum().sum() == 0)
check("Depth index 0-1",     (profiles['learning_depth_index'] >= 0).all() and
                             (profiles['learning_depth_index'] <= 1).all())
check("Positive courses",    (profiles['total_courses'] > 0).all())
check("12+ features exist",  len([c for c in profiles.columns
                             if c in [
                                 'total_courses','avg_spending',
                                 'learning_depth_index','diversity_score',
                                 'enrollment_frequency','recency_days'
                             ]]) == 6)

# ── STEP 3: CLUSTERING ─────────────────────────
print("\n[3/6] Clustering...")
from src.clustering import run_clustering, get_cluster_profiles, attach_cluster_labels

cr = run_clustering(profiles)

check("No NaN labels",       pd.isna(cr['labels']).sum() == 0)
check("Labels 0-3 only",     set(cr['labels']).issubset({0,1,2,3}))
check("4 segments found",    len(set(cr['labels'])) == 4)
check("Positive silhouette", cr['overall_silhouette'] > 0)
check("PCA shape correct",   cr['X_pca'].shape == (len(profiles), 2))
check("Scaler saved",        cr['scaler'] is not None)

profiles = attach_cluster_labels(profiles, cr)
cp = get_cluster_profiles(profiles, cr)
check("Profiles sum to total",cp['n_users'].sum() == len(profiles))
check("All 4 segment names", len(cp['SegmentName'].unique()) == 4)

print(f"\n  Segment Summary:")
for _, row in cp.sort_values('Cluster').iterrows():
    print(f"    {row['SegmentName']:<25} : "
          f"{int(row['n_users']):>4} users | "
          f"depth={row['learning_depth_index']:.2f} | "
          f"spend=R{row['avg_spending']:.0f}")

# ── STEP 4: RECOMMENDATIONS ────────────────────
print("\n[4/6] Recommendation Engine...")
from src.recommender import (
    recommend_courses, recommend_new_user,
    evaluate_recommendation_quality
)

# Test for 4 users from different segments
for seg_id in range(4):
    seg_users = profiles[profiles['Cluster'] == seg_id]['UserID']
    if len(seg_users) == 0:
        continue
    uid  = seg_users.iloc[0]
    recs = recommend_courses(uid, profiles, courses, transactions, top_n=5)
    check(f"Segment {seg_id} recs returned",  len(recs) > 0)
    check(f"Segment {seg_id} not taken",
          len(set(recs['CourseID']) &
              set(transactions[transactions['UserID']==uid]['CourseID'])) == 0)
    check(f"Segment {seg_id} scores 0-1",
          (recs['RecommendationScore'] >= 0).all() and
          (recs['RecommendationScore'] <= 1).all())

# Cold start test
from src.utils import CATEGORIES
cold = recommend_new_user(
    courses, 'Career',
    'DCD & Motor Disorders', 'Advanced', top_n=5
)
check("Cold start works",      len(cold) > 0)
check("Cold start sorted",     cold['CourseRating'].tolist() ==
                               sorted(cold['CourseRating'].tolist(), reverse=True))

# ── STEP 5: MODEL SAVING ───────────────────────
print("\n[5/6] Model Saving...")
import joblib
import os

os.makedirs('models', exist_ok=True)
joblib.dump(cr['km_model'], 'models/kmeans_model.pkl')
joblib.dump(cr['scaler'],   'models/scaler.pkl')
joblib.dump(cr['pca'],      'models/pca_model.pkl')

check("KMeans saved",   os.path.exists('models/kmeans_model.pkl'))
check("Scaler saved",   os.path.exists('models/scaler.pkl'))
check("PCA saved",      os.path.exists('models/pca_model.pkl'))

km_loaded = joblib.load('models/kmeans_model.pkl')
check("Model reloads OK", km_loaded is not None)

# ── STEP 6: DATA SAVING ────────────────────────
print("\n[6/6] Data Output Files...")
import os

os.makedirs('data/outputs', exist_ok=True)
profiles.to_csv('data/outputs/cluster_labels.csv', index=False)
cp.to_csv('data/outputs/segment_summary.csv', index=False)

check("cluster_labels.csv exists",  os.path.exists('data/outputs/cluster_labels.csv'))
check("segment_summary.csv exists", os.path.exists('data/outputs/segment_summary.csv'))

# Verify file contents
loaded = pd.read_csv('data/outputs/cluster_labels.csv')
check("cluster_labels has Cluster col", 'Cluster' in loaded.columns)
check("cluster_labels has SegmentName", 'SegmentName' in loaded.columns)

# ── FINAL REPORT ───────────────────────────────
print()
print("=" * 60)
print(f"   RESULTS: {passed} passed  |  {failed} failed")
print("=" * 60)
print()

if failed == 0:
    print("  ✅ ALL TESTS PASSED")
    print("  System is ready for dashboard and presentation")
else:
    print(f"  ⚠️  {failed} issues found — fix before presentation")

print(f"""
  Pipeline Summary:
    Users         : {len(users):,}
    Courses       : {len(courses):,}
    Transactions  : {len(transactions):,}
    Silhouette    : {cr['overall_silhouette']:.4f}
    Segments      : {len(set(cr['labels']))}
    NaN Labels    : {pd.isna(cr['labels']).sum()}
""")
