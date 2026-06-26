"""
EduPro Learner Intelligence Platform
File   : generate.py
Place  : data/raw/generate.py
Run    : python data/raw/generate.py
Output : data/raw/users.csv
         data/raw/courses.csv
         data/raw/transactions.csv
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# SETTINGS — change these if needed
# ─────────────────────────────────────────────

N_USERS    = 2000
N_COURSES  = 400
SEED       = 42
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2025, 12, 31)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CATEGORIES = [
    'Technology',
    'Data Science',
    'Business',
    'Design',
    'Marketing',
    'Finance',
    'Health & Wellness',
    'Arts & Creativity'
]

LEVELS      = ['Beginner', 'Intermediate', 'Advanced']
TYPES       = ['Video', 'Live', 'Hybrid', 'Self-Paced']
GENDERS     = ['Male', 'Female', 'Non-Binary']

# ─────────────────────────────────────────────
# SET RANDOM SEED (makes results reproducible)
# ─────────────────────────────────────────────

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

print("=" * 50)
print("   EduPro — Synthetic Data Generator")
print("=" * 50)


# ─────────────────────────────────────────────
# STEP 1 — GENERATE USERS
# ─────────────────────────────────────────────

print("\n[1/4] Generating users...")

users = pd.DataFrame({
    'UserID': [f'U{i:04d}' for i in range(1, N_USERS + 1)],
    'Age'   : rng.integers(18, 65, N_USERS),
    'Gender': rng.choice(
                  GENDERS,
                  N_USERS,
                  p=[0.48, 0.46, 0.06]   # Male / Female / Non-Binary
              )
})

print(f"   ✅ {len(users):,} users created")
print(f"   Age range   : {users['Age'].min()} – {users['Age'].max()}")
print(f"   Gender split: {users['Gender'].value_counts().to_dict()}")


# ─────────────────────────────────────────────
# STEP 2 — GENERATE COURSES
# ─────────────────────────────────────────────

print("\n[2/4] Generating courses...")

# Category weights — Technology & Data Science more common
cat_weights = [0.22, 0.18, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07]

# Level weights — more beginner than advanced
lvl_weights = [0.45, 0.35, 0.20]

courses = pd.DataFrame({
    'CourseID'      : [f'C{i:04d}' for i in range(1, N_COURSES + 1)],
    'CourseName'    : [f'Course {i:04d}' for i in range(1, N_COURSES + 1)],
    'CourseCategory': rng.choice(CATEGORIES, N_COURSES, p=cat_weights),
    'CourseType'    : rng.choice(TYPES, N_COURSES, p=[0.40, 0.20, 0.20, 0.20]),
    'CourseLevel'   : rng.choice(LEVELS,  N_COURSES, p=lvl_weights),
    'CourseRating'  : np.clip(
                          rng.normal(4.1, 0.5, N_COURSES), 1.0, 5.0
                      ).round(1),
    'CourseDuration': rng.integers(2, 40, N_COURSES),   # hours
})

# Price depends on level — advanced costs more
price_range = {
    'Beginner'    : (29,  99),
    'Intermediate': (59, 149),
    'Advanced'    : (99, 299),
}
courses['BasePrice'] = courses['CourseLevel'].map(
    lambda lvl: round(rng.uniform(*price_range[lvl]), 2)
)

print(f"   ✅ {len(courses):,} courses created")
print(f"   Category split:")
for cat, cnt in courses['CourseCategory'].value_counts().items():
    print(f"      {cat:<25} : {cnt}")


# ─────────────────────────────────────────────
# STEP 3 — GENERATE TRANSACTIONS
# ─────────────────────────────────────────────

print("\n[3/4] Generating transactions...")

# 4 Learner Archetypes — each user gets one dominant type
# Archetype 0 — Tech Explorer   : many courses, tech heavy
# Archetype 1 — Career Climber  : business/finance, mid-high spend
# Archetype 2 — Deep Specialist : few courses, advanced, high spend
# Archetype 3 — Casual Browser  : few courses, random, low spend

archetype_probs = rng.dirichlet(alpha=[2, 2, 2, 2], size=N_USERS)
user_archetype  = archetype_probs.argmax(axis=1)

# Enrollment count per archetype
enroll_dist = {
    0: (22, 5),   # Tech Explorer   — many
    1: (18, 4),   # Career Climber  — moderate
    2: (10, 3),   # Deep Specialist — few but focused
    3: ( 5, 2),   # Casual Browser  — minimal
}

# Category preference per archetype
cat_pref = {
    0: [0.35, 0.30, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05],  # Tech heavy
    1: [0.10, 0.08, 0.30, 0.05, 0.15, 0.25, 0.04, 0.03],  # Business/Finance
    2: [0.30, 0.40, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02],  # DS/Tech deep
    3: [0.13, 0.12, 0.13, 0.12, 0.12, 0.12, 0.13, 0.13],  # Random/uniform
}

# Level preference per archetype
lvl_pref = {
    0: [0.50, 0.35, 0.15],   # Explorer  — mostly beginner
    1: [0.25, 0.55, 0.20],   # Climber   — intermediate focus
    2: [0.10, 0.30, 0.60],   # Specialist— advanced focus
    3: [0.60, 0.30, 0.10],   # Browser   — beginner heavy
}

date_range_days = (END_DATE - START_DATE).days
records = []

for idx, uid in enumerate(users['UserID']):
    arch = user_archetype[idx]

    # Number of courses this user will enroll in
    mu, sd    = enroll_dist[arch]
    n_enroll  = max(1, int(rng.normal(mu, sd)))

    # Pick categories based on archetype preference
    chosen_cats = rng.choice(CATEGORIES, n_enroll, p=cat_pref[arch])
    chosen_lvls = rng.choice(LEVELS,     n_enroll, p=lvl_pref[arch])

    chosen_course_ids = []
    for cat, lvl in zip(chosen_cats, chosen_lvls):

        # Find courses matching category + level
        pool = courses[
            (courses['CourseCategory'] == cat) &
            (courses['CourseLevel']    == lvl)
        ]['CourseID'].values

        # Fallback: just match category
        if len(pool) == 0:
            pool = courses[courses['CourseCategory'] == cat]['CourseID'].values

        # Fallback: any course
        if len(pool) == 0:
            pool = courses['CourseID'].values

        chosen_course_ids.append(rng.choice(pool))

    # Remove duplicate courses (user can't enroll twice)
    chosen_course_ids = list(dict.fromkeys(chosen_course_ids))

    # Random transaction dates within date range, sorted
    tx_days = sorted(rng.integers(0, date_range_days, len(chosen_course_ids)))
    tx_dates = [START_DATE + timedelta(days=int(d)) for d in tx_days]

    for cid, txdate in zip(chosen_course_ids, tx_dates):
        base_price = courses.loc[
            courses['CourseID'] == cid, 'BasePrice'
        ].values[0]

        # Small price variation ±15%
        final_price = round(base_price * rng.uniform(0.85, 1.15), 2)

        records.append({
            'UserID'         : uid,
            'CourseID'       : cid,
            'TransactionDate': txdate.strftime('%Y-%m-%d'),
            'Amount'         : final_price
        })

transactions = pd.DataFrame(records)

print(f"   ✅ {len(transactions):,} transactions created")
print(f"   Date range  : {transactions['TransactionDate'].min()} → {transactions['TransactionDate'].max()}")
print(f"   Total Revenue: ${transactions['Amount'].sum():,.2f}")
print(f"   Avg per User : ${transactions.groupby('UserID')['Amount'].sum().mean():,.2f}")
print(f"   Archetype split:")
archetype_names = {
    0: 'Tech Explorer',
    1: 'Career Climber',
    2: 'Deep Specialist',
    3: 'Casual Browser'
}
for k, v in pd.Series(user_archetype).value_counts().sort_index().items():
    print(f"      {archetype_names[k]:<20}: {v} users")


# ─────────────────────────────────────────────
# STEP 4 — SAVE TO CSV
# ─────────────────────────────────────────────

print("\n[4/4] Saving files...")

# Make sure output folder exists
os.makedirs('data/raw', exist_ok=True)

users.to_csv('data/raw/users.csv',               index=False)
courses.to_csv('data/raw/courses.csv',           index=False)
transactions.to_csv('data/raw/transactions.csv', index=False)

print(f"   ✅ data/raw/users.csv        → {len(users):,} rows")
print(f"   ✅ data/raw/courses.csv      → {len(courses):,} rows")
print(f"   ✅ data/raw/transactions.csv → {len(transactions):,} rows")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("   DATA GENERATION COMPLETE")
print("=" * 50)
print(f"""
   Users        : {len(users):,}
   Courses      : {len(courses):,}
   Transactions : {len(transactions):,}
   Revenue      : ${transactions['Amount'].sum():,.2f}
   Date Range   : 2022-01-01 → 2025-12-31

   Files saved to data/raw/
   Next step → run notebooks/01_eda.ipynb
""")