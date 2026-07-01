"""
EduPro Learner Intelligence Platform
File: src/data_pipeline.py

Handles synthetic data generation AND loading existing CSVs.
This is the entry point for getting users / courses / transactions
into memory as pandas DataFrames.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .utils import (
    CATEGORIES, LEVELS, COURSE_TYPES, GENDERS,
    N_USERS, N_COURSES, RANDOM_SEED
)


# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_data(n_users: int = N_USERS,
                  n_courses: int = N_COURSES,
                  seed: int = RANDOM_SEED) -> tuple:
    """
    Generates synthetic EdUPro professional users, CPD courses, and enrollment records.
    Returns (users_df, courses_df, transactions_df).

    4 latent learner archetypes drive enrollment behaviour:
      0 = Eager Learner   (many courses, broad, beginner-intermediate)
      1 = CPD Collector   (mid-level, accreditation-driven, varied)
      2 = Clinical Expert (few courses, advanced, specialty-focused)
      3 = Curious Browser (low engagement, classroom/parent type)
    """
    from .utils import PROFESSION_TYPES
    rng = np.random.default_rng(seed)

    # ── Users (EdUPro professionals) ─────────
    users = pd.DataFrame({
        'UserID':          [f'P{i:04d}' for i in range(1, n_users + 1)],
        'Age':             rng.integers(22, 62, n_users),
        'Gender':          rng.choice(GENDERS, n_users, p=[0.72, 0.25, 0.03]),
        'ProfessionType':  rng.choice(
            PROFESSION_TYPES, n_users,
            p=[0.28, 0.22, 0.20, 0.18, 0.08, 0.04]
        ),
        'YearsExperience': rng.integers(0, 35, n_users),
        'Country':         rng.choice(
            ['South Africa', 'United Kingdom', 'Australia',
             'New Zealand', 'Netherlands', 'Other'],
            n_users, p=[0.55, 0.15, 0.12, 0.08, 0.05, 0.05]
        ),
    })

    # ── Courses (EdUPro CPD catalogue) ───────
    # 6 EdUPro categories — weights must sum to 1 and match len(CATEGORIES)=6
    cat_weights = [0.20, 0.18, 0.18, 0.17, 0.14, 0.13]
    cpd_points  = {'Beginner': 2, 'Intermediate': 4, 'Advanced': 6}
    price_zar   = {'Beginner': (299, 599), 'Intermediate': (499, 999), 'Advanced': (799, 1499)}

    courses = pd.DataFrame({
        'CourseID':          [f'C{i:04d}' for i in range(1, n_courses + 1)],
        'CourseCategory':    rng.choice(CATEGORIES, n_courses, p=cat_weights),
        'CourseType':        rng.choice(COURSE_TYPES, n_courses, p=[0.35, 0.30, 0.20, 0.15]),
        'CourseLevel':       rng.choice(LEVELS, n_courses, p=[0.45, 0.35, 0.20]),
        'CourseRating':      np.clip(rng.normal(4.3, 0.4, n_courses), 1.0, 5.0).round(1),
    })
    courses['PriceZAR']  = courses['CourseLevel'].map(
        lambda lvl: round(rng.uniform(*price_zar[lvl]), 2)
    )
    courses['CPDPoints'] = courses['CourseLevel'].map(cpd_points)

    # ── Enrollments ───────────────────────────
    archetype_probs = rng.dirichlet(alpha=[2, 2, 2, 2], size=n_users)
    user_archetype  = archetype_probs.argmax(axis=1)

    # 0=Eager Learner, 1=CPD Collector, 2=Clinical Expert, 3=Curious Browser
    enroll_dist = {0: (22, 5), 1: (16, 4), 2: (8, 3), 3: (4, 2)}

    # Category preference — 6 values matching CATEGORIES order
    cat_pref = {
        0: [0.25, 0.25, 0.15, 0.15, 0.10, 0.10],   # broad
        1: [0.20, 0.15, 0.20, 0.15, 0.10, 0.20],   # CPD-driven, varied
        2: [0.40, 0.15, 0.05, 0.20, 0.05, 0.15],   # DCD + Assessment heavy
        3: [0.15, 0.15, 0.35, 0.15, 0.15, 0.05],   # Classroom heavy
    }
    lvl_pref = {
        0: [0.55, 0.35, 0.10],
        1: [0.25, 0.55, 0.20],
        2: [0.05, 0.25, 0.70],
        3: [0.65, 0.30, 0.05],
    }
    completion_rate = {0: 0.75, 1: 0.80, 2: 0.90, 3: 0.45}

    start_date      = datetime(2022, 1, 1)
    end_date        = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    records = []
    for idx, uid in enumerate(users['UserID']):
        arch = user_archetype[idx]
        mu, sd   = enroll_dist[arch]
        n_enroll = max(1, int(rng.normal(mu, sd)))

        # Over-sample 3× before dedup to preserve intended count in small pools
        n_candidates = min(n_enroll * 3, len(courses))
        chosen_cats  = rng.choice(CATEGORIES, n_candidates, p=cat_pref[arch])
        chosen_lvls  = rng.choice(LEVELS,     n_candidates, p=lvl_pref[arch])

        chosen_ids = []
        for cat, lvl in zip(chosen_cats, chosen_lvls):
            pool = courses[(courses['CourseCategory'] == cat) &
                           (courses['CourseLevel']    == lvl)]['CourseID'].values
            if len(pool) == 0:
                pool = courses[courses['CourseCategory'] == cat]['CourseID'].values
            if len(pool) == 0:
                pool = courses['CourseID'].values
            chosen_ids.append(rng.choice(pool))

        chosen_ids = list(dict.fromkeys(chosen_ids))[:n_enroll]

        tx_days  = sorted(rng.integers(0, date_range_days, len(chosen_ids)))
        tx_dates = [start_date + timedelta(days=int(d)) for d in tx_days]

        for cid, txd in zip(chosen_ids, tx_dates):
            price   = courses.loc[courses['CourseID'] == cid, 'PriceZAR'].values[0]
            cpd_pts = courses.loc[courses['CourseID'] == cid, 'CPDPoints'].values[0]
            amount  = round(price * rng.uniform(0.90, 1.10), 2)
            done    = bool(rng.random() < completion_rate[arch])
            records.append({
                'UserID':            uid,
                'CourseID':          cid,
                'EnrollmentDate':    txd.strftime('%Y-%m-%d'),
                'Amount':            amount,
                'CompletionStatus':  'Completed' if done else 'In Progress',
                'CPDPointsEarned':   cpd_pts if done else 0,
                'CertificateIssued': done,
            })

    transactions = pd.DataFrame(records)
    return users, courses, transactions


# ─────────────────────────────────────────────
# 2. LOAD EXISTING CSVs (if already generated)
# ─────────────────────────────────────────────

def load_raw_data(data_dir: str = 'data/raw') -> tuple:
    """
    Loads users.csv, courses.csv, transactions.csv from disk.
    Use this once generate_data() has already been run and saved.
    Raises FileNotFoundError with a clear message if files are missing.
    """
    paths = {
        'users':        os.path.join(data_dir, 'users.csv'),
        'courses':      os.path.join(data_dir, 'courses.csv'),
        'transactions': os.path.join(data_dir, 'transactions.csv'),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {name} file at '{path}'. "
                f"Run generate_data() and save_raw_data() first."
            )

    users        = pd.read_csv(paths['users'])
    courses      = pd.read_csv(paths['courses'])
    transactions = pd.read_csv(paths['transactions'])

    # Support both legacy ('TransactionDate') and EdUPro v2 ('EnrollmentDate') schemas.
    if 'TransactionDate' in transactions.columns:
        transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
    elif 'EnrollmentDate' in transactions.columns:
        transactions['EnrollmentDate'] = pd.to_datetime(transactions['EnrollmentDate'])

    return users, courses, transactions


def save_raw_data(users: pd.DataFrame,
                  courses: pd.DataFrame,
                  transactions: pd.DataFrame,
                  data_dir: str = 'data/raw') -> None:
    """Saves the three raw tables to CSV under data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    users.to_csv(os.path.join(data_dir, 'users.csv'), index=False)
    courses.to_csv(os.path.join(data_dir, 'courses.csv'), index=False)
    transactions.to_csv(os.path.join(data_dir, 'transactions.csv'), index=False)


# ─────────────────────────────────────────────
# 3. SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Generating synthetic data...")
    users, courses, transactions = generate_data()
    print(f"  Users        : {len(users):,}")
    print(f"  Courses      : {len(courses):,}")
    print(f"  Transactions : {len(transactions):,}")
    print(f"  Revenue      : ${transactions['Amount'].sum():,.2f}")
    print("data_pipeline.py self-test passed ✅")