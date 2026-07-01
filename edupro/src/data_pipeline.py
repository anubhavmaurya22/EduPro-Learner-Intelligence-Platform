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
    Generates synthetic users, courses, and transactions.
    Returns (users_df, courses_df, transactions_df).

    4 latent learner archetypes drive enrollment behaviour
    so that clustering later recovers meaningful segments:
      0 = Tech Explorer    (many courses, tech-heavy)
      1 = Career Climber   (business/finance, high spend)
      2 = Deep Specialist  (few courses, advanced)
      3 = Casual Browser   (few courses, random, low spend)
    """
    rng = np.random.default_rng(seed)

    # ── Users ────────────────────────────────
    users = pd.DataFrame({
        'UserID': [f'U{i:04d}' for i in range(1, n_users + 1)],
        'Age':    rng.integers(18, 65, n_users),
        'Gender': rng.choice(GENDERS, n_users, p=[0.48, 0.46, 0.06])
    })

    # ── Courses ──────────────────────────────
    cat_weights = [0.22, 0.18, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07]
    courses = pd.DataFrame({
        'CourseID':       [f'C{i:04d}' for i in range(1, n_courses + 1)],
        'CourseCategory': rng.choice(CATEGORIES, n_courses, p=cat_weights),
        'CourseType':     rng.choice(COURSE_TYPES, n_courses, p=[0.40, 0.20, 0.20, 0.20]),
        'CourseLevel':    rng.choice(LEVELS, n_courses, p=[0.45, 0.35, 0.20]),
        'CourseRating':   np.clip(rng.normal(4.1, 0.5, n_courses), 1.0, 5.0).round(1)
    })
    price_map = {'Beginner': (29, 99), 'Intermediate': (59, 149), 'Advanced': (99, 299)}
    courses['BasePrice'] = courses['CourseLevel'].map(
        lambda lvl: round(rng.uniform(*price_map[lvl]), 2)
    )

    # ── Transactions ─────────────────────────
    archetype_probs = rng.dirichlet(alpha=[2, 2, 2, 2], size=n_users)
    user_archetype  = archetype_probs.argmax(axis=1)

    enroll_dist = {0: (22, 5), 1: (18, 4), 2: (10, 3), 3: (5, 2)}

    cat_pref = {
        0: [0.35, 0.30, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05],
        1: [0.10, 0.08, 0.30, 0.05, 0.15, 0.25, 0.04, 0.03],
        2: [0.30, 0.40, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02],
        3: [0.13, 0.12, 0.13, 0.12, 0.12, 0.12, 0.13, 0.13],
    }
    lvl_pref = {
        0: [0.50, 0.35, 0.15],
        1: [0.25, 0.55, 0.20],
        2: [0.10, 0.30, 0.60],
        3: [0.60, 0.30, 0.10],
    }

    start_date      = datetime(2022, 1, 1)
    end_date        = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    records = []
    for idx, uid in enumerate(users['UserID']):
        arch = user_archetype[idx]
        mu, sd = enroll_dist[arch]
        n_enroll = max(1, int(rng.normal(mu, sd)))

        # Over-sample by 3× so that after dedup we still have (roughly)
        # n_enroll unique courses even in a small course pool.  Collisions
        # are especially likely for high-activity archetypes (n~22) when
        # n_courses=100, so generating extra candidates before dedup keeps
        # total_courses and enrollment_frequency representative of the
        # intended archetype, preserving cluster separability.
        n_candidates = min(n_enroll * 3, len(courses))
        chosen_cats = rng.choice(CATEGORIES, n_candidates, p=cat_pref[arch])
        chosen_lvls = rng.choice(LEVELS,     n_candidates, p=lvl_pref[arch])

        chosen_ids = []
        for cat, lvl in zip(chosen_cats, chosen_lvls):
            pool = courses[(courses['CourseCategory'] == cat) &
                           (courses['CourseLevel']    == lvl)]['CourseID'].values
            if len(pool) == 0:
                pool = courses[courses['CourseCategory'] == cat]['CourseID'].values
            if len(pool) == 0:
                pool = courses['CourseID'].values
            chosen_ids.append(rng.choice(pool))

        # De-dupe (preserving archetype-driven order), then truncate to
        # the originally intended enrollment count.
        chosen_ids = list(dict.fromkeys(chosen_ids))[:n_enroll]

        tx_days  = sorted(rng.integers(0, date_range_days, len(chosen_ids)))
        tx_dates = [start_date + timedelta(days=int(d)) for d in tx_days]

        for cid, txd in zip(chosen_ids, tx_dates):
            base = courses.loc[courses['CourseID'] == cid, 'BasePrice'].values[0]
            amount = round(base * rng.uniform(0.85, 1.15), 2)
            records.append({
                'UserID':          uid,
                'CourseID':        cid,
                'TransactionDate': txd.strftime('%Y-%m-%d'),
                'Amount':          amount
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