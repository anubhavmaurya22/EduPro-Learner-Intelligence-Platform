
"""
EduPro Learner Intelligence Platform
Real Context: edupro-online.com
CPD Platform for Pediatric Professionals
South Africa
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

N_USERS   = 2000
N_COURSES = 400
SEED      = 42

START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2025, 12, 31)

# ─────────────────────────────────────────────
# EDUPRO-SPECIFIC CONSTANTS
# ─────────────────────────────────────────────

PROFESSION_TYPES = [
    'Occupational Therapist',
    'Physiotherapist',
    'Kinderkineticist',
    'Classroom Teacher',
    'School Psychologist',
    'Parent/Caregiver'
]

COURSE_CATEGORIES = [
    'DCD & Motor Disorders',
    'Fundamental Movement Skills',
    'Classroom Interventions',
    'Sensory Processing',
    'Sport & Leisure',
    'Assessment & Diagnosis'
]

LEVELS       = ['Beginner', 'Intermediate', 'Advanced']
COURSE_TYPES = ['Webinar', 'Self-Paced Video', 'Live Workshop', 'Mini Course']
GENDERS      = ['Female', 'Male', 'Non-Binary']

ACCREDITATION_BODIES = ['HPCSA', 'SAPIK', 'SASHP', 'None']

# CPD points awarded per level
CPD_POINTS = {
    'Beginner':     2,
    'Intermediate': 4,
    'Advanced':     6
}

# Price in ZAR per level
PRICE_RANGE_ZAR = {
    'Beginner':     (299,  599),
    'Intermediate': (499,  999),
    'Advanced':     (799, 1499),
}

# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

print("=" * 55)
print("   EdUPro — CPD Professional Data Generator")
print("   Context: edupro-online.com")
print("=" * 55)


# ─────────────────────────────────────────────
# STEP 1 — GENERATE USERS (PROFESSIONALS)
# ─────────────────────────────────────────────

print("\n[1/4] Generating professional profiles...")

users = pd.DataFrame({
    'UserID': [f'P{i:04d}' for i in range(1, N_USERS + 1)],
    'Age':    rng.integers(22, 62, N_USERS),
    'Gender': rng.choice(GENDERS, N_USERS, p=[0.72, 0.25, 0.03]),

    # Profession distribution — OTs and physios are largest group
    'ProfessionType': rng.choice(
        PROFESSION_TYPES, N_USERS,
        p=[0.28, 0.22, 0.20, 0.18, 0.08, 0.04]
    ),

    # Years of experience
    'YearsExperience': rng.integers(0, 35, N_USERS),

    # Country (EdUPro is South African but serves internationally)
    'Country': rng.choice(
        ['South Africa', 'United Kingdom', 'Australia',
         'New Zealand', 'Netherlands', 'Other'],
        N_USERS, p=[0.55, 0.15, 0.12, 0.08, 0.05, 0.05]
    )
})

print(f"   ✅ {len(users):,} professionals created")
print(f"   Profession breakdown:")
for prof, cnt in users['ProfessionType'].value_counts().items():
    print(f"      {prof:<30} : {cnt}")


# ─────────────────────────────────────────────
# STEP 2 — GENERATE COURSES (CPD CONTENT)
# ─────────────────────────────────────────────

print("\n[2/4] Generating CPD course catalogue...")

# Course titles per category (realistic EdUPro style)
COURSE_TITLE_TEMPLATES = {
    'DCD & Motor Disorders': [
        'Diagnosing DCD in Children',
        'DCD Intervention Strategies',
        'Motor Coordination Assessment',
        'Supporting DCD in the Classroom',
        'DCD and Daily Life Activities',
        'Advanced DCD Case Studies',
        'Handwriting Difficulties in DCD',
        'Dressing and Self-Care for DCD Children',
    ],
    'Fundamental Movement Skills': [
        'Foundations of Movement Development',
        'Running and Jumping Skills in Early Childhood',
        'Stability: The Core in FMS',
        'Ball Skills for Young Children',
        'Movement Screening and Assessment',
        'Advanced FMS Programming',
    ],
    'Classroom Interventions': [
        'Sensory Strategies for the Classroom',
        'Movement Breaks That Work',
        'Supporting Motor Difficulties in School',
        'Desk and Seating Positioning',
        'Writing and Homework Strategies',
        'PE Modifications for Special Needs',
    ],
    'Sensory Processing': [
        'Introduction to Sensory Processing',
        'Sensory Diets for Children',
        'Tactile Sensitivity Interventions',
        'Proprioception and Body Awareness',
        'Vestibular Processing in Practice',
        'Advanced Sensory Integration',
    ],
    'Sport & Leisure': [
        'How to Present Movement Classes Like a Pro',
        'Adapted Sport for Children With DCD',
        'Swimming and Aquatic Therapy Basics',
        'Team Sports Participation Strategies',
        'Leisure and Play for Motor-Challenged Children',
    ],
    'Assessment & Diagnosis': [
        'Introduction to Pediatric Assessment',
        'Movement ABC-2 in Practice',
        'Beery VMI Scoring and Interpretation',
        'Writing a Professional Assessment Report',
        'Goal Setting With the ICF Framework',
        'Advanced Clinical Reasoning in Pediatrics',
    ]
}

course_records = []
course_id = 1
for category, titles in COURSE_TITLE_TEMPLATES.items():
    n_per_cat = N_COURSES // len(COURSE_CATEGORIES)
    for i in range(n_per_cat):
        title_base = titles[i % len(titles)]
        level      = rng.choice(LEVELS, p=[0.45, 0.35, 0.20])
        price      = round(rng.uniform(*PRICE_RANGE_ZAR[level]), 2)
        course_records.append({
            'CourseID':           f'C{course_id:04d}',
            'CourseTitle':        f"{title_base} ({level})",
            'CourseCategory':     category,
            'CourseType':         rng.choice(
                                      COURSE_TYPES,
                                      p=[0.35, 0.30, 0.20, 0.15]
                                  ),
            'CourseLevel':        level,
            'CourseRating':       round(
                                      float(np.clip(rng.normal(4.3, 0.4), 1, 5)), 1
                                  ),
            'PriceZAR':           price,
            'CPDPoints':          CPD_POINTS[level],
            'AccreditationBody':  rng.choice(
                                      ACCREDITATION_BODIES,
                                      p=[0.40, 0.25, 0.15, 0.20]
                                  ),
        })
        course_id += 1

courses = pd.DataFrame(course_records)

print(f"   ✅ {len(courses):,} courses created")
print(f"   Category breakdown:")
for cat, cnt in courses['CourseCategory'].value_counts().items():
    print(f"      {cat:<35} : {cnt}")


# ─────────────────────────────────────────────
# STEP 3 — GENERATE ENROLLMENTS
# ─────────────────────────────────────────────

print("\n[3/4] Generating CPD enrollment records...")

# 4 Learner Archetypes (same ML approach, EdUPro names)
# 0 = Eager Learner      : many courses, beginner-intermediate, broad
# 1 = CPD Collector      : mid-level, accreditation-driven, varied
# 2 = Clinical Expert    : few courses, advanced, focused specialty
# 3 = Curious Browser    : low engagement, teacher/parent type

archetype_probs = rng.dirichlet(alpha=[2, 2, 2, 2], size=N_USERS)
user_archetype  = archetype_probs.argmax(axis=1)

enroll_dist = {
    0: (22, 5),    # Eager Learner   — many
    1: (16, 4),    # CPD Collector   — moderate
    2: (8,  3),    # Clinical Expert — few but deep
    3: (4,  2),    # Curious Browser — minimal
}

# Category preferences per archetype
cat_pref = {
    0: [0.25, 0.25, 0.15, 0.15, 0.10, 0.10],  # broad
    1: [0.20, 0.15, 0.20, 0.15, 0.10, 0.20],  # CPD-driven, varied
    2: [0.40, 0.15, 0.05, 0.20, 0.05, 0.15],  # DCD + Assessment heavy
    3: [0.15, 0.15, 0.35, 0.15, 0.15, 0.05],  # classroom heavy
}

# Level preferences per archetype
lvl_pref = {
    0: [0.55, 0.35, 0.10],   # Eager       — mostly beginner
    1: [0.25, 0.55, 0.20],   # CPD Collect — intermediate
    2: [0.05, 0.25, 0.70],   # Expert      — advanced
    3: [0.65, 0.30, 0.05],   # Browser     — beginner heavy
}

date_range_days = (END_DATE - START_DATE).days
records = []

for idx, uid in enumerate(users['UserID']):
    arch = user_archetype[idx]
    mu, sd = enroll_dist[arch]
    n_enroll = max(1, int(rng.normal(mu, sd)))

    chosen_cats = rng.choice(
        COURSE_CATEGORIES, n_enroll, p=cat_pref[arch]
    )
    chosen_lvls = rng.choice(
        LEVELS, n_enroll, p=lvl_pref[arch]
    )

    chosen_ids = []
    for cat, lvl in zip(chosen_cats, chosen_lvls):
        pool = courses[
            (courses['CourseCategory'] == cat) &
            (courses['CourseLevel']    == lvl)
        ]['CourseID'].values

        if len(pool) == 0:
            pool = courses[courses['CourseCategory'] == cat]['CourseID'].values
        if len(pool) == 0:
            pool = courses['CourseID'].values

        chosen_ids.append(rng.choice(pool))

    # Remove duplicates
    chosen_ids = list(dict.fromkeys(chosen_ids))

    tx_days  = sorted(rng.integers(0, date_range_days, len(chosen_ids)))
    tx_dates = [START_DATE + timedelta(days=int(d)) for d in tx_days]

    for cid, txdate in zip(chosen_ids, tx_dates):
        price = courses.loc[
            courses['CourseID'] == cid, 'PriceZAR'
        ].values[0]
        cpd_pts = courses.loc[
            courses['CourseID'] == cid, 'CPDPoints'
        ].values[0]

        final_price = round(price * rng.uniform(0.90, 1.10), 2)

        # Completion rate varies by archetype
        completion_rate = {0: 0.75, 1: 0.80, 2: 0.90, 3: 0.45}[arch]
        completed = rng.random() < completion_rate

        records.append({
            'UserID':            uid,
            'CourseID':          cid,
            'EnrollmentDate':    txdate.strftime('%Y-%m-%d'),
            'Amount':            final_price,
            'CompletionStatus':  'Completed' if completed else 'In Progress',
            'CPDPointsEarned':   cpd_pts if completed else 0,
            'CertificateIssued': completed,
        })

transactions = pd.DataFrame(records)

print(f"   ✅ {len(transactions):,} enrollment records created")
print(f"   Date range : {transactions['EnrollmentDate'].min()} → {transactions['EnrollmentDate'].max()}")
print(f"   Total Revenue  : R{transactions['Amount'].sum():,.2f}")
print(f"   Completed      : {transactions['CompletionStatus'].eq('Completed').sum():,}")
print(f"   Total CPD Pts  : {transactions['CPDPointsEarned'].sum():,}")
print(f"\n   Archetype split:")
archetype_names = {
    0: 'Eager Learner',
    1: 'CPD Collector',
    2: 'Clinical Expert',
    3: 'Curious Browser'
}
for k, v in pd.Series(user_archetype).value_counts().sort_index().items():
    print(f"      {archetype_names[k]:<20} : {v} professionals")


# ─────────────────────────────────────────────
# STEP 4 — SAVE TO CSV
# ─────────────────────────────────────────────

print("\n[4/4] Saving files...")
os.makedirs('data/raw', exist_ok=True)

users.to_csv('data/raw/users.csv',               index=False)
courses.to_csv('data/raw/courses.csv',           index=False)
transactions.to_csv('data/raw/transactions.csv', index=False)

print(f"   ✅ data/raw/users.csv        → {len(users):,} rows")
print(f"   ✅ data/raw/courses.csv      → {len(courses):,} rows")
print(f"   ✅ data/raw/transactions.csv → {len(transactions):,} rows")

print("\n" + "=" * 55)
print("   DATA GENERATION COMPLETE")
print("=" * 55)
print(f"""
   Professionals  : {len(users):,}
   CPD Courses    : {len(courses):,}
   Enrollments    : {len(transactions):,}
   Total Revenue  : R{transactions['Amount'].sum():,.2f}
   CPD Points     : {transactions['CPDPointsEarned'].sum():,}
   Completion Rate: {transactions['CompletionStatus'].eq('Completed').mean()*100:.1f}%
""")
