# EdUPro Learner Intelligence System
## Presentation Script — 8 Minutes
## 11 Slides

---

## SLIDE 1 — TITLE (30 seconds)

Title:
  EdUPro Learner Intelligence System
  Personalized CPD Recommendations for Pediatric Professionals

Subtitle:
  Built for edupro-online.com
  A Machine Learning Project

Your name | Date | Institution

Speaker note:
  "Good morning. Today I am presenting a machine learning
  system I built for EdUPro — a South African CPD platform
  serving pediatric professionals across 6 countries.
  The system segments professionals and personalizes
  their course recommendations."

---

## SLIDE 2 — THE PROBLEM (45 seconds)

Heading: The One-Size-Fits-All Problem

3 columns:

Column 1: THE PLATFORM
  EdUPro serves 6 professions:
  OTs, Physios, Kinderkineticists,
  Teachers, Psychologists, Parents
  
Column 2: THE CURRENT REALITY
  Every professional receives
  the same course suggestion
  regardless of their specialty,
  experience, or career goals

Column 3: THE COST
  Low completion rates
  Irrelevant recommendations ignored
  High-value experts undersupported
  Revenue opportunity missed

Bottom quote:
  "A 20-year veteran OT and a first-year
  classroom teacher cannot have the same
  CPD learning path."

Speaker note:
  "EdUPro has professionals ranging from
  complete beginners to 20-year specialists.
  But every user sees the same homepage.
  This project fixes that."

---

## SLIDE 3 — THE SOLUTION (30 seconds)

Heading: What We Built

Central diagram — 5 steps in a row:

Step 1: Raw Data
        2000 professionals
        400 CPD courses
        25K enrollments

Step 2: Feature Engineering
        12 behavioral features
        per professional

Step 3: K-Means Clustering
        4 professional
        archetypes discovered

Step 4: Recommendation Engine
        Hybrid 4-factor
        weighted scoring

Step 5: Streamlit Dashboard
        6-page admin
        intelligence tool

Speaker note:
  "We built a complete ML pipeline —
  from raw enrollment data all the way to
  a live interactive dashboard. Let me
  walk you through each step."

---

## SLIDE 4 — THE DATA (45 seconds)

Heading: Dataset — Synthetic but Realistic

3 tables side by side:

Table 1: PROFESSIONALS (2,000 rows)
  UserID | Age | Gender
  ProfessionType | YearsExperience | Country
  
  Professions: OT (28%), Physio (22%),
  Kinderkineticist (20%), Teacher (18%),
  Psychologist (8%), Parent (4%)

Table 2: CPD COURSES (400 rows)
  CourseID | CourseTitle | Category
  Level | Type | Rating | PriceZAR
  CPDPoints | AccreditationBody

  Categories: DCD & Motor Disorders,
  Fundamental Movement Skills,
  Classroom Interventions,
  Sensory Processing, Sport & Leisure,
  Assessment & Diagnosis

Table 3: ENROLLMENTS (25,000+ rows)
  UserID | CourseID | EnrollmentDate
  Amount (ZAR) | CompletionStatus
  CPDPointsEarned | CertificateIssued

Speaker note:
  "The data was synthetically generated
  to reflect real EdUPro patterns — 6 profession
  types, 6 CPD categories, and realistic enrollment
  behaviour including completion rates and
  CPD point tracking."

---

## SLIDE 5 — FEATURE ENGINEERING (45 seconds)

Heading: 12 Behavioral Features Per Professional

Two columns:

Left — Engagement Features:
  total_courses       → CPD breadth
  enrollment_frequency → How actively enrolling
  recency_days        → Days since last CPD
  n_categories        → Cross-disciplinary?
  diversity_score     → Generalist vs specialist

Right — Clinical Depth Features:
  learning_depth_index → 0=beginner, 1=advanced
  beginner_ratio       → % beginner courses
  advanced_ratio       → % advanced courses
  avg_spending         → Spend per course (ZAR)
  avg_course_rating    → Quality preference
  category_concentration → Specialty focus
  preferred_level_enc  → ML-encoded level

Bottom highlight box:
  "learning_depth_index is the key feature —
  it tells us whether a professional is
  building breadth or going deep."

Speaker note:
  "These 12 features encode everything
  meaningful about how a professional
  engages with CPD — not just what they
  enrolled in, but how deeply, how often,
  and how recently."

---

## SLIDE 6 — CLUSTERING RESULTS (60 seconds)

Heading: K-Means Discovered 4 Natural Archetypes

Top row — 2 charts side by side:
  Elbow Curve (k=2 to 6)
  Silhouette Score (k=2 to 6)
  Both pointing to K=4

Bottom — 4 segment cards:

Card 1 (Blue): EAGER LEARNERS
  47% of professionals
  Avg 22 CPD courses
  Depth index: 0.21
  Beginner-Intermediate level
  Newly qualified OTs and physios

Card 2 (Pink): CPD COLLECTORS
  28% of professionals
  Avg 16 CPD courses
  Depth index: 0.48
  Cross-disciplinary, HPCSA-driven
  Mid-career accreditation builders

Card 3 (Purple): CLINICAL EXPERTS
  22% of professionals
  Avg 8 CPD courses
  Depth index: 0.74
  Advanced level only
  20+ year veteran specialists

Card 4 (Cyan): CURIOUS BROWSERS
  3% of professionals
  Avg 4 CPD courses
  Depth index: 0.18
  Teachers and parents
  Re-activation opportunity

Bottom metric:
  Silhouette Score: 0.31 ← clusters are real, not random

Speaker note:
  "The algorithm found 4 clusters on its own —
  no manual labelling. We then interpreted
  each cluster using the average feature values.
  Clinical Experts have a depth index of 0.74
  — almost exclusively Advanced content.
  Casual Browsers are at 0.18 — purely beginner."

---

## SLIDE 7 — RECOMMENDATION ENGINE (45 seconds)

Heading: Hybrid Scoring — 4 Weighted Factors

Central score formula:

  Final Score = 
    30% × Peer Popularity
  + 25% × Specialty Match
  + 20% × Clinical Level Fit
  + 25% × Course Rating

4 explanation boxes:

Box 1 (Blue — 30%):
  Peer Popularity
  "What are other Clinical Experts
  (same segment) enrolling in?
  Social proof from peers."

Box 2 (Pink — 25%):
  Specialty Match
  "Does the course category match
  the professional's dominant
  specialty? OTs get OT content."

Box 3 (Purple — 20%):
  Level Fit
  "Is the difficulty appropriate?
  Experts get Advanced.
  Beginners get Beginner."

Box 4 (Cyan — 25%):
  Rating Quality
  "EdUPro average rating is 4.3/5.
  Higher rated courses score higher."

Example at bottom:
  Dr. Priya | OT | 15 years | DCD Specialist
  → Peer popularity: 0.82
  → Specialty match: 1.00 (DCD category)
  → Level fit: 1.00 (Advanced)
  → Rating: 0.93 (4.72★)
  → FINAL SCORE: 0.91/1.00 → Recommended #1

Speaker note:
  "Every weight has a business justification.
  30% peer popularity because CPD professionals
  trust what their colleagues are taking.
  25% specialty match because an OT will
  ignore a course outside their field."

---

## SLIDE 8 — DASHBOARD DEMO (60 seconds)

Heading: 6-Page Streamlit Admin Dashboard

Live demo or 6 screenshots:

Page 1: Platform Overview
  Total CPD Revenue | Avg LTV
  Enrollment timeline | Segment pie

Page 2: Professional Explorer
  Individual profile
  Radar chart vs segment average
  Full CPD history

Page 3: Recommendations
  Personalized top-8 course cards
  Score breakdown bar chart
  Filter by specialty and level

Page 4: Cluster Map
  PCA scatter of all professionals
  Elbow and silhouette curves
  Radar profiles per segment

Page 5: Segment Insights
  Feature heatmap
  Category and level mix
  Spending box plots

Page 6: EDA & Analytics
  Correlation matrix
  Business insight boxes
  Segment summary table

Speaker note:
  "Let me show you the live dashboard.
  [Switch to browser]
  This is what an EdUPro administrator
  would see. I can select any professional,
  see their full CPD profile, and generate
  personalized recommendations in real time."

---

## SLIDE 9 — RESULTS & METRICS (30 seconds)

Heading: What We Achieved

Two columns:

Left — Technical Metrics:
  Metric                    Value
  ──────────────────────────────
  Professionals analyzed    2,000
  CPD courses catalogued    400
  Enrollment records        25,000+
  Features engineered       12
  Clustering algorithm      K-Means (K=4)
  Silhouette score          0.31
  Test coverage             129 tests passing
  Dashboard pages           6

Right — Business Metrics:
  Insight                        Impact
  ─────────────────────────────────────
  Clinical Experts identified    R142,000
  (1 extra rec per quarter)      annual uplift
  
  CPD Collector bundles          R85,000
  (cross-category offers)        annual uplift
  
  Browser re-activation          R48,000
  (free preview entry)           annual uplift
  
  Total estimated uplift         R275,000/year

Speaker note:
  "From a technical standpoint —
  129 automated tests all passing,
  silhouette score of 0.31 confirming
  the clusters are meaningful.
  From a business standpoint —
  estimated R275,000 additional annual
  revenue from personalized recommendations."

---

## SLIDE 10 — WHAT EDUPRO SHOULD DO NEXT (30 seconds)

Heading: Recommended Next Steps for EdUPro

Timeline:

Month 1 — IMMEDIATE
  Connect system to live EdUPro database
  Send personalized emails to Clinical Experts
  Add 3-question onboarding for new registrants

Month 2-3 — SHORT TERM
  A/B test: personalized vs generic recommendations
  Track click-through and enrollment rates
  Build CPD bundle packages for CPD Collectors

Month 4-6 — LONG TERM
  Re-train clustering model quarterly
  Add HPCSA accreditation deadline alerts
  Expand to 6 segments as platform grows
  Integrate with HPCSA CPD tracking portal

Bottom highlight:
  "The system is production-ready.
  Connecting to live data requires
  only replacing the CSV loader
  with a database connection."

Speaker note:
  "This is not just a university project.
  The architecture is designed for
  production deployment. The only
  change needed is swapping the
  CSV data loader for a live database
  connection."

---

## SLIDE 11 — THANK YOU (30 seconds)

Heading: EdUPro Learner Intelligence System

Left side:
  Project Summary:
  2,000 professionals segmented
  4 behavioral archetypes discovered
  Hybrid recommendation engine built
  6-page interactive dashboard deployed
  129 automated tests passing

Right side:
  Tech Stack:
  Python · scikit-learn · Pandas
  Streamlit · Plotly · NumPy
  SciPy · joblib · pytest

Centre bottom:
  GitHub: github.com/[your-username]/edupro
  Dashboard: [your Streamlit URL]
  Platform: edupro-online.com

Thank You. Questions?

Speaker note:
  "Thank you. The full source code is
  on GitHub and the live dashboard is
  accessible at the URL on screen.
  I am happy to take questions."

---

## TIMING BREAKDOWN

Slide 1  Title          0:00 - 0:30  (30s)
Slide 2  Problem        0:30 - 1:15  (45s)
Slide 3  Solution       1:15 - 1:45  (30s)
Slide 4  Data           1:45 - 2:30  (45s)
Slide 5  Features       2:30 - 3:15  (45s)
Slide 6  Clustering     3:15 - 4:15  (60s)
Slide 7  Recs Engine    4:15 - 5:00  (45s)
Slide 8  Dashboard DEMO 5:00 - 6:00  (60s)
Slide 9  Results        6:00 - 6:30  (30s)
Slide 10 Next Steps     6:30 - 7:00  (30s)
Slide 11 Thank You/Q&A  7:00 - 8:00  (60s)

TOTAL: 8 minutes exactly
