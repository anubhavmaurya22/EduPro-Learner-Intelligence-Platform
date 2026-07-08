# Personalized CPD Recommendation System for Pediatric Professionals
## A Machine Learning Approach for EdUPro (edupro-online.com)

**Author:** Anubhav Maurya
**Date:** July 2026
**Platform:** edupro-online.com

---

## Abstract

EdUPro is a South African CPD platform serving pediatric professionals including occupational therapists, kinderkineticists, physiotherapists, and classroom teachers. This paper presents a machine learning pipeline that segments EdUPro professional learners into 4 behavioral archetypes using K-Means clustering on 12 engineered features derived from CPD enrollment history, spending patterns, and clinical learning depth indicators. The resulting hybrid recommendation engine combines peer-segment popularity, professional specialty match, clinical depth fit, and course quality ratings. Validated with a Silhouette Score of 0.387, the system identifies four distinct professional profiles each requiring fundamentally different course pathways.

---

## 1. Introduction

### 1.1 Background
EdUPro was founded by Marene Jooste, a registered Kinderkineticist with a Masters Degree in neuro-sensory-motor development of children and 20+ years of private practice experience. The platform serves professionals across South Africa, UK, Australia, New Zealand and the Netherlands.

### 1.2 Problem Statement
EdUPro currently recommends the same courses to all registered professionals regardless of their clinical background, experience level, or specialty focus. A newly qualified classroom teacher receives the same Advanced DCD clinical diagnosis recommendation as a 20-year veteran occupational therapist resulting in poor course-professional fit, lower completion rates, and unrealized revenue potential.

### 1.3 Objectives
1. Segment EdUPro professionals into meaningful behavioral groups
2. Build a hybrid recommendation engine personalised to each group
3. Deliver an interactive admin dashboard for EdUPro management
4. Identify re-engagement opportunities for lapsed professionals

---

## 2. Dataset Description

### 2.1 Data Sources (Synthetic)

| Table | Rows | Key Columns |
|---|---|---|
| Professionals | 2,000 | UserID, Age, Gender, ProfessionType, YearsExperience, Country |
| CPD Courses | 400 | CourseID, Title, Category, Level, Rating, PriceZAR, CPDPoints, Accreditation |
| Enrollments | 25,000+ | UserID, CourseID, EnrollmentDate, Amount, CompletionStatus, CPDPointsEarned |

### 2.2 Course Categories
- DCD and Motor Disorders
- Fundamental Movement Skills
- Classroom Interventions
- Sensory Processing
- Assessment and Diagnosis
- Sport and Leisure

### 2.3 Professional Distribution
- Occupational Therapists: 28%
- Physiotherapists: 22%
- Kinderkineticists: 20%
- Classroom Teachers: 18%
- School Psychologists: 8%
- Parents and Caregivers: 4%

---

## 3. Exploratory Data Analysis

### Key Findings
- DCD and Motor Disorders is the highest-enrolled category reflecting EdUPro core expertise
- Average course rating is 4.3 out of 5.0 across all categories
- Enrollment peaks in Q1 aligning with HPCSA CPD renewal deadlines
- Clinical Experts show 74% advanced-level enrollment vs 18% for Curious Browsers
- Total platform revenue across 25,000+ enrollments exceeds R2.5 million in synthetic data

---

## 4. Feature Engineering

| Feature | Description | Business Meaning |
|---|---|---|
| total_courses | Unique courses enrolled | CPD engagement breadth |
| avg_spending | Mean spend per course in ZAR | Investment commitment |
| avg_course_rating | Mean rating of enrolled courses | Quality sensitivity |
| diversity_score | Number of categories explored | Generalist vs specialist |
| learning_depth_index | Ratio of advanced content 0 to 1 | Clinical seniority proxy |
| beginner_ratio | Percentage of beginner courses | Early career indicator |
| enrollment_frequency | Courses per active day | Engagement intensity |
| category_concentration | Inverse of category spread | Specialty focus measure |
| n_categories | Raw category count | Breadth of interest |
| recency_days | Days since last enrollment | Lapse risk indicator |
| preferred_category_enc | Label-encoded dominant category | Specialty alignment |
| preferred_level_enc | Label-encoded dominant level | Experience alignment |

The learning_depth_index is the most discriminating feature — it encodes whether a professional is building breadth or going deep, which directly maps to career stage.

---

## 5. Clustering Methodology

### 5.1 K Selection
Elbow curve and Silhouette Score sweep from K=2 to K=6 confirmed K=4 as optimal. This is also supported by domain knowledge that 4 natural professional archetypes exist in pediatric CPD: novice exploring, actively collecting accreditation, deep specialist, and peripheral learner.

### 5.2 Final Model
- Algorithm: K-Means with K=4, n_init=20, max_iter=500
- Validation: Agglomerative Hierarchical Clustering on 500-professional sample
- Overall Silhouette Score: 0.387
- PCA used for 2D visualisation with 42% variance explained

### 5.3 Segment Interpretation

| Segment | Count | Avg Courses | Avg Spend | Depth Index |
|---|---|---|---|---|
| Eager Learners | 940 | 22.0 | R399 | 0.21 |
| CPD Collectors | 560 | 16.0 | R540 | 0.48 |
| Clinical Experts | 440 | 8.0 | R653 | 0.74 |
| Curious Browsers | 60 | 4.0 | R320 | 0.18 |

---

## 6. Recommendation Engine

### 6.1 Hybrid Scoring Architecture

The recommendation score for each course is computed as a weighted sum of four components:

Score = 0.30 multiplied by Peer Popularity
      + 0.25 multiplied by Specialty Match
      + 0.20 multiplied by Clinical Level Fit
      + 0.25 multiplied by Course Rating

### 6.2 Weight Justification

| Component | Weight | Rationale |
|---|---|---|
| Peer Popularity | 30% | CPD professionals trust behaviour of segment peers |
| Specialty Match | 25% | Off-specialty recommendations are consistently ignored |
| Level Fit | 20% | Recommending beginner content to experts wastes their time |
| Rating Quality | 25% | EdUPro average rating of 4.3 is a strong quality signal |

### 6.3 Cold Start Handling
New professionals with no enrollment history answer 3 onboarding questions: profession type, specialty focus, and years of experience. Profession type maps directly to a starting segment hint before any behavioral data is available.

---

## 7. Results and Evaluation

### 7.1 Technical Metrics

| Metric | Value |
|---|---|
| Overall Silhouette Score | 0.387 |
| Hierarchical Clustering Validation | Confirmed within 0.04 |
| PCA Variance Explained | 42% across 2 components |
| Label NaN Count | 0 |
| Automated Tests Passing | 129 out of 129 |
| Dashboard Pages | 6 |

### 7.2 Business Impact Estimates

| Action | Estimated Annual Impact |
|---|---|
| One extra Advanced recommendation to Clinical Experts per quarter | +R142,000 |
| CPD bundle offers to CPD Collectors | +R85,000 |
| Re-activation campaign for Curious Browsers | +R48,000 |
| Total Estimated Annual Uplift | R275,000 |

---

## 8. Dashboard

A 6-page Streamlit dashboard operationalizes all model outputs:

- Platform Overview: CPD revenue, enrollment timeline, segment distribution
- Professional Explorer: Individual CPD history, radar profile, course table
- Recommendations: Personalized top-8 course cards with score breakdown
- Cluster Map: PCA scatter, elbow curve, silhouette plot, radar profiles
- Segment Insights: Feature heatmap, category mix, spending analysis
- EDA and Analytics: Correlation matrix, rating analysis, business insights

Live dashboard: https://anubhavmaurya22-edupro-learner-intell-eduprodashboardapp-gooq0d.streamlit.app

---

## 9. Conclusion

This project demonstrates that machine learning can meaningfully personalize CPD recommendations for pediatric professionals on EdUPro. The 4-segment model aligns naturally with professional career stages and provides EdUPro management with actionable segment-specific marketing and content strategies. The Streamlit dashboard operationalizes these insights into an explorable admin tool ready for production deployment.

---

## 10. References

1. Jooste M. EdUPro CPD Platform. edupro-online.com. 2024.
2. HPCSA Continuing Professional Development Guidelines. 2023.
3. MacQueen J. Some Methods for Classification and Analysis of Multivariate Observations. Proc. 5th Berkeley Symposium on Mathematical Statistics and Probability. 1967.
4. Rousseeuw P. Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis. Computational Statistics and Data Analysis. 1987.
5. Linden G, Smith B, York J. Amazon.com Recommendations: Item-to-Item Collaborative Filtering. IEEE Internet Computing. 2003.
6. Pedregosa F et al. Scikit-learn: Machine Learning in Python. JMLR. 2011.
