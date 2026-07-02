# Personalized CPD Recommendation System for Pediatric Professionals
## A Machine Learning Approach for EdUPro (edupro-online.com)



## Abstract

EdUPro (edupro-online.com) is a South African CPD platform 
serving pediatric professionals including occupational 
therapists, kinderkineticists, physiotherapists, and 
classroom teachers. This paper presents a machine learning 
pipeline that segments EdUPro's professional learners into 
4 behavioral archetypes using K-Means clustering on 12 
engineered features derived from CPD enrollment history, 
spending patterns, and clinical learning depth indicators.

The resulting hybrid recommendation engine combines 
peer-segment popularity, professional specialty match, 
clinical depth fit, and course quality ratings to deliver 
personalized CPD course recommendations. Validated with a 
Silhouette Score of 0.31, the system identifies four 
distinct professional profiles: Eager Learners, CPD 
Collectors, Clinical Experts, and Curious Browsers — 
each requiring fundamentally different course pathways.

---

## 1. Introduction

### 1.1 Background
EdUPro was founded by Marene Jooste, a registered 
Kinderkineticist with a Masters Degree in neuro-sensory-motor 
development of children and 20+ years of private practice 
experience. The platform serves professionals across South 
Africa, UK, Australia, New Zealand and the Netherlands.

### 1.2 Problem Statement
EdUPro currently recommends the same courses to all 
registered professionals regardless of their clinical 
background, experience level, or specialty focus. A 
newly qualified classroom teacher receives the same 
Advanced DCD clinical diagnosis recommendation as a 
20-year veteran occupational therapist — resulting in 
poor course-professional fit, lower completion rates, 
and unrealized revenue potential.

### 1.3 Objectives
1. Segment EdUPro professionals into meaningful behavioral groups
2. Build a hybrid recommendation engine personalised to each group
3. Deliver an interactive admin dashboard for EdUPro management
4. Identify re-engagement opportunities for lapsed professionals

---

## 2. Dataset Description

### 2.1 Data Sources (Synthetic)
Three synthetic tables generated to reflect realistic 
EdUPro platform behaviour:

| Table | Rows | Key Columns |
|---|---|---|
| Professionals | 2,000 | UserID, Age, Gender, ProfessionType, YearsExperience, Country |
| CPD Courses | 400 | CourseID, Title, Category, Level, Rating, PriceZAR, CPDPoints, Accreditation |
| Enrollments | ~25,000 | UserID, CourseID, EnrollmentDate, Amount, CompletionStatus, CPDPointsEarned |

### 2.2 Course Categories