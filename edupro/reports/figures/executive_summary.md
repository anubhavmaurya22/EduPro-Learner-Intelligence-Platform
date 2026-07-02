
# EdUPro Learner Intelligence System
## Executive Summary
### Personalized CPD Recommendations for Pediatric Professionals

**Prepared by:** Anubhav maurywa
**Date:** July 2026
**Platform:** edupro-online.com


---

## The Problem in One Sentence

EdUPro currently sends the same course recommendation 
to a newly qualified classroom teacher and a 20-year 
veteran occupational therapist — and wonders why 
completion rates are low.

---

## What We Built

A machine learning pipeline that:

1. Reads every professional's enrollment history
2. Groups them into 4 career-stage archetypes
3. Recommends the right CPD course for where they 
   are in their professional journey
4. Shows EdUPro management everything in an 
   interactive dashboard

---

## The 4 Professional Archetypes

### 🎓 Eager Learners (47% of professionals)
**Who:** Newly qualified OTs, physios, and 
kinderkineticists in their first 1-5 years

**Behaviour:** Enroll in 20+ courses broadly across 
all categories. Beginner to intermediate level. 
High course volume, moderate spend per course.

**What They Need:** Structured learning pathways.
Show them the natural progression from Beginner → 
Intermediate within their specialty. Highlight 
CPD point accumulation toward HPCSA registration.

**Revenue Opportunity:** R399 avg spend × 940 
professionals = R375,060 segment value

---

### 📜 CPD Collectors (28% of professionals)
**Who:** Mid-career professionals (5-15 years) 
actively building their accreditation portfolio

**Behaviour:** Moderate course volume, intermediate 
level focus, deliberately cross-disciplinary to 
maximize CPD points across HPCSA categories.

**What They Need:** Bundle recommendations across 
categories. Show total CPD points earned. Send 
renewal reminders in Q4 before HPCSA deadlines.

**Revenue Opportunity:** R540 avg spend × 560 
professionals = R302,400 segment value

---

### 🔬 Clinical Experts (22% of professionals)
**Who:** Veteran therapists (15+ years) with deep 
specialization in DCD, sensory processing, or 
motor assessment

**Behaviour:** Few courses but exclusively Advanced. 
Highest spend-per-course (R653 avg). Extremely 
focused — only enroll if directly relevant to 
their specialty.

**What They Need:** Premium Advanced content.
New disorder-specific modules. Case study webinars.
Do NOT recommend beginner content — they ignore it.

**Revenue Opportunity:** R653 avg spend × 440 
professionals = R287,320 segment value
→ ONE extra Advanced course recommendation per 
  quarter = R142,000 additional annual revenue

---

### 👀 Curious Browsers (3% of professionals)
**Who:** Classroom teachers, school psychologists,
and parents/caregivers exploring EdUPro content

**Behaviour:** Very low enrollment (avg 4 courses),
beginner level only, longest time since last 
enrollment (highest recency_days).

**What They Need:** Free preview webinars as entry 
point. Low-price Mini Courses (R299). Re-activation 
email campaign for 90+ day lapsed users.

**Revenue Opportunity:** Currently lowest value —
but highest growth potential through re-activation

---

## How the Recommendation Engine Works

Every course gets a score from 0 to 100 for each 
professional. The score has 4 ingredients: