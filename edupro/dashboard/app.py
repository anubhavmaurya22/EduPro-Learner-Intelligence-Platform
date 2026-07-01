
"""
EduPro Learner Intelligence Platform
File: dashboard/app.py
Run : streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src import (
    generate_data, engineer_features,
    run_clustering, get_cluster_profiles, attach_cluster_labels,
    recommend_courses, recommend_new_user,
    SEGMENT_NAMES, SEGMENT_COLORS, SEGMENT_DESC,
    CATEGORIES, LEVELS, CLUSTERING_FEATURES, N_CLUSTERS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="EduPro Learner Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS THEME
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background: #0f1117; }
    section[data-testid="stSidebar"] { background: #161b27; }
    [data-testid="stMetric"] {
        background: #1e2535;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #2a3350;
    }
    [data-testid="stMetricLabel"]  { color: #8fa3bf !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"]  { color: #e8edf5 !important; font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricDelta"]  { color: #4cc9f0 !important; }
    .seg-card {
        background: #1e2535;
        border-radius: 14px;
        padding: 18px 20px;
        border-left: 5px solid;
        margin-bottom: 10px;
    }
    .seg-card h3 { margin:0 0 4px 0; font-size: 1.1rem; }
    .seg-card p  { margin:0; font-size: 0.82rem; color:#8fa3bf; }
    .section-title {
        font-size: 1.2rem; font-weight: 700;
        color: #e8edf5; margin: 0 0 12px 0;
        border-bottom: 2px solid #2a3350;
        padding-bottom: 6px;
    }
    .insight-box {
        background: #1e2535; border-radius:10px; padding:14px 18px;
        border-left:4px solid #f72585; margin-bottom:8px;
        font-size:0.85rem; color:#c8d6e8;
    }
    .course-card {
        background: #1e2535; border-radius: 12px;
        padding: 16px 18px; border: 1px solid #2a3350;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font_color   ="#c8d6e8",
    font_family  ="Inter, sans-serif",
    margin       =dict(l=30, r=20, t=40, b=30),
    xaxis        =dict(gridcolor="#2a3350", zerolinecolor="#2a3350"),
    yaxis        =dict(gridcolor="#2a3350", zerolinecolor="#2a3350"),
)

# ─────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=True)
def load_all():
    users, courses, transactions = generate_data()
    profiles = engineer_features(users, courses, transactions)
    cr = run_clustering(profiles)
    cp = get_cluster_profiles(profiles, cr)
    profiles = attach_cluster_labels(profiles, cr)
    return users, courses, transactions, profiles, cr, cp


with st.spinner("Loading EduPro Intelligence Engine..."):
    users, courses, transactions, profiles, cr, cp = load_all()

transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px'>
        <div style='font-size:2.4rem'>🎓</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e8edf5'>EduPro</div>
        <div style='font-size:0.72rem; color:#8fa3bf'>Learner Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "📊 Platform Overview",
            "👤 Learner Explorer",
            "🔮 Recommendations",
            "🔵 Cluster Map",
            "📈 Segment Insights",
            "🔍 EDA & Analytics",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#8fa3bf; line-height:2.0'>
    <b style='color:#e8edf5'>Dataset Stats</b><br>
    👥 {len(users):,} Learners<br>
    📚 {len(courses):,} Courses<br>
    🛒 {len(transactions):,} Transactions<br>
    🔵 {N_CLUSTERS} Segments<br>
    📏 Silhouette: <b style='color:#4cc9f0'>{cr["overall_silhouette"]:.3f}</b>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 1 — PLATFORM OVERVIEW
# ═══════════════════════════════════════════════

if page == "📊 Platform Overview":
    st.markdown("<h1 style='color:#e8edf5'>📊 Platform Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>Real-time snapshot of learner activity, revenue, and segment distribution</p>", unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────
    total_rev   = transactions["Amount"].sum()
    avg_ltv     = profiles["total_spending"].mean()
    avg_courses = profiles["total_courses"].mean()
    active_pct  = (profiles["total_courses"] >= 3).mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue",        f"${total_rev:,.0f}",  "↑ 18% YoY")
    k2.metric("Avg. Learner LTV",     f"${avg_ltv:,.2f}",   "↑ 11%")
    k3.metric("Avg. Courses/Learner", f"{avg_courses:.1f}", "↑ 2.3")
    k4.metric("Active Retention",     f"{active_pct:.1f}%", "↑ 4.2pp")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Enrollment Timeline + Category Split ──
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("<p class='section-title'>Enrollment Timeline</p>", unsafe_allow_html=True)
        monthly = (
            transactions
            .set_index("TransactionDate")
            .resample("ME")["CourseID"]
            .count()
            .reset_index()
        )
        monthly.columns = ["Month", "Enrollments"]

        monthly_rev = (
            transactions
            .set_index("TransactionDate")
            .resample("ME")["Amount"]
            .sum()
            .reset_index()
        )
        monthly_rev.columns = ["Month", "Revenue"]

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Enrollments"],
            mode="lines", name="Enrollments",
            line=dict(color="#4361ee", width=2.5),
            fill="tozeroy", fillcolor="rgba(67,97,238,0.12)"
        ))
        fig_line.add_trace(go.Scatter(
            x=monthly_rev["Month"], y=monthly_rev["Revenue"],
            mode="lines", name="Revenue ($)",
            line=dict(color="#f72585", width=2, dash="dot"),
            yaxis="y2"
        ))
        fig_line.update_layout(
            **PLOTLY_LAYOUT, height=300,
            yaxis2=dict(
                overlaying="y", side="right",
                gridcolor="#2a3350", zerolinecolor="#2a3350"
            ),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_b:
        st.markdown("<p class='section-title'>Course Categories</p>", unsafe_allow_html=True)
        cat_dist = (
            transactions
            .merge(courses[["CourseID", "CourseCategory"]], on="CourseID")
            .groupby("CourseCategory")["CourseID"]
            .count()
            .reset_index(name="count")
            .sort_values("count")
        )
        fig_bar = px.bar(
            cat_dist, x="count", y="CourseCategory",
            orientation="h", color="count",
            color_continuous_scale="Blues"
        )
        fig_bar.update_layout(
            **PLOTLY_LAYOUT, height=300,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Segment Cards + Pie ───────────────────
    st.markdown("---")
    col_c, col_d = st.columns([2, 1])

    with col_c:
        st.markdown("<p class='section-title'>Learner Segments</p>", unsafe_allow_html=True)
        for _, row in cp.sort_values("Cluster").iterrows():
            pct = row["n_users"] / len(profiles) * 100
            st.markdown(f"""
            <div class='seg-card' style='border-color:{row["color"]}'>
                <h3 style='color:{row["color"]}'>{row["SegmentName"]}</h3>
                <p>{row["description"]}</p>
                <div style='display:flex;gap:24px;margin-top:8px;font-size:0.8rem;color:#c8d6e8'>
                    <span>👥 <b>{int(row["n_users"]):,}</b> ({pct:.0f}%)</span>
                    <span>📚 <b>{row["total_courses"]:.1f}</b> avg courses</span>
                    <span>💰 <b>${row["avg_spending"]:.0f}</b> avg spend</span>
                    <span>⭐ <b>{row["avg_course_rating"]:.2f}</b> avg rating</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_d:
        st.markdown("<p class='section-title'>Segment Share</p>", unsafe_allow_html=True)
        color_map = {v: SEGMENT_COLORS[k] for k, v in SEGMENT_NAMES.items()}
        fig_pie = px.pie(
            cp, values="n_users", names="SegmentName",
            color="SegmentName", color_discrete_map=color_map,
            hole=0.5
        )
        fig_pie.update_traces(
            textinfo="percent",
            marker=dict(line=dict(color="#0f1117", width=2))
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Demographics ─────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-title'>Learner Demographics</p>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    with d1:
        gender_ct = users["Gender"].value_counts().reset_index()
        gender_ct.columns = ["Gender", "Count"]
        fig_g = px.pie(
            gender_ct, values="Count", names="Gender", hole=0.45,
            color_discrete_sequence=["#4361ee", "#f72585", "#7209b7"],
            title="Gender Distribution"
        )
        fig_g.update_traces(textinfo="percent+label", textfont_size=10)
        fig_g.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_g, use_container_width=True)

    with d2:
        bins   = [17, 24, 34, 44, 54, 65]
        labels = ["18-24", "25-34", "35-44", "45-54", "55-64"]
        users2 = users.copy()
        users2["AgeGroup"] = pd.cut(users2["Age"], bins=bins, labels=labels)
        age_ct = users2["AgeGroup"].value_counts().sort_index().reset_index()
        age_ct.columns = ["AgeGroup", "Count"]
        fig_a = px.bar(
            age_ct, x="AgeGroup", y="Count",
            color="Count", color_continuous_scale="Blues",
            title="Age Distribution"
        )
        fig_a.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False)
        st.plotly_chart(fig_a, use_container_width=True)

    with d3:
        lvl_ct = (
            transactions
            .merge(courses[["CourseID", "CourseLevel"]], on="CourseID")
            .groupby("CourseLevel")["CourseID"].count()
            .reset_index(name="Count")
        )
        fig_l = px.pie(
            lvl_ct, values="Count", names="CourseLevel", hole=0.45,
            color_discrete_sequence=["#4cc9f0", "#4361ee", "#7209b7"],
            title="Enrollment by Level"
        )
        fig_l.update_traces(textinfo="percent+label", textfont_size=10)
        fig_l.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_l, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2 — LEARNER EXPLORER
# ═══════════════════════════════════════════════

elif page == "👤 Learner Explorer":
    st.markdown("<h1 style='color:#e8edf5'>👤 Learner Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>Deep-dive into any individual learner profile</p>", unsafe_allow_html=True)

    uid   = st.selectbox("Select Learner", profiles["UserID"].tolist(), index=0)
    row   = profiles[profiles["UserID"] == uid].iloc[0]
    seg   = int(row["Cluster"])
    color = SEGMENT_COLORS[seg]

    # ── Profile Header ────────────────────────
    st.markdown(f"""
    <div class='seg-card' style='border-color:{color}; display:flex; gap:24px; align-items:center'>
        <div style='font-size:3rem'>🎓</div>
        <div>
            <h2 style='margin:0;color:{color}'>{uid}</h2>
            <div style='color:#8fa3bf;font-size:0.85rem'>
                Age {int(row["Age"])} &nbsp;·&nbsp; {row["Gender"]} &nbsp;·&nbsp;
                Segment: <b style='color:{color}'>{SEGMENT_NAMES[seg]}</b>
            </div>
            <div style='color:#8fa3bf;font-size:0.82rem;margin-top:4px'>
                Preferred: <b style='color:#e8edf5'>{row["preferred_category"]}</b> &nbsp;·&nbsp;
                Level: <b style='color:#e8edf5'>{row["preferred_level"]}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Courses Enrolled",   int(row["total_courses"]))
    m2.metric("Total Spent",        f"${row['total_spending']:,.2f}")
    m3.metric("Avg per Course",     f"${row['avg_spending']:.2f}")
    m4.metric("Avg Rating Given",   f"{row['avg_course_rating']:.2f}★")
    m5.metric("Learning Depth",     f"{row['learning_depth_index']:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    # ── Radar Chart ───────────────────────────
    with col_a:
        st.markdown("<p class='section-title'>Learner Radar Profile</p>", unsafe_allow_html=True)
        radar_feats  = ["total_courses", "avg_spending", "avg_course_rating",
                        "diversity_score", "learning_depth_index", "enrollment_frequency"]
        radar_labels = ["Volume", "Spend", "Rating", "Diversity", "Depth", "Frequency"]

        user_vals, seg_vals = [], []
        for f in radar_feats:
            mn, mx = profiles[f].min(), profiles[f].max()
            user_vals.append((row[f] - mn) / (mx - mn + 1e-9))
            seg_mean = profiles[profiles["Cluster"] == seg][f].mean()
            seg_vals.append((seg_mean - mn) / (mx - mn + 1e-9))

        uc = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor=f"rgba({uc[0]},{uc[1]},{uc[2]},0.3)",
            line=dict(color=color, width=2.5),
            name=uid
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=seg_vals + [seg_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(200,200,200,0.06)",
            line=dict(color="#8fa3bf", width=1.5, dash="dot"),
            name="Segment Avg"
        ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT, height=380,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="#2a3350", color="#8fa3bf"),
                angularaxis=dict(gridcolor="#2a3350", color="#8fa3bf"),
            )
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Category & Level Mix ──────────────────
    with col_b:
        st.markdown("<p class='section-title'>Category & Level Mix</p>", unsafe_allow_html=True)
        user_tx = (
            transactions[transactions["UserID"] == uid]
            .merge(courses[["CourseID", "CourseCategory", "CourseLevel"]], on="CourseID")
        )

        cat_counts = user_tx["CourseCategory"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        fig_cat = px.bar(
            cat_counts, x="Count", y="Category", orientation="h",
            color="Count", color_continuous_scale="Blues"
        )
        fig_cat.update_layout(
            **PLOTLY_LAYOUT, height=200,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(gridcolor="#2a3350", categoryorder="total ascending")
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        lvl_c = user_tx["CourseLevel"].value_counts().reset_index()
        lvl_c.columns = ["Level", "Count"]
        fig_lvl = px.pie(
            lvl_c, values="Count", names="Level", hole=0.5,
            color_discrete_sequence=["#4cc9f0", "#4361ee", "#7209b7"]
        )
        fig_lvl.update_layout(**PLOTLY_LAYOUT, height=200, margin=dict(l=0, r=0, t=10, b=0))
        fig_lvl.update_traces(textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig_lvl, use_container_width=True)

    # ── Enrollment History ────────────────────
    st.markdown("<p class='section-title'>Enrollment History</p>", unsafe_allow_html=True)
    user_tx2 = transactions[transactions["UserID"] == uid].copy()
    user_tx2["TransactionDate"] = pd.to_datetime(user_tx2["TransactionDate"])
    hist = (
        user_tx2
        .set_index("TransactionDate")
        .resample("ME")["CourseID"]
        .count()
        .reset_index()
    )
    hist.columns = ["Month", "Enrollments"]
    fig_hist = px.bar(hist, x="Month", y="Enrollments",
                      color_discrete_sequence=[color])
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=220)
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Course Table ──────────────────────────
    st.markdown("<p class='section-title'>Enrolled Courses</p>", unsafe_allow_html=True)
    course_tbl = (
        user_tx
        .merge(transactions[["CourseID", "Amount", "TransactionDate"]]
               .drop_duplicates("CourseID"), on="CourseID")
        [["CourseID", "CourseCategory", "CourseLevel", "TransactionDate", "Amount"]]
        .sort_values("TransactionDate", ascending=False)
        .reset_index(drop=True)
    )
    course_tbl.index += 1
    st.dataframe(
        course_tbl.style.format({"Amount": "${:,.2f}"}),
        use_container_width=True, height=280
    )


# ═══════════════════════════════════════════════
# PAGES 3-6 PLACEHOLDER (Day 7)
# ═══════════════════════════════════════════════

elif page == "🔮 Recommendations":
    st.markdown("<h1 style='color:#e8edf5'>🔮 Recommendations</h1>", unsafe_allow_html=True)
    st.info("Coming Day 7 — Recommendation engine page")

elif page == "🔵 Cluster Map":
    st.markdown("<h1 style='color:#e8edf5'>🔵 Cluster Map</h1>", unsafe_allow_html=True)
    st.info("Coming Day 7 — PCA scatter, elbow curves, silhouette plot")

elif page == "📈 Segment Insights":
    st.markdown("<h1 style='color:#e8edf5'>📈 Segment Insights</h1>", unsafe_allow_html=True)
    st.info("Coming Day 7 — Feature heatmap, spending analysis")

elif page == "🔍 EDA & Analytics":
    st.markdown("<h1 style='color:#e8edf5'>🔍 EDA & Analytics</h1>", unsafe_allow_html=True)
    st.info("Coming Day 7 — Correlation matrix, business insights")
