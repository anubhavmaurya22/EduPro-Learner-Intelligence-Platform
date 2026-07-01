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

# Helper: merge per-chart overrides into PLOTLY_LAYOUT without duplicate-kwarg errors.
# Usage: fig.update_layout(**_layout(height=300, margin=dict(...), yaxis=dict(...)))
def _layout(**overrides):
    return {**PLOTLY_LAYOUT, **overrides}

_MARGIN_TIGHT = dict(l=0, r=0, t=10, b=0)

# ─────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=True)
def load_all():
    users, courses, transactions = generate_data(n_users=500, n_courses=200)
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
        fig_cat.update_layout(**_layout(
            height=300,
            coloraxis_showscale=False,
            margin=_MARGIN_TIGHT,
            yaxis=dict(gridcolor="#2a3350", zerolinecolor="#2a3350",
                       categoryorder="total ascending")
        ))
        st.plotly_chart(fig_cat, use_container_width=True)

        lvl_c = user_tx["CourseLevel"].value_counts().reset_index()
        lvl_c.columns = ["Level", "Count"]
        fig_lvl = px.pie(
            lvl_c, values="Count", names="Level", hole=0.5,
            color_discrete_sequence=["#4cc9f0", "#4361ee", "#7209b7"]
        )
        fig_lvl.update_layout(**_layout(height=200, margin=_MARGIN_TIGHT))
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
    # NOTE: previously this merged against the GLOBAL transactions table
    # deduped by CourseID only — if two different users bought the same
    # course, drop_duplicates() could keep the other user's Amount/Date.
    # user_tx2 (already filtered to this uid) has the correct Amount +
    # TransactionDate directly, so merge onto that instead.
    course_tbl = (
        user_tx
        .merge(user_tx2[["CourseID", "Amount", "TransactionDate"]],
               on="CourseID", how="left")
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
# PAGE 3 — RECOMMENDATIONS
# ═══════════════════════════════════════════════

elif page == "🔮 Recommendations":
    st.markdown("<h1 style='color:#e8edf5'>🔮 Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>Personalized course recommendations, existing learner or cold-start</p>", unsafe_allow_html=True)

    mode = st.radio("Mode", ["Existing Learner", "New Learner (Cold Start)"], horizontal=True)

    if mode == "Existing Learner":
        c1, c2, c3 = st.columns(3)
        with c1:
            uid = st.selectbox("Select Learner", profiles["UserID"].tolist())
        with c2:
            fcat = st.selectbox("Filter Category", ["All"] + CATEGORIES)
        with c3:
            flvl = st.selectbox("Filter Level", ["All"] + LEVELS)

        recs = recommend_courses(uid, profiles, courses, transactions,
                                  top_n=8, filter_category=fcat, filter_level=flvl)

        if recs.empty:
            st.warning("No recommendations found — try loosening the filters.")
        else:
            seg = recs["Segment"].iloc[0]
            color = SEGMENT_COLORS[int(profiles.loc[profiles["UserID"] == uid, "Cluster"].iloc[0])]
            st.markdown(f"""
            <div class='seg-card' style='border-color:{color}'>
                <h3 style='color:{color}'>{uid} · {seg}</h3>
                <p>Target level: <b style='color:#e8edf5'>{recs["TargetLevel"].iloc[0]}</b></p>
            </div>""", unsafe_allow_html=True)

            for _, r in recs.iterrows():
                st.markdown(f"""
                <div class='course-card'>
                    <div style='display:flex;justify-content:space-between;align-items:center'>
                        <div>
                            <b style='color:#e8edf5;font-size:1.0rem'>#{r["Rank"]} · {r["CourseID"]}</b>
                            <span style='color:#8fa3bf;font-size:0.82rem'> — {r["CourseCategory"]} · {r["CourseLevel"]} · {r["CourseType"]}</span>
                        </div>
                        <div style='text-align:right'>
                            <span style='color:#4cc9f0;font-weight:700'>{r["RecommendationScore"]:.2f}</span>
                            <span style='color:#8fa3bf;font-size:0.78rem'> score</span><br>
                            <span style='color:#f72585'>{r["CourseRating"]}★</span>
                        </div>
                    </div>
                    <p style='margin-top:8px;color:#c8d6e8;font-size:0.82rem'>{r["MatchReason"]}</p>
                </div>""", unsafe_allow_html=True)

    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            goal = st.selectbox("Learning Goal", ["Career", "Skill", "Hobby"])
        with c2:
            cat = st.selectbox("Interest Category", CATEGORIES)
        with c3:
            lvl = st.selectbox("Starting Level", LEVELS)

        if st.button("Get Recommendations", type="primary"):
            cold = recommend_new_user(courses, goal=goal, category=cat, level=lvl, top_n=8)
            if cold.empty:
                st.warning("No matching courses found.")
            else:
                for _, r in cold.iterrows():
                    st.markdown(f"""
                    <div class='course-card'>
                        <b style='color:#e8edf5'>#{r["Rank"]} · {r["CourseID"]}</b>
                        <span style='color:#8fa3bf;font-size:0.82rem'> — {r["CourseCategory"]} · {r["CourseLevel"]} · {r["CourseType"]}</span>
                        <span style='float:right;color:#f72585'>{r["CourseRating"]}★</span>
                        <p style='margin-top:6px;color:#c8d6e8;font-size:0.82rem'>{r["MatchReason"]}</p>
                    </div>""", unsafe_allow_html=True)

elif page == "🔵 Cluster Map":
    st.markdown("<h1 style='color:#e8edf5'>🔵 Cluster Map</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>PCA projection, elbow curve, and silhouette diagnostics</p>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("<p class='section-title'>PCA Projection (2D)</p>", unsafe_allow_html=True)
        color_map = {v: SEGMENT_COLORS[k] for k, v in SEGMENT_NAMES.items()}
        fig_pca = px.scatter(
            profiles, x="PCA_1", y="PCA_2",
            color="SegmentName", color_discrete_map=color_map,
            opacity=0.65, hover_data=["UserID", "total_courses", "avg_spending"]
        )
        fig_pca.update_traces(marker=dict(size=6))
        fig_pca.update_layout(**PLOTLY_LAYOUT, height=460)
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_b:
        st.markdown("<p class='section-title'>Cluster Diagnostics</p>", unsafe_allow_html=True)
        st.metric("Overall Silhouette Score", f"{cr['overall_silhouette']:.3f}")
        st.metric("Clusters (K)", N_CLUSTERS)
        expl_var = cr["pca"].explained_variance_ratio_
        st.metric("PCA Variance Explained", f"{expl_var.sum()*100:.1f}%")
        st.caption(f"PC1: {expl_var[0]*100:.1f}% · PC2: {expl_var[1]*100:.1f}%")

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("<p class='section-title'>Elbow Method (Inertia)</p>", unsafe_allow_html=True)
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=cr["k_range"], y=cr["inertias"],
            mode="lines+markers", line=dict(color="#4361ee", width=2.5),
            marker=dict(size=8)
        ))
        fig_elbow.add_vline(x=N_CLUSTERS, line_dash="dot", line_color="#f72585")
        fig_elbow.update_layout(
            **PLOTLY_LAYOUT, height=320,
            xaxis_title="K", yaxis_title="Inertia"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_d:
        st.markdown("<p class='section-title'>Silhouette Score by K</p>", unsafe_allow_html=True)
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=cr["k_range"], y=cr["sil_scores"],
            mode="lines+markers", line=dict(color="#4cc9f0", width=2.5),
            marker=dict(size=8)
        ))
        fig_sil.add_vline(x=N_CLUSTERS, line_dash="dot", line_color="#f72585")
        fig_sil.update_layout(
            **PLOTLY_LAYOUT, height=320,
            xaxis_title="K", yaxis_title="Silhouette Score"
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")
    st.markdown("<p class='section-title'>Per-Learner Silhouette Distribution</p>", unsafe_allow_html=True)
    fig_sd = px.histogram(
        profiles, x="SilhouetteVal", color="SegmentName",
        color_discrete_map=color_map, nbins=40, opacity=0.75,
        barmode="overlay"
    )
    fig_sd.add_vline(x=0, line_dash="dot", line_color="#8fa3bf")
    fig_sd.update_layout(**PLOTLY_LAYOUT, height=300)
    st.plotly_chart(fig_sd, use_container_width=True)
    st.caption("Values near 0 or negative indicate learners sitting between segments — worth a closer look.")

elif page == "📈 Segment Insights":
    st.markdown("<h1 style='color:#e8edf5'>📈 Segment Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>Feature comparison and spending analysis across segments</p>", unsafe_allow_html=True)

    st.markdown("<p class='section-title'>Segment Feature Heatmap</p>", unsafe_allow_html=True)
    heat_feats = ['total_courses', 'avg_spending', 'total_spending', 'avg_course_rating',
                  'diversity_score', 'learning_depth_index', 'beginner_ratio',
                  'advanced_ratio', 'enrollment_frequency', 'n_categories', 'recency_days']
    heat_df = cp.set_index("SegmentName")[heat_feats]
    heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)

    fig_heat = px.imshow(
        heat_norm.T, aspect="auto", color_continuous_scale="Blues",
        labels=dict(color="Normalized"),
    )
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Values normalized 0-1 per feature (row) across segments for comparability.")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<p class='section-title'>Spending by Segment</p>", unsafe_allow_html=True)
        color_map = {v: SEGMENT_COLORS[k] for k, v in SEGMENT_NAMES.items()}
        fig_spend = px.box(
            profiles, x="SegmentName", y="total_spending",
            color="SegmentName", color_discrete_map=color_map, points=False
        )
        fig_spend.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig_spend, use_container_width=True)

    with col_b:
        st.markdown("<p class='section-title'>Learning Depth by Segment</p>", unsafe_allow_html=True)
        fig_depth = px.box(
            profiles, x="SegmentName", y="learning_depth_index",
            color="SegmentName", color_discrete_map=color_map, points=False
        )
        fig_depth.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig_depth, use_container_width=True)

    st.markdown("---")
    st.markdown("<p class='section-title'>Segment Summary Table</p>", unsafe_allow_html=True)
    display_cols = ['SegmentName', 'n_users', 'total_courses', 'avg_spending',
                     'avg_course_rating', 'learning_depth_index',
                     'top_preferred_category', 'top_preferred_level', 'SilhouetteVal']
    st.dataframe(
        cp[display_cols].sort_values("n_users", ascending=False)
          .style.format({
              'total_courses': '{:.1f}', 'avg_spending': '${:.2f}',
              'avg_course_rating': '{:.2f}★', 'learning_depth_index': '{:.2f}',
              'SilhouetteVal': '{:.3f}'
          }),
        use_container_width=True
    )

elif page == "🔍 EDA & Analytics":
    st.markdown("<h1 style='color:#e8edf5'>🔍 EDA & Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>Correlation structure and platform-wide business insights</p>", unsafe_allow_html=True)

    st.markdown("<p class='section-title'>Feature Correlation Matrix</p>", unsafe_allow_html=True)
    corr_feats = CLUSTERING_FEATURES
    corr = profiles[corr_feats].corr()
    fig_corr = px.imshow(
        corr, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=".2f"
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=520)
    fig_corr.update_traces(textfont_size=9)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<p class='section-title'>Spending vs. Course Volume</p>", unsafe_allow_html=True)
        color_map = {v: SEGMENT_COLORS[k] for k, v in SEGMENT_NAMES.items()}
        fig_scat = px.scatter(
            profiles, x="total_courses", y="total_spending",
            color="SegmentName", color_discrete_map=color_map,
            opacity=0.6, hover_data=["UserID"]
        )
        fig_scat.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig_scat, use_container_width=True)

    with col_b:
        st.markdown("<p class='section-title'>Rating Distribution</p>", unsafe_allow_html=True)
        fig_rate = px.histogram(
            courses, x="CourseRating", nbins=25,
            color_discrete_sequence=["#4361ee"]
        )
        fig_rate.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig_rate, use_container_width=True)

    st.markdown("---")
    st.markdown("<p class='section-title'>Key Business Insights</p>", unsafe_allow_html=True)

    top_seg = cp.loc[cp["n_users"].idxmax(), "SegmentName"]
    top_spend_seg = cp.loc[cp["avg_spending"].idxmax(), "SegmentName"]
    top_cat = courses["CourseCategory"].value_counts().idxmax() if "CourseCategory" in courses else None
    low_sil_pct = (profiles["SilhouetteVal"] < 0.1).mean() * 100

    insights = [
        f"🏆 <b>{top_seg}</b> is the largest segment by learner count — the core of the platform's user base.",
        f"💰 <b>{top_spend_seg}</b> spends the most per course on average — the highest-value segment to retain.",
        f"⚠️ <b>{low_sil_pct:.1f}%</b> of learners have a silhouette score below 0.1, meaning segment boundaries are fuzzy for them — good candidates for manual review before targeted campaigns.",
        f"📊 Overall clustering quality (silhouette): <b>{cr['overall_silhouette']:.3f}</b> — "
        + ("solid, well-separated segments." if cr['overall_silhouette'] > 0.25
           else "moderate separation — consider revisiting features or K." if cr['overall_silhouette'] > 0.1
           else "weak separation — segments overlap significantly."),
    ]
    for ins in insights:
        st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)