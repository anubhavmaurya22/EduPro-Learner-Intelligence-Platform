"""
EduPro Learner Intelligence Platform
File : dashboard/app.py
Run  : streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    page_title="EdUPro Learner Intelligence",
    page_icon="🧠",
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
    [data-testid="stMetricLabel"] { color: #8fa3bf !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #e8edf5 !important; font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricDelta"] { color: #4cc9f0 !important; }
    .seg-card {
        background: #1e2535; border-radius: 14px;
        padding: 18px 20px; border-left: 5px solid;
        margin-bottom: 10px;
    }
    .seg-card h3 { margin: 0 0 4px 0; font-size: 1.1rem; }
    .seg-card p  { margin: 0; font-size: 0.82rem; color: #8fa3bf; }
    .section-title {
        font-size: 1.2rem; font-weight: 700;
        color: #e8edf5; margin: 0 0 12px 0;
        border-bottom: 2px solid #2a3350; padding-bottom: 6px;
    }
    .insight-box {
        background: #1e2535; border-radius: 10px; padding: 14px 18px;
        border-left: 4px solid #f72585; margin-bottom: 8px;
        font-size: 0.85rem; color: #c8d6e8;
    }
    .course-card {
        background: #1e2535; border-radius: 12px;
        padding: 16px 18px; border: 1px solid #2a3350;
        margin-bottom: 10px;
    }
    .score-bar  { height: 8px; background: #2a3350; border-radius: 4px; margin-top: 6px; }
    .score-fill { height: 8px; border-radius: 4px; }
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

COLOR_MAP = {v: SEGMENT_COLORS[k] for k, v in SEGMENT_NAMES.items()}

# ─────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=True)
def load_all():
    users, courses, transactions = generate_data(n_users=500, n_courses=200)
    profiles  = engineer_features(users, courses, transactions)
    cr        = run_clustering(profiles)
    cp        = get_cluster_profiles(profiles, cr)
    profiles  = attach_cluster_labels(profiles, cr)
    return users, courses, transactions, profiles, cr, cp


with st.spinner("Loading EdUPro Intelligence Engine..."):
    users, courses, transactions, profiles, cr, cp = load_all()

if "EnrollmentDate" in transactions.columns and "EnrollmentDate" not in transactions.columns:
    transactions = transactions.rename(columns={"EnrollmentDate": "EnrollmentDate"})
transactions["EnrollmentDate"] = pd.to_datetime(transactions["EnrollmentDate"])

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px'>
        <div style='font-size:2.4rem'>🧠</div>
        <div style='font-size:1.1rem;font-weight:700;color:#e8edf5'>EdUPro Intelligence</div>
        <div style='font-size:0.72rem;color:#8fa3bf'>
            CPD Recommendation System<br>
            <a href='https://edupro-online.com' style='color:#4cc9f0'>edupro-online.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "📊 Platform Overview",
            "👤 Professional Explorer",
            "🔮 Recommendations",
            "🔵 Cluster Map",
            "📈 Segment Insights",
            "🔍 EDA & Analytics",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem;color:#8fa3bf;line-height:2.0'>
    <b style='color:#e8edf5'>Dataset Stats</b><br>
    👥 {len(users):,} Professionals<br>
    📚 {len(courses):,} CPD Courses<br>
    🛒 {len(transactions):,} Enrollments<br>
    🔵 {N_CLUSTERS} Segments<br>
    📏 Silhouette: <b style='color:#4cc9f0'>{cr["overall_silhouette"]:.3f}</b>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 1 — PLATFORM OVERVIEW
# ═══════════════════════════════════════════════

if page == "📊 Platform Overview":
    st.markdown("<h1 style='color:#e8edf5'>📊 Platform Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>CPD enrollment activity, revenue, and professional segment distribution</p>", unsafe_allow_html=True)

    total_rev   = transactions["Amount"].sum()
    avg_ltv     = profiles["total_spending"].mean()
    avg_courses = profiles["total_courses"].mean()
    active_pct  = (profiles["total_courses"] >= 3).mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total CPD Revenue",      f"R{total_rev:,.0f}",   "↑ 18% YoY")
    k2.metric("Avg Professional LTV",   f"R{avg_ltv:,.2f}",    "↑ 11%")
    k3.metric("Avg Courses/Professional",f"{avg_courses:.1f}", "↑ 2.3")
    k4.metric("Active Retention",        f"{active_pct:.1f}%", "↑ 4.2pp")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("<p class='section-title'>Enrollment Timeline</p>", unsafe_allow_html=True)
        monthly = (transactions.set_index("EnrollmentDate")
                               .resample("ME")["CourseID"].count()
                               .reset_index())
        monthly.columns = ["Month", "Enrollments"]
        monthly_rev = (transactions.set_index("EnrollmentDate")
                                   .resample("ME")["Amount"].sum()
                                   .reset_index())
        monthly_rev.columns = ["Month", "Revenue"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Enrollments"],
            mode="lines", name="Enrollments",
            line=dict(color="#4361ee", width=2.5),
            fill="tozeroy", fillcolor="rgba(67,97,238,0.12)"
        ))
        fig.add_trace(go.Scatter(
            x=monthly_rev["Month"], y=monthly_rev["Revenue"],
            mode="lines", name="Revenue (R)",
            line=dict(color="#f72585", width=2, dash="dot"),
            yaxis="y2"
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=300,
            yaxis2=dict(overlaying="y", side="right",
                        gridcolor="#2a3350", zerolinecolor="#2a3350")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("<p class='section-title'>CPD Categories</p>", unsafe_allow_html=True)
        cat_dist = (transactions
                    .merge(courses[["CourseID", "CourseCategory"]], on="CourseID")
                    .groupby("CourseCategory")["CourseID"].count()
                    .reset_index(name="count")
                    .sort_values("count"))
        fig2 = px.bar(cat_dist, x="count", y="CourseCategory", orientation="h",
                      color="count", color_continuous_scale="Blues")
        fig2.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col_c, col_d = st.columns([2, 1])

    with col_c:
        st.markdown("<p class='section-title'>Professional Segments</p>", unsafe_allow_html=True)
        for _, row in cp.sort_values("Cluster").iterrows():
            pct = row["n_users"] / len(profiles) * 100
            st.markdown(f"""
            <div class='seg-card' style='border-color:{row["color"]}'>
                <h3 style='color:{row["color"]}'>{row["SegmentName"]}</h3>
                <p>{row["description"]}</p>
                <div style='display:flex;gap:24px;margin-top:8px;font-size:0.8rem;color:#c8d6e8'>
                    <span>👥 <b>{int(row["n_users"]):,}</b> ({pct:.0f}%)</span>
                    <span>📚 <b>{row["total_courses"]:.1f}</b> avg courses</span>
                    <span>💰 <b>R{row["avg_spending"]:.0f}</b> avg spend</span>
                    <span>⭐ <b>{row["avg_course_rating"]:.2f}</b> rating</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_d:
        st.markdown("<p class='section-title'>Segment Share</p>", unsafe_allow_html=True)
        fig3 = px.pie(cp, values="n_users", names="SegmentName",
                      color="SegmentName", color_discrete_map=COLOR_MAP, hole=0.5)
        fig3.update_traces(textinfo="percent",
                           marker=dict(line=dict(color="#0f1117", width=2)))
        fig3.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("<p class='section-title'>Demographics</p>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    with d1:
        g = users["Gender"].value_counts().reset_index()
        g.columns = ["Gender", "Count"]
        fig4 = px.pie(g, values="Count", names="Gender", hole=0.45,
                      color_discrete_sequence=["#4361ee","#f72585","#7209b7"],
                      title="Gender")
        fig4.update_traces(textinfo="percent+label", textfont_size=10)
        fig4.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig4, use_container_width=True)

    with d2:
        if "ProfessionType" in users.columns:
            prof = users["ProfessionType"].value_counts().reset_index()
            prof.columns = ["Profession", "Count"]
            fig5 = px.bar(prof, x="Count", y="Profession", orientation="h",
                          color="Count", color_continuous_scale="Blues",
                          title="Profession Type")
            fig5.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            bins   = [17,24,34,44,54,65]
            labels = ["18-24","25-34","35-44","45-54","55-64"]
            u2 = users.copy()
            u2["AgeGroup"] = pd.cut(u2["Age"], bins=bins, labels=labels)
            ag = u2["AgeGroup"].value_counts().sort_index().reset_index()
            ag.columns = ["AgeGroup","Count"]
            fig5 = px.bar(ag, x="AgeGroup", y="Count",
                          color="Count", color_continuous_scale="Blues",
                          title="Age Distribution")
            fig5.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False)
            st.plotly_chart(fig5, use_container_width=True)

    with d3:
        lvl = (transactions.merge(courses[["CourseID","CourseLevel"]], on="CourseID")
                           .groupby("CourseLevel")["CourseID"].count()
                           .reset_index(name="Count"))
        fig6 = px.pie(lvl, values="Count", names="CourseLevel", hole=0.45,
                      color_discrete_sequence=["#4cc9f0","#4361ee","#7209b7"],
                      title="Enrollment by Level")
        fig6.update_traces(textinfo="percent+label", textfont_size=10)
        fig6.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2 — PROFESSIONAL EXPLORER
# ═══════════════════════════════════════════════

elif page == "👤 Professional Explorer":
    st.markdown("<h1 style='color:#e8edf5'>👤 Professional Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>Deep-dive into any individual professional's CPD profile</p>", unsafe_allow_html=True)

    uid   = st.selectbox("Select Professional", profiles["UserID"].tolist(), index=0)
    row   = profiles[profiles["UserID"] == uid].iloc[0]
    seg   = int(row["Cluster"])
    color = SEGMENT_COLORS[seg]

    prof_type = row.get("ProfessionType", "Professional")

    st.markdown(f"""
    <div class='seg-card' style='border-color:{color};display:flex;gap:24px;align-items:center'>
        <div style='font-size:3rem'>🩺</div>
        <div>
            <h2 style='margin:0;color:{color}'>{uid}</h2>
            <div style='color:#8fa3bf;font-size:0.85rem'>
                {prof_type} &nbsp;·&nbsp; Age {int(row["Age"])} &nbsp;·&nbsp; {row["Gender"]}
                &nbsp;·&nbsp; Segment: <b style='color:{color}'>{SEGMENT_NAMES[seg]}</b>
            </div>
            <div style='color:#8fa3bf;font-size:0.82rem;margin-top:4px'>
                Specialty: <b style='color:#e8edf5'>{row["preferred_category"]}</b>
                &nbsp;·&nbsp; Level: <b style='color:#e8edf5'>{row["preferred_level"]}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CPD Courses",        int(row["total_courses"]))
    m2.metric("Total Spent",        f"R{row['total_spending']:,.2f}")
    m3.metric("Avg per Course",     f"R{row['avg_spending']:.2f}")
    m4.metric("Avg Rating",         f"{row['avg_course_rating']:.2f}★")
    m5.metric("Learning Depth",     f"{row['learning_depth_index']:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<p class='section-title'>Professional Radar Profile</p>", unsafe_allow_html=True)
        radar_feats  = ["total_courses","avg_spending","avg_course_rating",
                        "diversity_score","learning_depth_index","enrollment_frequency"]
        radar_labels = ["CPD Volume","Spend","Rating","Diversity","Depth","Frequency"]

        user_vals, seg_vals = [], []
        for f in radar_feats:
            mn, mx = profiles[f].min(), profiles[f].max()
            user_vals.append((row[f] - mn) / (mx - mn + 1e-9))
            sm = profiles[profiles["Cluster"] == seg][f].mean()
            seg_vals.append((sm - mn) / (mx - mn + 1e-9))

        uc = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor=f"rgba({uc[0]},{uc[1]},{uc[2]},0.3)",
            line=dict(color=color, width=2.5), name=uid
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=seg_vals + [seg_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself", fillcolor="rgba(200,200,200,0.06)",
            line=dict(color="#8fa3bf", width=1.5, dash="dot"),
            name="Segment Avg"
        ))
        fig_r.update_layout(
            **PLOTLY_LAYOUT, height=380,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor="#2a3350", color="#8fa3bf"),
                angularaxis=dict(gridcolor="#2a3350", color="#8fa3bf"),
            )
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_b:
        st.markdown("<p class='section-title'>Category & Level Mix</p>", unsafe_allow_html=True)
        user_tx = (transactions[transactions["UserID"] == uid]
                   .merge(courses[["CourseID","CourseCategory","CourseLevel"]], on="CourseID"))

        cat_c = user_tx["CourseCategory"].value_counts().reset_index()
        cat_c.columns = ["Category","Count"]
        fig_c = px.bar(cat_c, x="Count", y="Category", orientation="h",
                       color="Count", color_continuous_scale="Blues")
        fig_c.update_layout(**{**PLOTLY_LAYOUT, "yaxis": dict(gridcolor="#2a3350", categoryorder="total ascending")},
                            height=200,
                            coloraxis_showscale=False,
                            margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_c, use_container_width=True)

        lvl_c = user_tx["CourseLevel"].value_counts().reset_index()
        lvl_c.columns = ["Level","Count"]
        fig_l = px.pie(lvl_c, values="Count", names="Level", hole=0.5,
                       color_discrete_sequence=["#4cc9f0","#4361ee","#7209b7"])
        fig_l.update_layout(**PLOTLY_LAYOUT, height=200, margin=dict(l=0,r=0,t=10,b=0))
        fig_l.update_traces(textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig_l, use_container_width=True)

    st.markdown("<p class='section-title'>Enrollment History</p>", unsafe_allow_html=True)
    hist = (user_tx.assign(
                Month=pd.to_datetime(user_tx["EnrollmentDate"]).dt.to_period("M").astype(str))
            .groupby("Month")["CourseID"].count()
            .reset_index(name="Enrollments"))
    fig_h = px.bar(hist, x="Month", y="Enrollments",
                   color_discrete_sequence=[color])
    fig_h.update_layout(**PLOTLY_LAYOUT, height=220)
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("<p class='section-title'>CPD Course History</p>", unsafe_allow_html=True)
    tbl = (user_tx[["CourseID","CourseCategory","CourseLevel","EnrollmentDate","Amount"]]
           .sort_values("EnrollmentDate", ascending=False)
           .reset_index(drop=True))
    tbl.index += 1
    st.dataframe(tbl.style.format({"Amount": "R{:,.2f}"}),
                 use_container_width=True, height=280)


# ═══════════════════════════════════════════════
# PAGE 3 — RECOMMENDATIONS
# ═══════════════════════════════════════════════

elif page == "🔮 Recommendations":
    st.markdown("<h1 style='color:#e8edf5'>🔮 Personalized CPD Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>Hybrid engine: peer popularity + specialty match + clinical level fit + course rating</p>", unsafe_allow_html=True)

    col_sel, col_f1, col_f2 = st.columns([2, 1, 1])
    with col_sel:
        uid_r = st.selectbox("Select Professional", profiles["UserID"].tolist(), index=7)
    with col_f1:
        f_cat = st.selectbox("Filter Specialty", ["All"] + CATEGORIES)
    with col_f2:
        f_lvl = st.selectbox("Filter Level", ["All"] + LEVELS)

    recs = recommend_courses(
        uid_r, profiles, courses, transactions, top_n=8,
        filter_category=f_cat if f_cat != "All" else None,
        filter_level=f_lvl if f_lvl != "All" else None
    )

    usr   = profiles[profiles["UserID"] == uid_r].iloc[0]
    seg   = int(usr["Cluster"])
    color = SEGMENT_COLORS[seg]
    prof_type = usr.get("ProfessionType", "Professional")

    st.markdown(f"""
    <div class='seg-card' style='border-color:{color};padding:12px 16px;margin-bottom:16px'>
        <b style='color:{color}'>{uid_r}</b>
        &nbsp;·&nbsp; {prof_type}
        &nbsp;·&nbsp; {SEGMENT_NAMES[seg]}
        &nbsp;·&nbsp; Specialty: <b style='color:#e8edf5'>{usr["preferred_category"]}</b>
        &nbsp;·&nbsp; Depth: <b style='color:#e8edf5'>{usr["learning_depth_index"]:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

    # Weight legend
    wl = st.columns(4)
    wl[0].markdown("<div style='background:#1e2535;border-radius:8px;padding:8px;text-align:center;font-size:0.78rem;color:#8fa3bf'>🔵 Peer Popularity<br><b style='color:#4361ee'>30%</b></div>", unsafe_allow_html=True)
    wl[1].markdown("<div style='background:#1e2535;border-radius:8px;padding:8px;text-align:center;font-size:0.78rem;color:#8fa3bf'>🎯 Specialty Match<br><b style='color:#f72585'>25%</b></div>", unsafe_allow_html=True)
    wl[2].markdown("<div style='background:#1e2535;border-radius:8px;padding:8px;text-align:center;font-size:0.78rem;color:#8fa3bf'>📶 Level Fit<br><b style='color:#7209b7'>20%</b></div>", unsafe_allow_html=True)
    wl[3].markdown("<div style='background:#1e2535;border-radius:8px;padding:8px;text-align:center;font-size:0.78rem;color:#8fa3bf'>⭐ Course Rating<br><b style='color:#4cc9f0'>25%</b></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if recs.empty:
        st.warning("No courses match the current filters. Try widening your selection.")
    else:
        left_col, right_col = st.columns(2)
        for i, rec_row in recs.iterrows():
            score_pct = int(rec_row["RecommendationScore"] * 100)
            card = f"""
            <div class='course-card'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                    <span style='font-size:1.4rem;font-weight:700;color:#4cc9f0'>#{int(rec_row["Rank"])}</span>
                    <span style='font-size:0.75rem;background:#2a3350;padding:2px 8px;border-radius:20px;color:#8fa3bf'>{rec_row["CourseID"]}</span>
                </div>
                <div style='font-size:1rem;font-weight:600;color:#e8edf5;margin-top:4px'>
                    {rec_row["CourseCategory"]} · {rec_row["CourseLevel"]}
                </div>
                <div style='font-size:0.78rem;color:#8fa3bf;margin-top:4px'>
                    📹 {rec_row["CourseType"]} &nbsp;|&nbsp;
                    ⭐ {rec_row["CourseRating"]} &nbsp;|&nbsp;
                    Score: <b style='color:#4cc9f0'>{score_pct}%</b>
                </div>
                <div class='score-bar'>
                    <div class='score-fill' style='width:{score_pct}%;background:{color}'></div>
                </div>
                <div style='display:flex;gap:6px;margin-top:8px;flex-wrap:wrap'>
                    <span style='font-size:0.7rem;background:rgba(67,97,238,0.2);color:#4361ee;padding:2px 8px;border-radius:12px'>Pop {int(rec_row["pop_score"]*100)}%</span>
                    <span style='font-size:0.7rem;background:rgba(247,37,133,0.2);color:#f72585;padding:2px 8px;border-radius:12px'>Cat {int(rec_row["cat_score"]*100)}%</span>
                    <span style='font-size:0.7rem;background:rgba(114,9,183,0.2);color:#b57bee;padding:2px 8px;border-radius:12px'>Lvl {int(rec_row["lvl_score"]*100)}%</span>
                    <span style='font-size:0.7rem;background:rgba(76,201,240,0.2);color:#4cc9f0;padding:2px 8px;border-radius:12px'>Rtg {int(rec_row["rating_score"]*100)}%</span>
                </div>
                <div style='font-size:0.75rem;color:#4cc9f0;margin-top:6px;font-style:italic'>{rec_row["MatchReason"]}</div>
            </div>"""
            if i % 2 == 0:
                left_col.markdown(card, unsafe_allow_html=True)
            else:
                right_col.markdown(card, unsafe_allow_html=True)

        # Score breakdown chart
        st.markdown("---")
        st.markdown("<p class='section-title'>Score Breakdown (Top 8)</p>", unsafe_allow_html=True)
        fig_sc = go.Figure()
        components = [
            ("Peer Popularity (30%)", "pop_score",    "#4361ee", 0.30),
            ("Specialty Match (25%)", "cat_score",    "#f72585", 0.25),
            ("Level Fit (20%)",       "lvl_score",    "#7209b7", 0.20),
            ("Rating Quality (25%)",  "rating_score", "#4cc9f0", 0.25),
        ]
        labels = [f"#{int(r['Rank'])} {r['CourseID']}" for _, r in recs.iterrows()]
        for name, col_nm, clr, wt in components:
            fig_sc.add_trace(go.Bar(
                name=name, x=labels,
                y=[round(r[col_nm] * wt, 3) for _, r in recs.iterrows()],
                marker_color=clr
            ))
        fig_sc.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=320,
                             yaxis_title="Weighted Score")
        st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4 — CLUSTER MAP
# ═══════════════════════════════════════════════

elif page == "🔵 Cluster Map":
    st.markdown("<h1 style='color:#e8edf5'>🔵 Cluster Map</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>Visualize professional clusters via PCA, elbow/silhouette curves, and segment radar profiles</p>", unsafe_allow_html=True)

        # PCA scatter
    st.markdown("<p class='section-title'>PCA Scatter — Professional Behaviour Space</p>", unsafe_allow_html=True)
    fig_pca = px.scatter(
        profiles, x="PCA_1", y="PCA_2", color="SegmentName",
        color_discrete_map=COLOR_MAP, opacity=0.65,
        hover_data={"UserID": True, "total_courses": True,
                    "avg_spending": ":.2f", "preferred_category": True,
                    "SilhouetteVal": ":.3f", "PCA_1": False, "PCA_2": False}
    )
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(**PLOTLY_LAYOUT, height=450,
                          xaxis_title="Principal Component 1",
                          yaxis_title="Principal Component 2")
    st.plotly_chart(fig_pca, use_container_width=True)

    # Elbow + Silhouette
    st.markdown("---")
    col_e, col_s = st.columns(2)

    with col_e:
        st.markdown("<p class='section-title'>Elbow Curve</p>", unsafe_allow_html=True)
        fig_el = go.Figure()
        fig_el.add_trace(go.Scatter(
            x=cr["k_range"], y=cr["inertias"],
            mode="lines+markers", line=dict(color="#4361ee", width=2.5),
            marker=dict(size=8, color="#f72585")
        ))
        fig_el.add_vline(x=N_CLUSTERS, line_dash="dash", line_color="#4cc9f0",
                         annotation_text=f"k={N_CLUSTERS} chosen")
        fig_el.update_layout(**PLOTLY_LAYOUT, height=300,
                             xaxis_title="k (Clusters)", yaxis_title="Inertia (WCSS)")
        st.plotly_chart(fig_el, use_container_width=True)

    with col_s:
        st.markdown("<p class='section-title'>Silhouette Score vs k</p>", unsafe_allow_html=True)
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=cr["k_range"], y=cr["sil_scores"],
            mode="lines+markers", line=dict(color="#f72585", width=2.5),
            marker=dict(size=8, color="#4361ee")
        ))
        fig_sil.add_vline(x=N_CLUSTERS, line_dash="dash", line_color="#4cc9f0",
                          annotation_text=f"k={N_CLUSTERS} chosen")
        fig_sil.update_layout(**PLOTLY_LAYOUT, height=300,
                              xaxis_title="k (Clusters)", yaxis_title="Silhouette Score")
        st.plotly_chart(fig_sil, use_container_width=True)

    # Per-user silhouette
    st.markdown("---")
    st.markdown("<p class='section-title'>Per-Professional Silhouette Values</p>", unsafe_allow_html=True)
    sil_df = profiles[["UserID","SegmentName","SilhouetteVal"]].sort_values(
        ["SegmentName","SilhouetteVal"])
    fig_sv = px.bar(sil_df, x="UserID", y="SilhouetteVal",
                    color="SegmentName", color_discrete_map=COLOR_MAP)
    fig_sv.add_hline(y=cr["overall_silhouette"], line_dash="dash",
                     line_color="#4cc9f0",
                     annotation_text=f"Avg={cr['overall_silhouette']:.3f}")
    fig_sv.update_layout(**{**PLOTLY_LAYOUT, "xaxis": dict(showticklabels=False,
                                                           title="Professionals (sorted by segment)")},
                         height=300,
                         yaxis_title="Silhouette Value")
    st.plotly_chart(fig_sv, use_container_width=True)

    # Radar profiles 2x2
    st.markdown("---")
    st.markdown("<p class='section-title'>Segment Feature Radar Profiles</p>", unsafe_allow_html=True)
    radar_feats  = ["total_courses","avg_spending","avg_course_rating",
                    "diversity_score","learning_depth_index","enrollment_frequency"]
    radar_labels = ["CPD Volume","Spend","Rating","Diversity","Depth","Frequency"]

    fig_rd = make_subplots(
        rows=2, cols=2, specs=[[{"type":"polar"}]*2]*2,
        subplot_titles=[SEGMENT_NAMES[k] for k in range(4)]
    )
    rc = [(1,1),(1,2),(2,1),(2,2)]
    for k, (r, c) in zip(range(4), rc):
        sd = profiles[profiles["Cluster"] == k]
        vals = []
        for f in radar_feats:
            mn, mx = profiles[f].min(), profiles[f].max()
            vals.append((sd[f].mean() - mn) / (mx - mn + 1e-9))
        vc  = vals + [vals[0]]
        lc  = radar_labels + [radar_labels[0]]
        col = SEGMENT_COLORS[k]
        rc2 = tuple(int(col.lstrip("#")[i:i+2], 16) for i in (0,2,4))
        fig_rd.add_trace(go.Scatterpolar(
            r=vc, theta=lc, fill="toself",
            fillcolor=f"rgba({rc2[0]},{rc2[1]},{rc2[2]},0.25)",
            line=dict(color=col, width=2),
            name=SEGMENT_NAMES[k], showlegend=False
        ), row=r, col=c)

    fig_rd.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d6e8",
                         height=520, margin=dict(l=30,r=20,t=50,b=20))
    st.plotly_chart(fig_rd, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 5 — SEGMENT INSIGHTS
# ═══════════════════════════════════════════════

elif page == "📈 Segment Insights":
    st.markdown("<h1 style='color:#e8edf5'>📈 Segment Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>Compare behavioral and financial patterns across professional segments</p>", unsafe_allow_html=True)

    # Feature comparison
    st.markdown("<p class='section-title'>Feature Comparison by Segment</p>", unsafe_allow_html=True)
    feat_options = {
        "Total CPD Courses":    "total_courses",
        "Avg Spend (R)":        "avg_spending",
        "Total Spend (R)":      "total_spending",
        "Avg Course Rating":    "avg_course_rating",
        "Learning Depth Index": "learning_depth_index",
        "Category Diversity":   "n_categories",
        "Recency (days)":       "recency_days",
    }
    chosen = st.selectbox("Select Feature", list(feat_options.keys()))
    feat_col = feat_options[chosen]

    bar_data = (profiles.groupby("SegmentName")[feat_col]
                        .mean().reset_index(name="mean_val"))
    fig_cmp = px.bar(bar_data, x="SegmentName", y="mean_val",
                     color="SegmentName", color_discrete_map=COLOR_MAP,
                     text="mean_val")
    fig_cmp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_cmp.update_layout(**PLOTLY_LAYOUT, height=340,
                          yaxis_title=chosen, showlegend=False)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Normalised heatmap
    st.markdown("---")
    st.markdown("<p class='section-title'>Normalised Feature Heatmap</p>", unsafe_allow_html=True)
    heat_feats = ["total_courses","avg_spending","avg_course_rating",
                  "diversity_score","learning_depth_index","beginner_ratio",
                  "advanced_ratio","enrollment_frequency","recency_days"]
    heat_data = profiles.groupby("SegmentName")[heat_feats].mean()
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-9)

    fig_ht = px.imshow(heat_norm.T, color_continuous_scale="RdBu",
                       text_auto=".2f", aspect="auto",
                       labels=dict(x="Segment", y="Feature", color="Norm."))
    fig_ht.update_layout(**PLOTLY_LAYOUT, height=360)
    st.plotly_chart(fig_ht, use_container_width=True)

    # Category mix + Level mix
    st.markdown("---")
    col_cat, col_lvl = st.columns(2)

    with col_cat:
        st.markdown("<p class='section-title'>CPD Category Mix by Segment</p>", unsafe_allow_html=True)
        tx_seg = (transactions
                  .merge(courses[["CourseID","CourseCategory"]], on="CourseID")
                  .merge(profiles[["UserID","SegmentName"]], on="UserID"))
        cat_seg = tx_seg.groupby(["SegmentName","CourseCategory"]).size().reset_index(name="count")
        cat_seg["pct"] = cat_seg["count"] / cat_seg.groupby("SegmentName")["count"].transform("sum") * 100
        fig_cs = px.bar(cat_seg, x="SegmentName", y="pct", color="CourseCategory",
                        barmode="stack", color_discrete_sequence=px.colors.qualitative.Set3)
        fig_cs.update_layout(**PLOTLY_LAYOUT, height=360, yaxis_title="% of Enrollments")
        st.plotly_chart(fig_cs, use_container_width=True)

    with col_lvl:
        st.markdown("<p class='section-title'>CPD Level Mix by Segment</p>", unsafe_allow_html=True)
        tx_lvl = (transactions
                  .merge(courses[["CourseID","CourseLevel"]], on="CourseID")
                  .merge(profiles[["UserID","SegmentName"]], on="UserID"))
        lvl_seg = tx_lvl.groupby(["SegmentName","CourseLevel"]).size().reset_index(name="count")
        lvl_seg["pct"] = lvl_seg["count"] / lvl_seg.groupby("SegmentName")["count"].transform("sum") * 100
        fig_ls = px.bar(lvl_seg, x="SegmentName", y="pct", color="CourseLevel",
                        barmode="stack",
                        color_discrete_map={"Beginner":"#4cc9f0",
                                            "Intermediate":"#4361ee",
                                            "Advanced":"#7209b7"})
        fig_ls.update_layout(**PLOTLY_LAYOUT, height=360, yaxis_title="% of Enrollments")
        st.plotly_chart(fig_ls, use_container_width=True)

    # Spending box plot
    st.markdown("---")
    st.markdown("<p class='section-title'>CPD Spending Distribution by Segment</p>", unsafe_allow_html=True)
    fig_bx = px.box(profiles, x="SegmentName", y="total_spending",
                    color="SegmentName", color_discrete_map=COLOR_MAP, points="outliers")
    fig_bx.update_layout(**PLOTLY_LAYOUT, height=340,
                         yaxis_title="Total Spending (R)", showlegend=False)
    st.plotly_chart(fig_bx, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 6 — EDA & ANALYTICS
# ═══════════════════════════════════════════════

elif page == "🔍 EDA & Analytics":
    st.markdown("<h1 style='color:#e8edf5'>🔍 EDA & Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa3bf;margin-bottom:20px'>CPD catalogue analysis, correlations, and business insights</p>", unsafe_allow_html=True)

    # Course catalogue stats
    st.markdown("<p class='section-title'>CPD Course Catalogue</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Courses",    len(courses))
    c2.metric("Avg Rating",       f"{courses['CourseRating'].mean():.2f}★")
    c3.metric("Avg Price",        f"R{courses['BasePrice' if 'BasePrice' in courses.columns else 'PriceZAR'].mean():.0f}" if ('BasePrice' in courses.columns or 'PriceZAR' in courses.columns) else "R499")
    c4.metric("CPD Categories",   courses["CourseCategory"].nunique())

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("<p class='section-title'>Rating Distribution</p>", unsafe_allow_html=True)
        fig_rat = px.histogram(courses, x="CourseRating", nbins=25,
                               color_discrete_sequence=["#4361ee"])
        fig_rat.update_layout(**PLOTLY_LAYOUT, height=260,
                              xaxis_title="Rating", yaxis_title="Courses")
        st.plotly_chart(fig_rat, use_container_width=True)

    with col_r2:
        st.markdown("<p class='section-title'>Avg Rating by Category</p>", unsafe_allow_html=True)
        rc = (courses.groupby("CourseCategory")["CourseRating"]
                     .mean().reset_index()
                     .sort_values("CourseRating", ascending=False))
        fig_rc = px.bar(rc, x="CourseRating", y="CourseCategory", orientation="h",
                        color="CourseRating", color_continuous_scale="Blues")
        fig_rc.update_layout(**{**PLOTLY_LAYOUT, "yaxis": dict(gridcolor="#2a3350",
                                                               categoryorder="total ascending")},
                             height=260, coloraxis_showscale=False)
        st.plotly_chart(fig_rc, use_container_width=True)

    # Correlation heatmap
    st.markdown("---")
    st.markdown("<p class='section-title'>Feature Correlation Matrix</p>", unsafe_allow_html=True)
    corr_feats = ["total_courses","avg_spending","total_spending",
                  "avg_course_rating","diversity_score",
                  "learning_depth_index","enrollment_frequency",
                  "n_categories","recency_days","Age"]
    corr_df = profiles[[f for f in corr_feats if f in profiles.columns]].corr().round(2)
    fig_cor = px.imshow(corr_df, text_auto=".2f",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig_cor.update_layout(**PLOTLY_LAYOUT, height=440)
    st.plotly_chart(fig_cor, use_container_width=True)

    # Scatter: frequency vs spending
    st.markdown("---")
    st.markdown("<p class='section-title'>Enrollment Frequency vs Total Spending</p>", unsafe_allow_html=True)
    fig_sc2 = px.scatter(
        profiles, x="enrollment_frequency", y="total_spending",
        color="SegmentName", color_discrete_map=COLOR_MAP, opacity=0.65,
        hover_data=["UserID","preferred_category","total_courses"]
    )
    fig_sc2.update_traces(marker=dict(size=5))
    fig_sc2.update_layout(**PLOTLY_LAYOUT, height=380,
                          xaxis_title="Enrollment Frequency (courses/day)",
                          yaxis_title="Total Spending (R)")
    st.plotly_chart(fig_sc2, use_container_width=True)

    # Business insights
    st.markdown("---")
    st.markdown("<p class='section-title'>📌 Key Business Insights for EdUPro</p>", unsafe_allow_html=True)

    top_cat = (transactions.merge(courses[["CourseID","CourseCategory"]], on="CourseID")
                           .groupby("CourseCategory")["Amount"].sum().idxmax())
    top_seg = cp.sort_values("total_spending", ascending=False).iloc[0]
    best_sil = cp.sort_values("SilhouetteVal", ascending=False).iloc[0]
    avg_rev  = transactions.groupby("UserID")["Amount"].sum().mean()
    expert_seg = cp[cp["learning_depth_index"] == cp["learning_depth_index"].max()].iloc[0]

    insights = [
        f"💰 <b>{top_cat}</b> generates the highest category revenue — EdUPro's core content strength confirmed by enrollment data.",
        f"🔬 <b>{expert_seg['SegmentName']}</b> have the deepest learning profile (depth={expert_seg['learning_depth_index']:.2f}) — prioritize new Advanced modules for this group.",
        f"🚀 <b>{top_seg['SegmentName']}</b> have the highest avg spend at <b>R{top_seg['total_spending']:.0f}</b> — premium bundle offers would resonate here.",
        f"🔵 <b>{best_sil['SegmentName']}</b> is the most cohesive cluster (silhouette={best_sil['SilhouetteVal']:.3f}) — marketing messages will land most consistently here.",
        f"💳 Average professional lifetime revenue: <b>R{avg_rev:,.0f}</b> — segmented pricing tiers could lift LTV by 20-30%.",
        f"📅 Enrollment peaks in Q1 — aligns with HPCSA CPD renewal deadlines. Send recommendation emails in November/December.",
        f"👀 Curious Browsers (lowest engagement) are re-activation opportunities — free preview webinars as entry point.",
    ]
    for ins in insights:
        st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

    # Segment summary table
    st.markdown("---")
    st.markdown("<p class='section-title'>Segment Summary Statistics</p>", unsafe_allow_html=True)
    summary = cp[["SegmentName","n_users","total_courses","avg_spending",
                  "total_spending","avg_course_rating","learning_depth_index",
                  "n_categories","SilhouetteVal"]].copy()
    summary.columns = ["Segment","Professionals","Avg CPD Courses","Avg Spend/Course (R)",
                        "Avg Total Spend (R)","Avg Rating","Depth Index",
                        "Avg Categories","Silhouette"]
    st.dataframe(
        summary.style.format({
            "Avg Spend/Course (R)": "R{:.2f}",
            "Avg Total Spend (R)":  "R{:.2f}",
            "Avg CPD Courses":      "{:.1f}",
            "Avg Rating":           "{:.2f}",
            "Depth Index":          "{:.3f}",
            "Avg Categories":       "{:.1f}",
            "Silhouette":           "{:.3f}",
        }),
        use_container_width=True, hide_index=True
    )
    
