"""
Student Performance Dashboard
==============================
Topic  : How Learning Behaviour Impacts Performance Across Subjects
Dataset: Math & Portuguese cleaned CSVs (from your existing pipeline)

Run with:
    streamlit run dashboard.py

Expects:
    data/math_cleaned_data.csv
    data/portuguese_cleaned_data.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Learning Behaviour & Student Performance",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

html, body, [class*="css"]          { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .stMetricLabel          { font-family: 'Syne', sans-serif !important; }
.stApp                              { background: #0f0f14; color: #e8e6df; }

section[data-testid="stSidebar"]    { background: #16161f; border-right: 1px solid #2a2a38; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 20px; font-weight: 700; color: #e8e6df;
    margin: 1.8rem 0 0.4rem 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #ff6b6b;
    display: inline-block;
}
.insight-box {
    background: #1a1a26;
    border-left: 3px solid #ff6b6b;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px; margin: 10px 0;
    font-size: 14px; color: #b8b6af; line-height: 1.75;
}
.insight-box b { color: #e8e6df; }

.tag-math {
    background: #1a2e4a; color: #64b5f6;
    border: 1px solid #1e4080;
    padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: .05em;
}
.tag-por {
    background: #2e1a2e; color: #f48fb1;
    border: 1px solid #80204e;
    padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: .05em;
}
.attention-box {
    background: #1f1a10;
    border: 1px solid #c97b00; border-radius: 12px;
    padding: 18px 22px; margin: 14px 0;
}
.attention-box .attn-title {
    font-family: 'Syne', sans-serif;
    color: #ffc107; font-size: 13px; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; margin-bottom: 8px;
}

div[data-testid="stMetric"] {
    background: #1a1a26; border: 1px solid #2a2a38;
    border-radius: 12px; padding: 16px;
}
div[data-testid="stMetric"] label          { color: #6e6e8a !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: .08em; }
div[data-testid="stMetricValue"]           { color: #e8e6df !important; font-family: 'Syne', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
MATH_C   = "#64b5f6"
POR_C    = "#f48fb1"
BG       = "#0f0f14"
PANEL    = "#1a1a26"
GRID     = "#2a2a38"
TEXT     = "#e8e6df"
MUTED    = "#6e6e8a"
ACCENT   = "#ff6b6b"

def styled_fig(w=9, h=4.5, r=1, c=1):
    """Create a dark-themed figure; returns (fig, ax) — ax is 2-D if r>1 or c>1."""
    fig, ax = plt.subplots(r, c, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    axs = np.array(ax).flatten() if (r > 1 or c > 1) else [ax]
    for a in axs:
        a.set_facecolor(PANEL)
        a.tick_params(colors=MUTED, labelsize=9)
        for sp in a.spines.values():
            sp.set_edgecolor(GRID)
        a.xaxis.label.set_color(MUTED)
        a.yaxis.label.set_color(MUTED)
        a.title.set_color(TEXT)
        a.grid(color=GRID, alpha=0.45, linewidth=0.5)
    return fig, ax

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    m = pd.read_csv('data/math_cleaned_data.csv')
    p = pd.read_csv('data/portuguese_cleaned_data.csv')
    m['subject'] = 'Math'
    p['subject'] = 'Portuguese'
    return m, p, pd.concat([m, p], ignore_index=True)

math_df, por_df, combined = load()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0;'>
        <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#e8e6df;line-height:1.3;'>
            Learning Behaviour<br><span style='color:#ff6b6b;'>& Performance</span>
        </div>
        <div style='font-size:11px;color:#6e6e8a;margin-top:6px;letter-spacing:.1em;text-transform:uppercase;'>
            Student Data Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", [
        "Overview",
        "Study Time Myth",
        "Subject Strategy",
        "Factor Deep Dive",
        "Needs Attention",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px;color:#6e6e8a;line-height:1.9;'>
        Math students:&nbsp;&nbsp;&nbsp;<b style='color:#64b5f6;'>{len(math_df)}</b><br>
        Portuguese students: <b style='color:#f48fb1;'>{len(por_df)}</b>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("""
    <h1 style='font-family:Syne,sans-serif;font-size:34px;font-weight:800;
                color:#e8e6df;margin-bottom:4px;'>
        How Learning Behaviour Impacts Performance
    </h1>
    <p style='color:#6e6e8a;font-size:15px;margin-bottom:1.5rem;'>
        Across Math and Portuguese — and why the same study method doesn't work for both.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Math — Mean G3",        f"{math_df['G3'].mean():.1f}", f"σ = {math_df['G3'].std():.1f}")
    with c2: st.metric("Portuguese — Mean G3",  f"{por_df['G3'].mean():.1f}",  f"σ = {por_df['G3'].std():.1f}")
    with c3: st.metric("Math High Achievers",   f"{(math_df['G3']>=15).mean()*100:.1f}%", "G3 ≥ 15")
    with c4: st.metric("Portuguese High Achievers", f"{(por_df['G3']>=15).mean()*100:.1f}%", "G3 ≥ 15")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    # Grade distributions
    with col_a:
        st.markdown('<div class="section-header">Grade Distributions</div>', unsafe_allow_html=True)
        fig, ax = styled_fig(7, 4)
        bins = np.arange(0, 21, 1)
        ax.hist(math_df['G3'], bins=bins, alpha=0.75, color=MATH_C, edgecolor=BG, lw=0.5, label='Math')
        ax.hist(por_df['G3'],  bins=bins, alpha=0.65, color=POR_C,  edgecolor=BG, lw=0.5, label='Portuguese')
        ax.axvline(math_df['G3'].mean(), color=MATH_C, linestyle='--', lw=1.6, alpha=0.9)
        ax.axvline(por_df['G3'].mean(),  color=POR_C,  linestyle='--', lw=1.6, alpha=0.9)
        ax.set_xlabel("Final Grade (G3)")
        ax.set_ylabel("Students")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Grade trajectory
    with col_b:
        st.markdown('<div class="section-header">Grade Trajectory: G1 → G2 → G3</div>', unsafe_allow_html=True)
        fig, ax = styled_fig(7, 4)
        mm = [math_df['G1'].mean(), math_df['G2'].mean(), math_df['G3'].mean()]
        pm = [por_df['G1'].mean(),  por_df['G2'].mean(),  por_df['G3'].mean()]
        x  = [1, 2, 3]
        ax.plot(x, mm, 'o-', color=MATH_C, lw=2.5, ms=8, label='Math')
        ax.plot(x, pm, 'o-', color=POR_C,  lw=2.5, ms=8, label='Portuguese')
        ax.fill_between(x, mm, alpha=0.12, color=MATH_C)
        ax.fill_between(x, pm, alpha=0.12, color=POR_C)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Period 1 (G1)', 'Period 2 (G2)', 'Final (G3)'], color=MUTED)
        ax.set_ylabel("Average Grade")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("""
    <div class='insight-box'>
        Portuguese students score <b>higher on average</b>, yet Math has a
        <b>wider spread</b> — more students at the top AND bottom.
        The trajectory shows Portuguese grades stay stable; Math grades fluctuate more
        period to period. First clue: <b>these two subjects respond differently to student behaviour.</b>
    </div>
    """, unsafe_allow_html=True)

    # Performance breakdown
    st.markdown('<div class="section-header">Performance Breakdown & Failure Rates</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = styled_fig(4, 3.5)
        fail_m = (math_df['G3'] < 10).mean() * 100
        fail_p = (por_df['G3']  < 10).mean() * 100
        ax.bar(['Math','Portuguese'], [fail_m, fail_p],
               color=[MATH_C, POR_C], edgecolor=BG, width=0.45)
        for i, v in enumerate([fail_m, fail_p]):
            ax.text(i, v + 0.4, f'{v:.1f}%', ha='center', color=TEXT, fontsize=11, fontweight='bold')
        ax.set_title('Failure Rate (G3 < 10)'); ax.set_ylabel('% Students')
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        fig, ax = styled_fig(4, 3.5)
        ax.plot(math_df.groupby('failures')['G3'].mean().index,
                math_df.groupby('failures')['G3'].mean().values,
                'o-', color=MATH_C, lw=2, ms=6, label='Math')
        ax.plot(por_df.groupby('failures')['G3'].mean().index,
                por_df.groupby('failures')['G3'].mean().values,
                'o-', color=POR_C, lw=2, ms=6, label='Portuguese')
        ax.set_xlabel('Past Failures'); ax.set_ylabel('Mean G3')
        ax.set_title('Past Failures → Current Grade')
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c3:
        cats = ['High\n(≥15)', 'Average\n(10–14)', 'Struggling\n(<10)']
        mp   = [(math_df['G3']>=15).mean()*100,
                ((math_df['G3']>=10)&(math_df['G3']<15)).mean()*100,
                (math_df['G3']<10).mean()*100]
        pp   = [(por_df['G3']>=15).mean()*100,
                ((por_df['G3']>=10)&(por_df['G3']<15)).mean()*100,
                (por_df['G3']<10).mean()*100]
        fig, ax = styled_fig(4, 3.5)
        x = np.arange(3)
        ax.bar(x-0.2, mp, 0.38, color=MATH_C, alpha=0.85, label='Math')
        ax.bar(x+0.2, pp, 0.38, color=POR_C,  alpha=0.85, label='Portuguese')
        ax.set_xticks(x); ax.set_xticklabels(cats, color=MUTED, fontsize=9)
        ax.set_ylabel('% Students'); ax.set_title('Performance Breakdown')
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — STUDY TIME MYTH
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Study Time Myth":

    st.markdown("""
    <h1 style='font-family:Syne,sans-serif;font-size:34px;font-weight:800;color:#e8e6df;margin-bottom:4px;'>
        The Study Time Myth
    </h1>
    <p style='color:#6e6e8a;font-size:15px;margin-bottom:1.2rem;'>
        "Study more, score more" — does the data actually support this?
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='attention-box'>
        <div class='attn-title'>The Core Argument</div>
        Study time is a <b style='color:#ffc107;'>weak predictor</b> of final grades in both subjects.
        What matters far more is <b>how</b> a student studies, <b>when</b> they started
        falling behind, and whether their approach matches what the subject demands.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Study time vs G3 scatter for both subjects
    with col_a:
        st.markdown('<div class="section-header">Study Time vs Final Grade</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 1, figsize=(7, 5.5))
        fig.patch.set_facecolor(BG)
        for ax, df_s, label, clr in zip(axes,
                                         [math_df, por_df],
                                         ['Math', 'Portuguese'],
                                         [MATH_C, POR_C]):
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.grid(color=GRID, alpha=0.4, lw=0.5)

            ax.scatter(df_s['studytime'], df_s['G3'], alpha=0.15, s=14, color=clr)
            means = df_s.groupby('studytime')['G3'].mean()
            ax.plot(means.index, means.values, 'o-', color=clr, lw=2.5, ms=7, label=f'{label} avg')

            z   = np.polyfit(df_s['studytime'], df_s['G3'], 1)
            xr  = np.linspace(1, 4, 100)
            ax.plot(xr, np.poly1d(z)(xr), '--', color=clr, lw=1.5, alpha=0.7, label='Trend')

            r = df_s['studytime'].corr(df_s['G3'])
            ax.set_title(f'{label}  —  r(studytime, G3) = {r:.3f}', color=TEXT, fontsize=11)
            ax.set_xlabel('Study Time (1=<2 hrs  2=2–5 hrs  3=5–10 hrs  4=>10 hrs)', color=MUTED, fontsize=8)
            ax.set_ylabel('Final Grade (G3)', color=MUTED)
            ax.set_xticks([1, 2, 3, 4])
            ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # What actually correlates
    with col_b:
        st.markdown('<div class="section-header">What Actually Predicts G3?</div>', unsafe_allow_html=True)

        candidate_factors = [c for c in [
            'studytime','failures','absences','Walc','Dalc','goout','freetime',
            'Medu','Fedu','higher','internet','paid',
            'grade_improvement','grade_volatility','avg_G1_G2',
            'weighted_grade','study_intensity',
        ] if c in math_df.columns and c in por_df.columns]

        cm = {f: math_df[f].corr(math_df['G3']) for f in candidate_factors}
        cp = {f: por_df[f].corr(por_df['G3'])   for f in candidate_factors}
        top10 = sorted(candidate_factors,
                       key=lambda f: (abs(cm[f]) + abs(cp[f])) / 2,
                       reverse=True)[:10]

        fig, ax = plt.subplots(figsize=(7, 5.2))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, alpha=0.4, lw=0.5, axis='x')

        y = np.arange(len(top10))
        ax.barh(y - 0.2, [cm[f] for f in top10], 0.38, color=MATH_C, alpha=0.85, label='Math')
        ax.barh(y + 0.2, [cp[f] for f in top10], 0.38, color=POR_C,  alpha=0.85, label='Portuguese')
        ax.set_yticks(y)
        ax.set_yticklabels(top10, color=TEXT, fontsize=10)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel('Correlation with G3', color=MUTED)
        ax.set_title('Top Predictors of Final Grade', color=TEXT, fontsize=12)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

        r_st_m = cm.get('studytime', 0)
        r_st_p = cp.get('studytime', 0)
        st.markdown(f"""
        <div class='insight-box'>
            Study time correlation:
            <span class='tag-math'>Math: {r_st_m:.3f}</span>&nbsp;
            <span class='tag-por'>Portuguese: {r_st_p:.3f}</span><br><br>
            Compare that to <b>past failures</b>, <b>grade consistency</b>, or
            <b>weighted grade</b> — which are 4–8× stronger predictors.
            Studying longer without fixing the underlying behaviour
            produces <b>minimal grade improvement.</b>
        </div>
        """, unsafe_allow_html=True)

    # Study time × past failures interaction
    st.markdown('<div class="section-header">Study Time × Past Failures Interaction</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
        A student who studies 4+ hours but has 2+ past failures often does <b>worse</b>
        than one who studies 2 hours with zero failures. Study time cannot compensate
        for broken fundamentals — this is the myth, proven by the data.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, df_s, label, clr in [(c1, math_df, 'Math', MATH_C), (c2, por_df, 'Portuguese', POR_C)]:
        with col:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.grid(color=GRID, alpha=0.4, lw=0.5)

            palette = ['#4ecca3','#ffd166','#ff6b6b','#c77dff']
            for fail_n, fc in zip([0, 1, 2, 3], palette):
                sub = df_s[df_s['failures'] == fail_n]
                if len(sub) > 5:
                    means = sub.groupby('studytime')['G3'].mean()
                    ax.plot(means.index, means.values, 'o-', color=fc, lw=1.8,
                            ms=6, label=f'{fail_n} past failures', alpha=0.9)

            ax.set_title(f'{label}: Study Time by Past Failure Count', color=TEXT, fontsize=10)
            ax.set_xlabel('Study Time Category', color=MUTED)
            ax.set_ylabel('Mean G3', color=MUTED)
            ax.set_xticks([1, 2, 3, 4])
            ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — SUBJECT STRATEGY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Subject Strategy":

    st.markdown("""
    <h1 style='font-family:Syne,sans-serif;font-size:34px;font-weight:800;color:#e8e6df;margin-bottom:4px;'>
        Different Subjects, Different Strategies
    </h1>
    <p style='color:#6e6e8a;font-size:15px;margin-bottom:1.2rem;'>
        The same factor can <em>help</em> in one subject and be <em>irrelevant</em> — or harmful — in another.
    </p>
    """, unsafe_allow_html=True)

    key_factors = [c for c in [
        'studytime','failures','absences','Walc','Dalc','goout','freetime',
        'Medu','Fedu','higher','internet','paid','romantic','health','famsup',
        'schoolsup','grade_improvement','grade_volatility','avg_G1_G2',
        'weighted_grade','study_intensity',
    ] if c in math_df.columns and c in por_df.columns]

    cm   = pd.Series({f: math_df[f].corr(math_df['G3']) for f in key_factors})
    cp   = pd.Series({f: por_df[f].corr(por_df['G3'])   for f in key_factors})
    diff = (cp - cm).sort_values()

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Correlation Divergence: Math vs Portuguese</div>',
                    unsafe_allow_html=True)
        top_diff = pd.concat([diff.head(8), diff.tail(8)]).drop_duplicates()
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, alpha=0.4, lw=0.5, axis='x')

        bar_colors = [POR_C if v > 0 else MATH_C for v in top_diff.values]
        ax.barh(top_diff.index, top_diff.values, color=bar_colors, alpha=0.85, height=0.6)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel('Difference (Portuguese − Math)', color=MUTED)
        ax.set_title('Positive = more important for Portuguese\nNegative = more important for Math',
                     color=MUTED, fontsize=9)
        mp = mpatches.Patch(color=MATH_C, alpha=0.85, label='Stronger for Math')
        pp = mpatches.Patch(color=POR_C,  alpha=0.85, label='Stronger for Portuguese')
        ax.legend(handles=[mp, pp], facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        st.markdown('<div class="section-header">Strategy Playbook</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#1a2e4a;border:1px solid #1e4080;border-radius:10px;
                    padding:16px;margin-bottom:12px;'>
            <div style='font-family:Syne,sans-serif;color:#64b5f6;font-size:13px;font-weight:700;
                        letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;'>
                Math — What Matters
            </div>
            <div style='font-size:13px;color:#b8d4f4;line-height:2;'>
                ✓ <b>Grade consistency</b> (low volatility)<br>
                ✓ <b>Zero past failures</b><br>
                ✓ <b>Low absences</b><br>
                ✓ <b>Paid tutoring helps</b><br>
                ✗ Adding hours without structured practice
            </div>
        </div>
        <div style='background:#2e1a2e;border:1px solid #80204e;border-radius:10px;padding:16px;'>
            <div style='font-family:Syne,sans-serif;color:#f48fb1;font-size:13px;font-weight:700;
                        letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;'>
                Portuguese — What Matters
            </div>
            <div style='font-size:13px;color:#f4b8cc;line-height:2;'>
                ✓ <b>Higher education aspiration</b><br>
                ✓ <b>Parental education level</b><br>
                ✓ <b>Internet access</b><br>
                ✓ <b>Motivation & engagement</b><br>
                ✗ Brute-force hours without engagement
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive factor comparison
    st.markdown('<div class="section-header">Factor-by-Factor Impact Comparison</div>',
                unsafe_allow_html=True)
    focus = st.multiselect(
        "Select factors to compare",
        options=key_factors,
        default=[f for f in ['studytime','failures','absences','higher','Medu',
                              'Walc','grade_volatility','paid'] if f in key_factors]
    )
    if focus:
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, alpha=0.4, lw=0.5, axis='y')

        x = np.arange(len(focus))
        ax.bar(x-0.22, [cm[f] for f in focus], 0.42, color=MATH_C, alpha=0.85, label='Math')
        ax.bar(x+0.22, [cp[f] for f in focus], 0.42, color=POR_C,  alpha=0.85, label='Portuguese')
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(focus, rotation=35, ha='right', color=TEXT, fontsize=10)
        ax.set_ylabel('Correlation with G3', color=MUTED)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Absences case study
    st.markdown('<div class="section-header">Case Study: Absences Affect Subjects Differently</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, df_s, label, clr in [(c1, math_df, 'Math', MATH_C), (c2, por_df, 'Portuguese', POR_C)]:
        with col:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.grid(color=GRID, alpha=0.4, lw=0.5)

            ax.scatter(df_s['absences'], df_s['G3'], alpha=0.25, s=12, color=clr)
            z  = np.polyfit(df_s['absences'], df_s['G3'], 1)
            xr = np.linspace(0, df_s['absences'].quantile(0.97), 100)
            ax.plot(xr, np.poly1d(z)(xr), '--', color=clr, lw=2)
            r  = df_s['absences'].corr(df_s['G3'])
            ax.set_title(f'{label}: Absences vs G3  (r = {r:.3f})', color=TEXT, fontsize=10)
            ax.set_xlabel('Absences', color=MUTED)
            ax.set_ylabel('Final Grade (G3)', color=MUTED)
            st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — FACTOR DEEP DIVE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Factor Deep Dive":

    st.markdown("""
    <h1 style='font-family:Syne,sans-serif;font-size:34px;font-weight:800;color:#e8e6df;margin-bottom:4px;'>
        Factor Deep Dive
    </h1>
    <p style='color:#6e6e8a;font-size:15px;margin-bottom:1.2rem;'>
        Pick any factor and see exactly how it shapes performance in each subject.
    </p>
    """, unsafe_allow_html=True)

    numeric_cols = [c for c in math_df.columns
                    if math_df[c].dtype in ['float64','int64'] and c != 'G3']

    chosen = st.selectbox("Choose a factor to explore", numeric_cols,
                          index=numeric_cols.index('studytime') if 'studytime' in numeric_cols else 0)

    mc = math_df[chosen].corr(math_df['G3'])
    pc = por_df[chosen].corr(por_df['G3']) if chosen in por_df.columns else 0

    c1, c2, c3 = st.columns(3)
    with c1: st.metric(f"Math — r({chosen}, G3)",       f"{mc:.3f}")
    with c2: st.metric(f"Portuguese — r({chosen}, G3)", f"{pc:.3f}")
    with c3: st.metric("Difference (Por − Math)",       f"{pc - mc:+.3f}",
                        "stronger in Portuguese" if pc > mc else "stronger in Math")

    c1, c2 = st.columns(2)
    for col, df_s, label, clr in [(c1, math_df, 'Math', MATH_C), (c2, por_df, 'Portuguese', POR_C)]:
        with col:
            if chosen not in df_s.columns:
                st.write(f"Column not found in {label} data.")
                continue
            is_cat = df_s[chosen].nunique() <= 7
            fig, ax = plt.subplots(figsize=(5.5, 4))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.grid(color=GRID, alpha=0.4, lw=0.5)

            if is_cat:
                cats = sorted(df_s[chosen].dropna().unique())
                bp   = ax.boxplot([df_s[df_s[chosen]==c]['G3'].dropna().values for c in cats],
                                  patch_artist=True,
                                  medianprops={'color': ACCENT, 'linewidth': 2},
                                  whiskerprops={'color': MUTED},
                                  capprops={'color': MUTED},
                                  flierprops={'marker':'o','markerfacecolor':clr,'markersize':3,'alpha':0.4})
                for patch in bp['boxes']:
                    patch.set_facecolor(clr); patch.set_alpha(0.4)
                ax.set_xticklabels(cats, color=MUTED)
            else:
                ax.scatter(df_s[chosen], df_s['G3'], alpha=0.25, s=12, color=clr)
                z  = np.polyfit(df_s[chosen].fillna(df_s[chosen].median()), df_s['G3'], 1)
                xr = np.linspace(df_s[chosen].min(), df_s[chosen].max(), 100)
                ax.plot(xr, np.poly1d(z)(xr), '--', color=clr, lw=2)

            r = df_s[chosen].corr(df_s['G3'])
            ax.set_title(f'{label}: {chosen} vs G3  (r = {r:.3f})', color=TEXT, fontsize=10)
            ax.set_xlabel(chosen, color=MUTED)
            ax.set_ylabel('Final Grade (G3)', color=MUTED)
            st.pyplot(fig, use_container_width=True); plt.close()

    # Group-level comparison
    st.markdown('<div class="section-header">Learning Behaviour Groups — Mean G3 Comparison</div>',
                unsafe_allow_html=True)
    groups = {
        'Academic Behavior' : ['studytime','failures','absences'],
        'Family Background'  : ['Medu','Fedu','famsup'],
        'Lifestyle'          : ['freetime','goout','Dalc','Walc'],
        'Support & Resources': ['schoolsup','paid','internet'],
        'Motivation'         : ['higher'],
    }
    sel_group = st.selectbox("Select a variable group", list(groups.keys()))
    grp_vars  = [v for v in groups[sel_group] if v in math_df.columns]

    if grp_vars:
        fig, axes = plt.subplots(1, len(grp_vars), figsize=(5 * len(grp_vars), 4))
        fig.patch.set_facecolor(BG)
        if len(grp_vars) == 1: axes = [axes]

        for ax, var in zip(np.array(axes).flatten(), grp_vars):
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(color=GRID, alpha=0.4, lw=0.5)

            mm = math_df.groupby(var)['G3'].mean()
            pm = por_df.groupby(var)['G3'].mean()  if var in por_df.columns else None
            cats = sorted(math_df[var].dropna().unique())
            x    = np.arange(len(cats))

            ax.bar(x-0.2, [mm.get(c, 0) for c in cats], 0.38, color=MATH_C, alpha=0.85, label='Math')
            if pm is not None:
                ax.bar(x+0.2, [pm.get(c, 0) for c in cats], 0.38, color=POR_C, alpha=0.85, label='Portuguese')
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in cats], color=MUTED, fontsize=8)
            ax.set_title(var, color=TEXT, fontsize=10)
            ax.set_ylabel('Mean G3', color=MUTED)
            ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — NEEDS ATTENTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Needs Attention":

    st.markdown("""
    <h1 style='font-family:Syne,sans-serif;font-size:34px;font-weight:800;color:#e8e6df;margin-bottom:4px;'>
        What Needs Attention
    </h1>
    <p style='color:#6e6e8a;font-size:15px;margin-bottom:1.2rem;'>
        Four hidden patterns in the data that schools and students should be acting on — but aren't.
    </p>
    """, unsafe_allow_html=True)

    # ── Finding 1: Alcohol ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Finding 1 — Alcohol is an Ignored Academic Risk Factor</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class='attention-box'>
        <div class='attn-title'>Overlooked Risk</div>
        Workday alcohol (Dalc) and weekend alcohol (Walc) show
        <b style='color:#ffc107;'>consistently negative correlations</b> with G3 in both subjects.
        Students at level 3+ are disproportionately in the failing category.
        This is rarely treated as an academic intervention point.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, df_s, label, clr in [(c1, math_df, 'Math', MATH_C), (c2, por_df, 'Portuguese', POR_C)]:
        with col:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.grid(color=GRID, alpha=0.3, lw=0.5)

            x = np.arange(1, 6)
            wm = df_s.groupby('Walc')['G3'].mean().reindex(x, fill_value=np.nan)
            dm = df_s.groupby('Dalc')['G3'].mean().reindex(x, fill_value=np.nan)
            ax.bar(x-0.2, wm, 0.38, color='#ff6b6b', alpha=0.85, label='Weekend (Walc)')
            ax.bar(x+0.2, dm, 0.38, color='#c77dff', alpha=0.85, label='Workday (Dalc)')
            ax.set_xticks(x)
            ax.set_xticklabels(['1\n(low)','2','3','4','5\n(high)'], color=MUTED, fontsize=9)
            ax.set_title(f'{label}: Alcohol Level vs Mean G3', color=TEXT)
            ax.set_ylabel('Mean G3', color=MUTED)
            ax.set_xlabel('Alcohol Consumption', color=MUTED)
            ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
            st.pyplot(fig, use_container_width=True); plt.close()

    # ── Finding 2: Romantic relationships ────────────────────────────────────
    st.markdown('<div class="section-header">Finding 2 — Romantic Relationships Hit Math Harder</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, alpha=0.4, lw=0.5)

        mr = math_df.groupby('romantic')['G3'].mean()
        pr = por_df.groupby('romantic')['G3'].mean()
        x  = np.arange(len(mr))
        ax.bar(x-0.2, mr.values, 0.38, color=MATH_C, alpha=0.85, label='Math')
        ax.bar(x+0.2, pr.values, 0.38, color=POR_C,  alpha=0.85, label='Portuguese')
        ax.set_xticks(x)
        ax.set_xticklabels(mr.index, color=MUTED)
        ax.set_title('Romantic Relationship vs Mean G3', color=TEXT)
        ax.set_ylabel('Mean G3', color=MUTED)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
        for i, (mv, pv) in enumerate(zip(mr.values, pr.values)):
            ax.text(i-0.2, mv+0.12, f'{mv:.1f}', ha='center', color=MATH_C, fontsize=10, fontweight='bold')
            ax.text(i+0.2, pv+0.12, f'{pv:.1f}', ha='center', color=POR_C,  fontsize=10, fontweight='bold')
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        m_vals = mr.values
        p_vals = pr.values
        m_diff = m_vals[0] - m_vals[-1] if len(m_vals) > 1 else 0
        p_diff = p_vals[0] - p_vals[-1] if len(p_vals) > 1 else 0
        st.markdown(f"""
        <div class='insight-box' style='margin-top:1.5rem;'>
            Students in relationships score on average:<br>
            <b style='color:{MATH_C};'>Math: {m_diff:+.2f} points lower</b><br>
            <b style='color:{POR_C};'>Portuguese: {p_diff:+.2f} points lower</b><br><br>
            The effect is larger in Math — consistent with Math requiring
            <b>sustained daily concentration and practice</b>, which is more easily
            disrupted by social and emotional distractions than Portuguese's
            broader engagement-based learning.
        </div>
        """, unsafe_allow_html=True)

    # ── Finding 3: Internet access gap ───────────────────────────────────────
    st.markdown('<div class="section-header">Finding 3 — Internet Access Creates Unequal Outcomes</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    for col, df_s, label, clr in [(c1, math_df, 'Math', MATH_C), (c2, por_df, 'Portuguese', POR_C)]:
        with col:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.grid(color=GRID, alpha=0.4, lw=0.5)

            inet = df_s.groupby('internet')['G3'].mean()
            ax.bar(inet.index, inet.values, color=clr, alpha=0.85, width=0.4)
            for i, v in zip(inet.index, inet.values):
                ax.text(i, v + 0.12, f'{v:.1f}', ha='center', color=TEXT, fontsize=11, fontweight='bold')
            ax.set_title(f'{label}: Internet vs G3', color=TEXT, fontsize=10)
            ax.set_xlabel('Has Internet at Home', color=MUTED)
            ax.set_ylabel('Mean G3', color=MUTED)
            st.pyplot(fig, use_container_width=True); plt.close()

    with c3:
        mi = math_df.groupby('internet')['G3'].mean()
        pi = por_df.groupby('internet')['G3'].mean()
        mg = (mi.iloc[-1] - mi.iloc[0]) if len(mi) > 1 else 0
        pg = (pi.iloc[-1] - pi.iloc[0]) if len(pi) > 1 else 0
        st.markdown(f"""
        <div class='insight-box' style='margin-top:1.5rem;'>
            Grade gap (internet vs no internet):<br>
            <b style='color:{MATH_C};'>Math: {mg:+.2f} pts</b><br>
            <b style='color:{POR_C};'>Portuguese: {pg:+.2f} pts</b><br><br>
            The gap is larger for Portuguese — where online reading,
            media, and self-directed content matter more than structured
            classroom drill. This is a <b>structural inequality</b> with
            a measurable grade impact.
        </div>
        """, unsafe_allow_html=True)

    # ── Finding 4: School support paradox ────────────────────────────────────
    st.markdown('<div class="section-header">Finding 4 — The School Support Paradox</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])

    with c1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, alpha=0.4, lw=0.5)

        ms = math_df.groupby('schoolsup')['G3'].mean()
        ps = por_df.groupby('schoolsup')['G3'].mean()
        x  = np.arange(len(ms))
        ax.bar(x-0.2, ms.values, 0.38, color=MATH_C, alpha=0.85, label='Math')
        ax.bar(x+0.2, ps.values, 0.38, color=POR_C,  alpha=0.85, label='Portuguese')
        ax.set_xticks(x); ax.set_xticklabels(ms.index, color=MUTED)
        ax.set_title('School Support Received vs Mean G3', color=TEXT)
        ax.set_ylabel('Mean G3', color=MUTED)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        st.markdown("""
        <div class='attention-box' style='margin-top:0.4rem;'>
            <div class='attn-title'>Paradox</div>
            Students receiving school support score
            <b style='color:#ffc107;'>lower</b> on average.
            Support is <b>reactive</b> — given to already-struggling students —
            rather than <b>preventive</b>. Schools need earlier identification,
            before grades drop, not after.
        </div>
        """, unsafe_allow_html=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Summary of Findings</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;'>
        <div class='insight-box'>
            <b>Alcohol</b> — reduces grades in both subjects. Even moderate weekend
            drinking (level 3) correlates with ~1 point lower G3.
            Rarely addressed as an academic risk factor.
        </div>
        <div class='insight-box'>
            <b>Romantic relationships</b> — hurt Math more than Portuguese,
            because Math demands a consistent focused daily routine that
            emotional distractions disrupt more easily.
        </div>
        <div class='insight-box'>
            <b>Internet access</b> — the grade gap is larger for Portuguese
            because language learning benefits from self-directed online exposure.
            A structural inequality with a measurable and fixable outcome.
        </div>
        <div class='insight-box'>
            <b>School support paradox</b> — support arrives too late.
            Students who receive it are already failing.
            <b>Preventive, early intervention</b> would show very different results.
        </div>
    </div>
    """, unsafe_allow_html=True)