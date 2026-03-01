import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import io
import warnings
import json
warnings.filterwarnings("ignore")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AUBMC Physician Performance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #2563eb;
        margin-bottom: 8px;
    }
    .metric-card.warning { border-left-color: #f59e0b; }
    .metric-card.danger  { border-left-color: #ef4444; }
    .metric-card.success { border-left-color: #10b981; }
    .metric-card.neutral { border-left-color: #6366f1; }
    .metric-label { font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 32px; font-weight: 700; color: #111827; line-height: 1.2; }
    .metric-sub   { font-size: 12px; color: #9ca3af; margin-top: 2px; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #1f2937;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 8px; margin-bottom: 16px;
    }
    .pill-red    { background:#fef2f2; color:#dc2626; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
    .pill-yellow { background:#fffbeb; color:#d97706; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
    .pill-green  { background:#f0fdf4; color:#16a34a; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
    .pill-grey   { background:#f3f4f6; color:#6b7280; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
    div[data-testid="stSidebarNav"] { display: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: white; border-radius: 8px 8px 0 0;
        font-weight: 600; color: #374151;
    }
    .stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }
    .comment-card {
        background: white; border-radius: 10px; padding: 14px 18px;
        margin-bottom: 10px; border-left: 3px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .comment-card.neg { border-left-color: #ef4444; }
    .comment-card.pos { border-left-color: #10b981; }
    .comment-card.neu { border-left-color: #d1d5db; }
</style>
""", unsafe_allow_html=True)

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────
RATING_SCALE = {
    "Always": 4, "Most of the time": 3,
    "Sometimes": 2, "Hardly ever": 1, "Never": 0, "nan": np.nan
}

def clean_question_col(col):
    if not isinstance(col, str): return col
    c = col.strip()
    c = (c.replace("â€™","'").replace("â€¦","...").replace("\r"," ").replace("\n"," "))
    c = re.sub(r"\s+", " ", c).strip()
    if c == "Subject ID":            return "physician_id"
    if c == "Raters Group":          return "raters_group"
    if c == "Completed by":          return "completed_by"
    if c == "Rater Name":            return "rater_name"
    if c.startswith("Fillout Date"): return "fillout_date"
    if c.startswith("Q2_Comments"):  return "comments"
    if not c.startswith("Q1_"):      return c
    m = re.search(r"_(.*?)_First Scale", c)
    core = m.group(1) if m else c
    core = core.lower()
    core = re.sub(r"[^a-z0-9]+", "_", core).strip("_")
    return f"q_{core}"

def clean_headers(df):
    df = df.copy()
    df.columns = [clean_question_col(c) for c in df.columns]
    return df

def map_ratings(df):
    df = df.copy()
    q_cols = [c for c in df.columns if c.startswith("q_")]
    for c in q_cols:
        s = df[c].astype(str).str.strip()
        mapped  = s.map(RATING_SCALE)
        numeric = pd.to_numeric(s, errors="coerce")
        df[c] = numeric.where(numeric.notna(), mapped)
    return df

def compute_score(df):
    q_cols = [c for c in df.columns if c.startswith("q_")]
    df["overall_score"] = df[q_cols].mean(axis=1, skipna=True)
    return df

def add_year(df):
    if "fillout_date" in df.columns:
        df["year"] = pd.to_datetime(df["fillout_date"], errors="coerce").dt.year
    return df

def aggregate_physician(df):
    return (
        df.groupby("physician_id", as_index=False)
        .agg(avg_behavior_score=("overall_score","mean"),
             n_forms=("overall_score","count"))
    )

def add_outlier_flags(phys_df):
    df = phys_df.copy()
    scores   = df["avg_behavior_score"]
    pop_mean = scores.mean()
    pop_std  = scores.std(ddof=0)
    df["z_score"]         = (scores - pop_mean) / pop_std
    df["low_z_outlier"]   = df["z_score"] <= -2
    Q1, Q3                = scores.quantile(0.25), scores.quantile(0.75)
    IQR                   = Q3 - Q1
    df["low_iqr_outlier"] = scores < (Q1 - 1.5 * IQR)
    df["low_bottom10"]    = scores <= scores.quantile(0.10)
    return df, pop_mean, pop_std

vader = SentimentIntensityAnalyzer()

def score_vader(text, threshold=-0.05):
    try:
        s = vader.polarity_scores(str(text))
        c = s["compound"]
        label = "POSITIVE" if c >= abs(threshold) else ("NEGATIVE" if c <= threshold else "NEUTRAL")
        return {"compound": c, "sentiment": label,
                "vader_pos": s["pos"], "vader_neg": s["neg"]}
    except:
        return {"compound": 0.0, "sentiment": "NEUTRAL", "vader_pos": 0.0, "vader_neg": 0.0}

def run_sentiment(df, threshold=-0.05):
    df_s = df[
        (df.get("raters_group", pd.Series(dtype=str)) != "Faculty Self-Evaluation") &
        (df["comments"].notna()) &
        (df["comments"].astype(str).str.strip() != "")
    ].copy()
    df_s["comments"] = df_s["comments"].astype(str).str.strip()
    results = df_s["comments"].apply(lambda t: score_vader(t, threshold))
    df_s = pd.concat([df_s, pd.DataFrame(results.tolist(), index=df_s.index)], axis=1)
    return df_s

def sentiment_summary(df_sent, min_comments=5):
    s = (
        df_sent.assign(is_neg=(df_sent["sentiment"]=="NEGATIVE"))
        .groupby("physician_id", as_index=False)
        .agg(total_comments=("is_neg","count"),
             negative_comments=("is_neg","sum"),
             avg_compound=("compound","mean"))
    )
    s["negative_ratio"] = s["negative_comments"] / s["total_comments"]
    Q1, Q3 = s["negative_ratio"].quantile(0.25), s["negative_ratio"].quantile(0.75)
    ub = Q3 + 1.5*(Q3-Q1)
    s["negative_outlier"] = (s["negative_ratio"] > ub) & (s["total_comments"] >= min_comments)
    return s

def merge_sentiment(phys_df, sent_s):
    out = phys_df.merge(
        sent_s[["physician_id","total_comments","negative_ratio","avg_compound","negative_outlier"]],
        on="physician_id", how="left"
    )
    out["negative_outlier"] = out["negative_outlier"].fillna(False)
    return out

def add_risk(phys_df):
    df = phys_df.copy()
    # Dimension 1 — Behaviour Score: flagged if ANY 2 of 3 methods agree
    score_signals = (
        df["low_iqr_outlier"].astype(int) +
        df["low_z_outlier"].astype(int) +
        df["low_bottom10"].astype(int)
    )
    df["behaviour_flag"] = score_signals >= 2
    # Dimension 2 — Patient Experience: sentiment flag (complaints x sentiment handled in Tab 6)
    # Here we use negative_outlier from survey comments as the patient experience signal
    df["experience_flag"] = df["negative_outlier"].fillna(False)
    # Risk score = sum of 2 dimensions (0, 1, or 2)
    df["risk_score"] = df["behaviour_flag"].astype(int) + df["experience_flag"].astype(int)
    df["final_flag"] = df["risk_score"] == 2
    return df

def risk_pill(score):
    if score == 2:   return '<span class="pill-red">⚠ Priority</span>'
    if score == 1:   return '<span class="pill-yellow">👁 Monitor</span>'
    return '<span class="pill-green">✓ Clear</span>'

def process_dept(df_raw, dept_name, threshold=-0.05):
    df = clean_headers(df_raw)
    df = map_ratings(df)
    df = compute_score(df)
    df = add_year(df)
    phys = aggregate_physician(df)
    phys, mean, std = add_outlier_flags(phys)
    sent_raw = run_sentiment(df, threshold) if "comments" in df.columns else pd.DataFrame()
    if not sent_raw.empty:
        sent_s = sentiment_summary(sent_raw)
        phys   = merge_sentiment(phys, sent_s)
    else:
        phys["total_comments"]   = 0
        phys["negative_ratio"]   = np.nan
        phys["avg_compound"]     = np.nan
        phys["negative_outlier"] = False
    phys = add_risk(phys)
    phys["department"] = dept_name
    return df, phys, sent_raw

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 AUBMC Dashboard")
    st.markdown("**Physician Behaviour Performance**")
    st.markdown("---")
    st.markdown("### 📁 Upload Data Files")
    st.markdown("Upload CSV exports from BLUE Explorance for each department and year.")

    with st.expander("AUBMC General", expanded=True):
        f_aubmc_23 = st.file_uploader("AUBMC 2023", type="csv", key="a23")
        f_aubmc_24 = st.file_uploader("AUBMC 2024", type="csv", key="a24")
        f_aubmc_25 = st.file_uploader("AUBMC 2025", type="csv", key="a25")

    with st.expander("Emergency Department", expanded=False):
        f_ed_23 = st.file_uploader("ED 2023", type="csv", key="e23")
        f_ed_24 = st.file_uploader("ED 2024", type="csv", key="e24")
        f_ed_25 = st.file_uploader("ED 2025", type="csv", key="e25")

    with st.expander("Pathology & Lab", expanded=False):
        f_patho_23 = st.file_uploader("Patho 2023", type="csv", key="p23")
        f_patho_24 = st.file_uploader("Patho 2024", type="csv", key="p24")
        f_patho_25 = st.file_uploader("Patho 2025", type="csv", key="p25")

    st.markdown("---")
    st.markdown("### 🔧 Settings")
    min_forms   = st.slider("Min. evaluations to include", 1, 20, 3)
    sent_thresh = st.slider("VADER negative threshold", -0.5, 0.0, -0.05, 0.01,
                            help="Compound score ≤ this value = NEGATIVE")
    st.markdown("---")
    st.markdown("**v4.0 · VADER Sentiment**  \n*All IDs anonymised*", unsafe_allow_html=True)

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(files_aubmc, files_ed, files_patho, min_f, threshold):
    depts = {}

    def load_dept(file_list, name):
        frames = [pd.read_csv(f) for f in file_list if f is not None]
        if not frames: return None, None, None
        raw = pd.concat(frames, ignore_index=True)
        return process_dept(raw, name, threshold)

    aubmc_raw, aubmc_phys, aubmc_sent = load_dept(files_aubmc, "AUBMC")
    ed_raw,    ed_phys,    ed_sent    = load_dept(files_ed,    "ED")
    patho_raw, patho_phys, patho_sent = load_dept(files_patho, "Pathology")

    # apply min_forms filter
    for phys in [aubmc_phys, ed_phys, patho_phys]:
        if phys is not None:
            phys.drop(phys[phys["n_forms"] < min_f].index, inplace=True)

    return {
        "AUBMC":     (aubmc_raw, aubmc_phys, aubmc_sent),
        "ED":        (ed_raw,    ed_phys,    ed_sent),
        "Pathology": (patho_raw, patho_phys, patho_sent),
    }

files_aubmc = [f_aubmc_23, f_aubmc_24, f_aubmc_25]
files_ed    = [f_ed_23,    f_ed_24,    f_ed_25]
files_patho = [f_patho_23, f_patho_24, f_patho_25]

any_uploaded = any(f is not None for f in files_aubmc + files_ed + files_patho)

if not any_uploaded:
    # ── LANDING PAGE ─────────────────────────────────────────────────────────
    st.markdown("# 🏥 AUBMC Physician Performance Dashboard")
    st.markdown("### Multi-method outlier detection · VADER sentiment · 2023–2025")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Methodology</div>
            <div style="margin-top:8px; font-size:14px; color:#374151; line-height:1.6">
                📏 IQR Outlier Detection<br>
                📉 Z-Score Analysis<br>
                🔢 Bottom 10% Threshold<br>
                🔗 Complaints × Sentiment
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card warning">
            <div class="metric-label">Sentiment Engine</div>
            <div style="margin-top:8px; font-size:14px; color:#374151; line-height:1.6">
                💬 VADER NLP (rule-based)<br>
                📝 Free-text comment scoring<br>
                🔄 −1.0 to +1.0 compound scale<br>
                🚫 Self-evaluations excluded
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card success">
            <div class="metric-label">Coverage</div>
            <div style="margin-top:8px; font-size:14px; color:#374151; line-height:1.6">
                🏥 3 Departments<br>
                📅 3 Years (2023–2025)<br>
                🔒 Anonymised physician IDs<br>
                📤 Export-ready outputs
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Upload your CSV files in the sidebar to begin. You can upload one or all three departments.")
    st.stop()

# ── PROCESS DATA ─────────────────────────────────────────────────────────────
with st.spinner("Processing data and running VADER sentiment analysis..."):
    data = load_and_process(
        tuple(f for f in files_aubmc),
        tuple(f for f in files_ed),
        tuple(f for f in files_patho),
        min_forms,
        sent_thresh
    )

# Build combined physician table from available departments
all_phys_frames = []
for name, (raw, phys, sent) in data.items():
    if phys is not None and len(phys) > 0:
        all_phys_frames.append(phys)

if not all_phys_frames:
    st.error("No data could be processed. Please check your uploaded files.")
    st.stop()

all_phys = pd.concat(all_phys_frames, ignore_index=True)
available_depts = [n for n, (r,p,s) in data.items() if p is not None and len(p) > 0]

# ─── MAIN HEADER ─────────────────────────────────────────────────────────────
st.markdown("# 🏥 AUBMC Physician Performance Dashboard")
st.markdown(f"**Departments active:** {'  ·  '.join(available_depts)}  &nbsp;&nbsp; **Physicians:** {len(all_phys):,}  &nbsp;&nbsp; **Years:** 2023–2025")
st.markdown("---")

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Executive Summary",
    "🎯 Flagged Physicians",
    "📊 Department View",
    "💬 Sentiment Explorer",
    "📈 Trends (2023–2025)",
    "🏢 Departments & Divisions"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🔑 Key Performance Indicators</div>', unsafe_allow_html=True)

    total      = len(all_phys)
    priority   = (all_phys["risk_score"] == 2).sum()
    monitor    = (all_phys["risk_score"] == 1).sum()
    clear      = (all_phys["risk_score"] == 0).sum()
    avg_score  = all_phys["avg_behavior_score"].mean()
    pct_neg    = (all_phys["negative_outlier"] == True).sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card neutral">
            <div class="metric-label">Total Physicians</div>
            <div class="metric-value">{total}</div>
            <div class="metric-sub">across {len(available_depts)} dept(s)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card danger">
            <div class="metric-label">Priority Review</div>
            <div class="metric-value">{priority}</div>
            <div class="metric-sub">{priority/total*100:.1f}% of physicians</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card warning">
            <div class="metric-label">Monitor</div>
            <div class="metric-value">{monitor}</div>
            <div class="metric-sub">{monitor/total*100:.1f}% of physicians</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card success">
            <div class="metric-label">Clear</div>
            <div class="metric-value">{clear}</div>
            <div class="metric-sub">{clear/total*100:.1f}% of physicians</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        color_class = "danger" if avg_score < 2.5 else ("warning" if avg_score < 3.0 else "success")
        st.markdown(f"""<div class="metric-card {color_class}">
            <div class="metric-label">Overall Avg Score</div>
            <div class="metric-value">{avg_score:.2f}</div>
            <div class="metric-sub">scale: 0 – 4</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-header">Department Score Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        dept_data  = [data[d][1]["avg_behavior_score"] for d in available_depts]
        colours    = ["#3b82f6","#f59e0b","#10b981","#8b5cf6"][:len(available_depts)]
        bp = ax.boxplot(dept_data, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2.5))
        for patch, col in zip(bp["boxes"], colours):
            patch.set_facecolor(col); patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(available_depts)+1))
        ax.set_xticklabels(available_depts, fontsize=11)
        ax.set_ylabel("Avg Behaviour Score (0–4)", fontsize=10)
        ax.set_title("Score Distribution by Department", fontsize=12, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_facecolor("#fafafa")
        fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown('<div class="section-header">Risk Score Breakdown</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        risk_counts = all_phys["risk_score"].value_counts().sort_index()
        bar_labels  = {0:"Clear (0)", 1:"Monitor (1)", 2:"Priority (2)"}
        bar_colors  = {0:"#10b981", 1:"#f59e0b", 2:"#ef4444"}
        bars = ax2.bar(
            [bar_labels.get(i, str(i)) for i in risk_counts.index],
            risk_counts.values,
            color=[bar_colors.get(i,"#6366f1") for i in risk_counts.index],
            edgecolor="white", linewidth=1.5, width=0.55
        )
        for bar, val in zip(bars, risk_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(val), ha="center", va="bottom", fontweight="700", fontsize=12)
        ax2.set_ylabel("Number of Physicians", fontsize=10)
        ax2.set_title("Composite Risk Distribution", fontsize=12, fontweight="bold", pad=10)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        ax2.set_facecolor("#fafafa")
        fig2.patch.set_facecolor("white")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-header">Department Summary Table</div>', unsafe_allow_html=True)
    summary_rows = []
    for dept in available_depts:
        _, phys, _ = data[dept]
        if phys is None: continue
        row = {
            "Department":      dept,
            "Physicians":      len(phys),
            "Avg Score":       f"{phys['avg_behavior_score'].mean():.2f}",
            "Priority (2)":    int((phys['risk_score']==2).sum()),
            "Monitor (1)":     int((phys['risk_score']==1).sum()),
            "Clear (0)":       int((phys['risk_score']==0).sum()),
            "Behaviour Flags": int(phys['behaviour_flag'].sum()),
            "Experience Flags":int(phys['experience_flag'].sum()),
        }
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FLAGGED PHYSICIANS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🎯 Physician Risk Register</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dept_filter = st.selectbox("Department", ["All"] + available_depts, key="flag_dept")
    with col_f2:
        risk_filter = st.selectbox("Risk Level", ["All","Priority (2)","Monitor (1)","Clear (0)"], key="flag_risk")
    with col_f3:
        sort_by = st.selectbox("Sort by", ["Risk Score ↓", "Avg Score ↑", "Neg. Ratio ↓"], key="flag_sort")

    df_view = all_phys.copy()
    if dept_filter != "All":
        df_view = df_view[df_view["department"] == dept_filter]
    if risk_filter == "Priority (2)":
        df_view = df_view[df_view["risk_score"] == 2]
    elif risk_filter == "Monitor (1)":
        df_view = df_view[df_view["risk_score"] == 1]
    elif risk_filter == "Clear (0)":
        df_view = df_view[df_view["risk_score"] == 0]

    if sort_by == "Risk Score ↓":
        df_view = df_view.sort_values(["risk_score","avg_behavior_score"], ascending=[False,True])
    elif sort_by == "Avg Score ↑":
        df_view = df_view.sort_values("avg_behavior_score")
    else:
        df_view = df_view.sort_values("negative_ratio", ascending=False)

    # Table with risk pills
    display_cols = {
        "physician_id":       "Physician ID",
        "department":         "Department",
        "avg_behavior_score": "Avg Score",
        "n_forms":            "Evaluations",
        "z_score":            "Z-Score",
        "behaviour_flag":     "Behaviour Flag",
        "low_iqr_outlier":    "IQR Flag",
        "low_z_outlier":      "Z-Flag",
        "negative_ratio":     "Neg. Ratio",
        "avg_compound":       "VADER Score",
        "experience_flag":    "Experience Flag",
        "negative_outlier":   "Neg. Sentiment",
        "risk_score":         "Risk Score",
    }
    show_df = df_view[[c for c in display_cols if c in df_view.columns]].copy()
    show_df.columns = [display_cols[c] for c in show_df.columns]
    if "Avg Score" in show_df.columns:
        show_df["Avg Score"] = show_df["Avg Score"].round(3)
    if "Z-Score" in show_df.columns:
        show_df["Z-Score"] = show_df["Z-Score"].round(2)
    if "Neg. Ratio" in show_df.columns:
        show_df["Neg. Ratio"] = show_df["Neg. Ratio"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    if "VADER Score" in show_df.columns:
        show_df["VADER Score"] = show_df["VADER Score"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

    st.dataframe(
        show_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=2, format="%d"
            ),
            "Avg Score": st.column_config.ProgressColumn(
                "Avg Score", min_value=0, max_value=4, format="%.3f"
            ),
        }
    )

    # CSV export
    csv_out = df_view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export filtered table as CSV", csv_out,
                       "flagged_physicians.csv", "text/csv")

    # ── Physician deep-dive ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Individual Physician Deep-Dive</div>', unsafe_allow_html=True)

    # Only show physicians that appear in the current filter
    phys_options = df_view["physician_id"].tolist()
    if phys_options:
        selected_id = st.selectbox("Select Physician ID", phys_options, key="deep_id")
        row = df_view[df_view["physician_id"] == selected_id].iloc[0]

        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            st.metric("Department", row.get("department","—"))
        with dc2:
            st.metric("Avg Behaviour Score", f"{row['avg_behavior_score']:.3f} / 4.0")
        with dc3:
            st.metric("Evaluations Received", int(row["n_forms"]))
        with dc4:
            st.metric("Composite Risk Score", f"{int(row['risk_score'])} / 2")

        dc5, dc6, dc7, dc8 = st.columns(4)
        with dc5:
            st.metric("Z-Score", f"{row.get('z_score', 0):.2f}")
        with dc6:
            beh_f = "🔴 FLAGGED" if row.get("behaviour_flag", False) else "🟢 Clear"
            st.metric("Behaviour Flag (2/3)", beh_f)
            exp_f = "🔴 FLAGGED" if row.get("experience_flag", False) else "🟢 Clear"
        with dc7:
            neg_r = row.get("negative_ratio", np.nan)
            st.metric("Neg. Comment Ratio", f"{neg_r:.1%}" if pd.notna(neg_r) else "—")
        with dc8:
            exp_label = "🔴 FLAGGED" if row.get("experience_flag", False) else "🟢 Clear"
            st.metric("Experience Flag", exp_label)

        # Show this physician's comments
        dept_name = row.get("department","AUBMC")
        if dept_name in data:
            _, _, sent_raw = data[dept_name]
            if sent_raw is not None and not sent_raw.empty and "physician_id" in sent_raw.columns:
                phys_comments = sent_raw[sent_raw["physician_id"] == selected_id].copy()
                if not phys_comments.empty:
                    st.markdown(f"**Comments for this physician** ({len(phys_comments)} total):")
                    phys_comments_sorted = phys_comments.sort_values("compound")
                    for _, crow in phys_comments_sorted.iterrows():
                        css_class = "neg" if crow["sentiment"]=="NEGATIVE" else ("pos" if crow["sentiment"]=="POSITIVE" else "neu")
                        emoji     = "🔴" if crow["sentiment"]=="NEGATIVE" else ("🟢" if crow["sentiment"]=="POSITIVE" else "⚪")
                        year_str  = str(int(crow["year"])) if "year" in crow and pd.notna(crow.get("year")) else "—"
                        rater     = crow.get("raters_group","—")
                        st.markdown(f"""
                        <div class="comment-card {css_class}">
                            <div style="font-size:11px; color:#9ca3af; margin-bottom:6px">
                                {emoji} <b>{crow["sentiment"]}</b> &nbsp;·&nbsp; Score: <b>{crow["compound"]:.3f}</b>
                                &nbsp;·&nbsp; Year: {year_str} &nbsp;·&nbsp; {rater}
                            </div>
                            <div style="font-size:14px; color:#374151">{crow["comments"]}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("No comments available for this physician.")
    else:
        st.info("No physicians match the current filter.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEPARTMENT VIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📊 Department-Level Analysis</div>', unsafe_allow_html=True)

    dept_sel = st.selectbox("Select Department", available_depts, key="dept_view")
    _, phys_d, _ = data[dept_sel]

    if phys_d is None or phys_d.empty:
        st.warning("No data available for this department.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Physicians", len(phys_d))
        with d2: st.metric("Dept. Mean Score", f"{phys_d['avg_behavior_score'].mean():.3f}")
        with d3: st.metric("Behaviour Flags", int(phys_d["behaviour_flag"].sum()))
        with d4: st.metric("Priority Flags", int((phys_d["risk_score"]==2).sum()))

        col_l, col_r = st.columns(2)

        # IQR scatter plot
        with col_l:
            st.markdown("**IQR Outlier View — Score Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            scores_d  = phys_d["avg_behavior_score"]
            Q1d, Q3d  = scores_d.quantile(0.25), scores_d.quantile(0.75)
            iqr_fence = Q1d - 1.5 * (Q3d - Q1d)
            normal    = phys_d[~phys_d["behaviour_flag"]]
            outliers  = phys_d[phys_d["behaviour_flag"]]
            ax.scatter(normal.index,  normal["avg_behavior_score"],
                       alpha=0.6, color="#3b82f6", s=55, label="Within range", zorder=3)
            ax.scatter(outliers.index, outliers["avg_behavior_score"],
                       color="#ef4444", s=100, zorder=5, label=f"Behaviour Flagged (n={len(outliers)})")
            ax.axhline(iqr_fence, color="#ef4444", linewidth=2, linestyle="--",
                       label=f"IQR Lower Fence ({iqr_fence:.2f})")
            ax.axhline(scores_d.mean(), color="#1d4ed8", linewidth=1.5, linestyle=":",
                       label=f"Mean ({scores_d.mean():.2f})")
            ax.set_xlabel("Physician Index", fontsize=10)
            ax.set_ylabel("Avg Behaviour Score", fontsize=10)
            ax.set_title(f"{dept_sel} — Behaviour Score Outliers (2-of-3 methods)", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, linestyle="--")
            ax.set_facecolor("#fafafa")
            fig.patch.set_facecolor("white")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Within-dept colleague comparison histogram
        with col_r:
            st.markdown("**Score Distribution — Colleague Comparison**")
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            scores = phys_d["avg_behavior_score"]
            n, bins, patches = ax2.hist(scores, bins=20, edgecolor="white",
                                         linewidth=0.8, color="#3b82f6", alpha=0.75)

            # Colour IQR outliers red in the histogram
            Q1h, Q3h   = scores.quantile(0.25), scores.quantile(0.75)
            iqr_thresh = Q1h - 1.5 * (Q3h - Q1h)
            for patch, left_edge in zip(patches, bins[:-1]):
                if left_edge < iqr_thresh:
                    patch.set_facecolor("#ef4444")
                    patch.set_alpha(0.8)

            ax2.axvline(scores.mean(), color="#1d4ed8", linewidth=2,
                        linestyle="-", label=f"Mean ({scores.mean():.2f})")
            ax2.axvline(scores.quantile(0.10), color="#f59e0b", linewidth=1.5,
                        linestyle=":", label=f"10th pct ({scores.quantile(.1):.2f})")

            red_patch   = mpatches.Patch(color="#ef4444", alpha=0.8, label="Below IQR fence")
            blue_patch  = mpatches.Patch(color="#3b82f6", alpha=0.75, label="Within range")
            ax2.legend(handles=[red_patch, blue_patch] +
                       [plt.Line2D([0],[0],color="#1d4ed8",linewidth=2,label=f"Mean ({scores.mean():.2f})"),
                        plt.Line2D([0],[0],color="#f59e0b",linewidth=1.5,linestyle=":",label=f"10th pct")],
                       fontsize=8)

            ax2.set_xlabel("Avg Behaviour Score (0–4)", fontsize=10)
            ax2.set_ylabel("Number of Physicians", fontsize=10)
            ax2.set_title(f"{dept_sel} — Colleague Comparison", fontsize=11, fontweight="bold")
            ax2.grid(axis="y", alpha=0.3, linestyle="--")
            ax2.set_facecolor("#fafafa")
            fig2.patch.set_facecolor("white")
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Outlier method comparison table
        st.markdown("**Outlier Method Comparison**")
        method_df = pd.DataFrame({
                "Method":       ["IQR Lower Fence", "Z-Score (≤−2)", "Bottom 10%", "Complaints + Neg. Sentiment"],
                "Flag Column":  ["low_iqr_outlier", "low_z_outlier", "low_bottom10", "combined_flag"],
            })
        method_df["Physicians Flagged"] = method_df["Flag Column"].apply(
            lambda c: int(phys_d[c].sum()) if c in phys_d.columns else 0
        )
        method_df["% of Department"] = (
            method_df["Physicians Flagged"] / len(phys_d) * 100
        ).round(1).astype(str) + "%"
        st.dataframe(method_df[["Method","Physicians Flagged","% of Department"]],
                     use_container_width=True, hide_index=True)

        # Within-dept ranking table
        st.markdown("**Physician Ranking within Department**")
        rank_df = phys_d[[
            "physician_id","avg_behavior_score","n_forms","z_score",
            "low_iqr_outlier","low_z_outlier","low_bottom10",
            "behaviour_flag","experience_flag","risk_score"
        ]].copy()
        rank_df = rank_df.sort_values("avg_behavior_score")
        rank_df["Percentile"] = (rank_df["avg_behavior_score"].rank(pct=True)*100).round(1).astype(str) + "%"
        rank_df["avg_behavior_score"] = rank_df["avg_behavior_score"].round(3)
        rank_df["z_score"] = rank_df["z_score"].round(2)
        rank_df.columns = ["Physician ID","Avg Score","Evaluations","Z-Score",
                           "IQR","Z-Score Flag","Bottom 10%",
                           "Behaviour Flag","Experience Flag","Risk Score","Percentile"]
        st.dataframe(rank_df.reset_index(drop=True), use_container_width=True, hide_index=True,
                     column_config={"Risk Score": st.column_config.ProgressColumn(min_value=0, max_value=2, format="%d"),
                                    "Avg Score":  st.column_config.ProgressColumn(min_value=0, max_value=4, format="%.3f")})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENTIMENT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">💬 VADER Sentiment Explorer</div>', unsafe_allow_html=True)

    sent_dept = st.selectbox("Department", available_depts, key="sent_dept")
    _, _, sent_raw = data[sent_dept]

    if sent_raw is None or sent_raw.empty:
        st.info("No comment data available for this department.")
    else:
        sc1, sc2, sc3, sc4 = st.columns(4)
        total_comments  = len(sent_raw)
        neg_count       = (sent_raw["sentiment"] == "NEGATIVE").sum()
        pos_count       = (sent_raw["sentiment"] == "POSITIVE").sum()
        neu_count       = (sent_raw["sentiment"] == "NEUTRAL").sum()
        with sc1: st.metric("Total Comments", total_comments)
        with sc2: st.metric("🔴 Negative", f"{neg_count} ({neg_count/total_comments*100:.1f}%)")
        with sc3: st.metric("🟢 Positive", f"{pos_count} ({pos_count/total_comments*100:.1f}%)")
        with sc4: st.metric("⚪ Neutral", f"{neu_count} ({neu_count/total_comments*100:.1f}%)")

        col_sl, col_sr = st.columns(2)

        # Compound score histogram
        with col_sl:
            st.markdown("**VADER Compound Score Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            compounds = sent_raw["compound"].dropna()
            n, bins, patches = ax.hist(compounds, bins=30, edgecolor="white", linewidth=0.5)
            for patch, left in zip(patches, bins[:-1]):
                if left <= -0.05:   patch.set_facecolor("#ef4444"); patch.set_alpha(0.8)
                elif left >= 0.05:  patch.set_facecolor("#10b981"); patch.set_alpha(0.8)
                else:               patch.set_facecolor("#9ca3af"); patch.set_alpha(0.7)
            ax.axvline(-0.05, color="#ef4444", linestyle="--", linewidth=1.2, label="Neg threshold")
            ax.axvline(+0.05, color="#10b981", linestyle="--", linewidth=1.2, label="Pos threshold")
            ax.set_xlabel("Compound Score (−1 = very negative, +1 = very positive)", fontsize=9)
            ax.set_ylabel("Comment Count", fontsize=9)
            ax.set_title(f"{sent_dept} — VADER Compound Distribution", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, linestyle="--")
            ax.set_facecolor("#fafafa")
            fig.patch.set_facecolor("white")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Physician negative ratio scatter
        with col_sr:
            st.markdown("**Negative Ratio per Physician**")
            sent_s = sentiment_summary(sent_raw)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(sent_s["total_comments"], sent_s["negative_ratio"],
                        alpha=0.6, color="#3b82f6", s=50, label="Physicians")
            flagged_s = sent_s[sent_s["negative_outlier"]]
            ax2.scatter(flagged_s["total_comments"], flagged_s["negative_ratio"],
                        color="#ef4444", s=90, zorder=5, label=f"Sentiment Outliers (n={len(flagged_s)})")
            ax2.set_xlabel("Number of Comments", fontsize=9)
            ax2.set_ylabel("Negative Comment Ratio", fontsize=9)
            ax2.set_title(f"{sent_dept} — Negative Ratio by Physician", fontsize=11, fontweight="bold")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3, linestyle="--")
            ax2.set_facecolor("#fafafa")
            fig2.patch.set_facecolor("white")
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Comment browser
        st.markdown("---")
        st.markdown("**Comment Browser**")
        cc1, cc2 = st.columns(2)
        with cc1:
            sentiment_filter = st.selectbox("Filter by Sentiment", ["All","NEGATIVE","POSITIVE","NEUTRAL"], key="sent_filter")
        with cc2:
            min_conf = st.slider("Min. VADER confidence (|compound|)", 0.0, 1.0, 0.0, 0.05, key="sent_conf")

        display_sent = sent_raw.copy()
        if sentiment_filter != "All":
            display_sent = display_sent[display_sent["sentiment"] == sentiment_filter]
        display_sent = display_sent[display_sent["compound"].abs() >= min_conf]
        display_sent = display_sent.sort_values("compound")

        st.markdown(f"Showing **{len(display_sent)}** comments")
        for _, crow in display_sent.head(50).iterrows():
            css  = "neg" if crow["sentiment"]=="NEGATIVE" else ("pos" if crow["sentiment"]=="POSITIVE" else "neu")
            emo  = "🔴" if crow["sentiment"]=="NEGATIVE" else ("🟢" if crow["sentiment"]=="POSITIVE" else "⚪")
            yr   = str(int(crow["year"])) if "year" in crow and pd.notna(crow.get("year")) else "—"
            rg   = crow.get("raters_group","—")
            pid  = crow.get("physician_id","—")
            st.markdown(f"""
            <div class="comment-card {css}">
                <div style="font-size:11px; color:#9ca3af; margin-bottom:5px">
                    {emo} <b>{crow["sentiment"]}</b> &nbsp;·&nbsp; Compound: <b>{crow["compound"]:.3f}</b>
                    &nbsp;·&nbsp; Physician: <b>{pid}</b>
                    &nbsp;·&nbsp; Year: {yr} &nbsp;·&nbsp; {rg}
                </div>
                <div style="font-size:13px; color:#374151">{crow["comments"]}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRENDS (2023–2025)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📈 Year-on-Year Trends (2023–2025)</div>', unsafe_allow_html=True)

    # ── Filters row ───────────────────────────────────────────────────────────
    tf1, tf2, tf3 = st.columns([1, 1.5, 1])
    with tf1:
        trend_dept = st.selectbox("Department", available_depts, key="trend_dept")
    raw_d, phys_d, _ = data[trend_dept]

    if raw_d is None or "year" not in raw_d.columns or raw_d["year"].isna().all():
        st.warning("Year data not available. Check that your files include a Fillout Date column.")
    else:
        years_avail = sorted(raw_d["year"].dropna().unique().astype(int))

        # Physician selector — "All" shows department-level view
        all_phys_ids = sorted(raw_d["physician_id"].dropna().unique().tolist())
        with tf2:
            view_mode = st.radio("View", ["Department Overall", "Individual Physician"],
                                 horizontal=True, key="trend_mode")
        with tf3:
            if view_mode == "Individual Physician":
                selected_phys = st.selectbox("Physician ID", all_phys_ids, key="trend_phys")
            else:
                selected_phys = None

        st.markdown("---")

        # ── DEPARTMENT OVERALL VIEW ───────────────────────────────────────────
        if view_mode == "Department Overall":

            trend_rows = []
            for yr in years_avail:
                df_yr   = raw_d[raw_d["year"] == yr]
                phys_yr = aggregate_physician(df_yr)
                phys_yr, _, _ = add_outlier_flags(phys_yr)
                trend_rows.append({
                    "Year":             yr,
                    "Physicians":       len(phys_yr),
                    "Avg Score":        round(phys_yr["avg_behavior_score"].mean(), 3),
                    "Behaviour Flags":  int(phys_yr["behaviour_flag"].sum()),
                    "% Flagged":        round(phys_yr["behaviour_flag"].mean()*100, 1),
                    "Median Score":     round(phys_yr["avg_behavior_score"].median(), 3),
                    "Score Std":        round(phys_yr["avg_behavior_score"].std(), 3),
                })
            trend_df = pd.DataFrame(trend_rows)

            if len(trend_df) >= 2:
                delta_score = trend_df["Avg Score"].iloc[-1] - trend_df["Avg Score"].iloc[0]
                delta_flag  = trend_df["% Flagged"].iloc[-1] - trend_df["% Flagged"].iloc[0]
            else:
                delta_score = 0; delta_flag = 0

            tc1, tc2, tc3 = st.columns(3)
            with tc1: st.metric("Score Change (first→last year)", f"{delta_score:+.3f}",
                                 delta=f"{delta_score:+.3f}", delta_color="normal")
            with tc2: st.metric("% Flagged Change", f"{delta_flag:+.1f}%",
                                 delta=f"{delta_flag:+.1f}%", delta_color="inverse")
            with tc3: st.metric("Years Covered", len(years_avail))

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("**Average Score Trend**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(trend_df["Year"], trend_df["Avg Score"], "o-",
                        color="#3b82f6", linewidth=2.5, markersize=8, label="Mean")
                ax.fill_between(
                    trend_df["Year"],
                    trend_df["Avg Score"] - trend_df["Score Std"],
                    trend_df["Avg Score"] + trend_df["Score Std"],
                    alpha=0.15, color="#3b82f6", label="±1 SD"
                )
                ax.plot(trend_df["Year"], trend_df["Median Score"], "s--",
                        color="#8b5cf6", linewidth=1.5, markersize=6, label="Median")
                for _, row in trend_df.iterrows():
                    ax.annotate(f"{row['Avg Score']:.2f}",
                                 (row["Year"], row["Avg Score"]),
                                 textcoords="offset points", xytext=(0,10),
                                 ha="center", fontsize=9, fontweight="600")
                ax.set_xticks(years_avail)
                ax.set_xlabel("Year", fontsize=10)
                ax.set_ylabel("Avg Behaviour Score (0–4)", fontsize=10)
                ax.set_title(f"{trend_dept} Score Trend", fontsize=11, fontweight="bold")
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3, linestyle="--")
                ax.set_facecolor("#fafafa")
                fig.patch.set_facecolor("white")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col_t2:
                st.markdown("**% Physicians Flagged (Behaviour Dimension)**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                bar_cols = ["#10b981" if p < 10 else ("#f59e0b" if p < 20 else "#ef4444")
                            for p in trend_df["% Flagged"]]
                bars = ax2.bar(trend_df["Year"], trend_df["% Flagged"],
                               color=bar_cols, edgecolor="white", linewidth=1.5, width=0.5)
                for bar, val in zip(bars, trend_df["% Flagged"]):
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.3, f"{val}%",
                             ha="center", va="bottom", fontsize=10, fontweight="700")
                ax2.set_xticks(years_avail)
                ax2.set_xlabel("Year", fontsize=10)
                ax2.set_ylabel("% Physicians with Behaviour Flag", fontsize=10)
                ax2.set_title(f"{trend_dept} — Behaviour Flag Rate Over Time", fontsize=11, fontweight="bold")
                ax2.grid(axis="y", alpha=0.3, linestyle="--")
                ax2.set_facecolor("#fafafa")
                fig2.patch.set_facecolor("white")
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                st.pyplot(fig2, use_container_width=True)
                plt.close()

            st.markdown("**Score Distribution by Year (Colleague Comparison)**")
            fig3, ax3 = plt.subplots(figsize=(10, 4.5))
            year_scores = [raw_d[raw_d["year"]==yr]["overall_score"].dropna() for yr in years_avail]
            positions   = list(range(1, len(years_avail)+1))
            bp = ax3.boxplot(year_scores, positions=positions, patch_artist=True,
                              medianprops=dict(color="white", linewidth=2.5))
            yr_colours = ["#3b82f6","#f59e0b","#10b981","#8b5cf6"]
            for patch, col in zip(bp["boxes"], yr_colours[:len(years_avail)]):
                patch.set_facecolor(col); patch.set_alpha(0.75)
            ax3.set_xticks(positions)
            ax3.set_xticklabels([str(y) for y in years_avail], fontsize=11)
            ax3.set_ylabel("Form-Level Score (0–4)", fontsize=10)
            ax3.set_title(f"{trend_dept} — Score Distribution Per Year (All Forms)", fontsize=12, fontweight="bold")
            ax3.grid(axis="y", alpha=0.3, linestyle="--")
            ax3.set_facecolor("#fafafa")
            fig3.patch.set_facecolor("white")
            st.pyplot(fig3, use_container_width=True)
            plt.close()

            st.markdown("**Year-over-Year Summary Table**")
            st.dataframe(trend_df, use_container_width=True, hide_index=True)

        # ── INDIVIDUAL PHYSICIAN VIEW ─────────────────────────────────────────
        else:
            phys_raw = raw_d[raw_d["physician_id"] == selected_phys].copy()

            if phys_raw.empty:
                st.warning(f"No data found for physician {selected_phys}.")
            else:
                # Build per-year stats for this physician
                phys_year_rows = []
                for yr in years_avail:
                    yr_data = phys_raw[phys_raw["year"] == yr]
                    if yr_data.empty:
                        continue
                    q_cols = [c for c in yr_data.columns if c.startswith("q_")]
                    phys_year_rows.append({
                        "Year":           yr,
                        "Forms":          len(yr_data),
                        "Avg Score":      round(yr_data["overall_score"].mean(), 3),
                        "Median Score":   round(yr_data["overall_score"].median(), 3),
                        "Min Score":      round(yr_data["overall_score"].min(), 3),
                        "Max Score":      round(yr_data["overall_score"].max(), 3),
                        "Score Std":      round(yr_data["overall_score"].std(), 3) if len(yr_data) > 1 else 0.0,
                    })
                phys_trend = pd.DataFrame(phys_year_rows)

                if phys_trend.empty:
                    st.warning("No year-level data available for this physician.")
                else:
                    # Compute dept avg per year for benchmarking
                    dept_avgs = {}
                    for yr in years_avail:
                        yr_all = raw_d[raw_d["year"] == yr]["overall_score"].dropna()
                        dept_avgs[yr] = yr_all.mean() if not yr_all.empty else np.nan

                    # KPI summary cards
                    years_seen = phys_trend["Year"].tolist()
                    first_score = phys_trend["Avg Score"].iloc[0]
                    last_score  = phys_trend["Avg Score"].iloc[-1]
                    delta_phys  = last_score - first_score
                    avg_forms   = phys_trend["Forms"].mean()
                    overall_avg = phys_trend["Avg Score"].mean()

                    pk1, pk2, pk3, pk4 = st.columns(4)
                    with pk1:
                        st.metric("Years on Record", len(years_seen))
                    with pk2:
                        st.metric("Overall Avg Score", f"{overall_avg:.3f} / 4.0")
                    with pk3:
                        st.metric("Score Change", f"{delta_phys:+.3f}",
                                  delta=f"{delta_phys:+.3f}", delta_color="normal")
                    with pk4:
                        st.metric("Avg Forms / Year", f"{avg_forms:.1f}")

                    st.markdown("---")
                    col_p1, col_p2 = st.columns(2)

                    # Physician score trend vs department average
                    with col_p1:
                        st.markdown(f"**Score Trend — Physician {selected_phys} vs Department Mean**")
                        fig, ax = plt.subplots(figsize=(6, 4))

                        # Department average line
                        dept_yr_list   = [yr for yr in years_avail if yr in dept_avgs]
                        dept_avg_list  = [dept_avgs[yr] for yr in dept_yr_list]
                        ax.plot(dept_yr_list, dept_avg_list, "s--",
                                color="#9ca3af", linewidth=1.5, markersize=5,
                                label=f"{trend_dept} Mean", alpha=0.8)

                        # Physician score line
                        ax.plot(phys_trend["Year"], phys_trend["Avg Score"], "o-",
                                color="#3b82f6", linewidth=2.5, markersize=9,
                                label=f"Physician {selected_phys}", zorder=5)

                        # Min-max shading for spread
                        if "Min Score" in phys_trend.columns and "Max Score" in phys_trend.columns:
                            ax.fill_between(phys_trend["Year"],
                                            phys_trend["Min Score"], phys_trend["Max Score"],
                                            alpha=0.1, color="#3b82f6", label="Score range")

                        # Label each data point
                        for _, row in phys_trend.iterrows():
                            ax.annotate(f"{row['Avg Score']:.2f}",
                                        (row["Year"], row["Avg Score"]),
                                        textcoords="offset points", xytext=(0, 10),
                                        ha="center", fontsize=9, fontweight="700",
                                        color="#1d4ed8")

                        ax.set_xticks(years_avail)
                        ax.set_xlabel("Year", fontsize=10)
                        ax.set_ylabel("Avg Behaviour Score (0–4)", fontsize=10)
                        ax.set_title(f"Physician {selected_phys} — Score Over Time",
                                     fontsize=11, fontweight="bold")
                        ax.legend(fontsize=9)
                        ax.grid(alpha=0.3, linestyle="--")
                        ax.set_facecolor("#fafafa")
                        fig.patch.set_facecolor("white")
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                    # Number of forms received per year
                    with col_p2:
                        st.markdown("**Evaluations Received Per Year**")
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        bar_colors = ["#3b82f6"] * len(phys_trend)
                        bars = ax2.bar(phys_trend["Year"], phys_trend["Forms"],
                                       color=bar_colors, edgecolor="white",
                                       linewidth=1.5, width=0.5, alpha=0.85)
                        for bar, val in zip(bars, phys_trend["Forms"]):
                            ax2.text(bar.get_x() + bar.get_width()/2,
                                     bar.get_height() + 0.1, str(int(val)),
                                     ha="center", va="bottom",
                                     fontsize=11, fontweight="700")
                        ax2.set_xticks(phys_trend["Year"].tolist())
                        ax2.set_xlabel("Year", fontsize=10)
                        ax2.set_ylabel("Number of Evaluations", fontsize=10)
                        ax2.set_title(f"Physician {selected_phys} — Evaluations Per Year",
                                      fontsize=11, fontweight="bold")
                        ax2.grid(axis="y", alpha=0.3, linestyle="--")
                        ax2.set_facecolor("#fafafa")
                        fig2.patch.set_facecolor("white")
                        ax2.spines["top"].set_visible(False)
                        ax2.spines["right"].set_visible(False)
                        st.pyplot(fig2, use_container_width=True)
                        plt.close()

                    # Percentile rank vs department per year
                    st.markdown("**Percentile Rank within Department — Year by Year**")
                    pct_rows = []
                    for yr in phys_trend["Year"].tolist():
                        yr_all   = raw_d[raw_d["year"] == yr]
                        phys_agg = aggregate_physician(yr_all)
                        if selected_phys in phys_agg["physician_id"].values:
                            phys_agg["pct"] = phys_agg["avg_behavior_score"].rank(pct=True) * 100
                            pct_val = phys_agg.loc[
                                phys_agg["physician_id"] == selected_phys, "pct"
                            ].values[0]
                            dept_avg_yr = phys_agg["avg_behavior_score"].mean()
                            pct_rows.append({
                                "Year": yr,
                                "Percentile Rank": round(pct_val, 1),
                                "Physician Score": round(phys_agg.loc[
                                    phys_agg["physician_id"] == selected_phys,
                                    "avg_behavior_score"].values[0], 3),
                                "Dept Mean Score":  round(dept_avg_yr, 3),
                                "Physicians in Dept": len(phys_agg),
                            })
                    pct_df = pd.DataFrame(pct_rows)

                    if not pct_df.empty:
                        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
                        colours_pct = ["#10b981" if p >= 50 else ("#f59e0b" if p >= 25 else "#ef4444")
                                       for p in pct_df["Percentile Rank"]]
                        bars3 = ax3.barh(pct_df["Year"].astype(str),
                                         pct_df["Percentile Rank"],
                                         color=colours_pct, edgecolor="white",
                                         linewidth=1, height=0.45, alpha=0.85)
                        ax3.axvline(50, color="#6b7280", linestyle="--",
                                    linewidth=1.2, label="50th percentile")
                        ax3.axvline(25, color="#ef4444", linestyle=":",
                                    linewidth=1.2, label="25th percentile (concern)")
                        for bar, val in zip(bars3, pct_df["Percentile Rank"]):
                            ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                                     f"{val:.0f}th",
                                     va="center", fontsize=10, fontweight="700")
                        ax3.set_xlabel("Percentile Rank within Department (100 = best)", fontsize=10)
                        ax3.set_title(f"Physician {selected_phys} — Percentile Rank Over Time",
                                      fontsize=11, fontweight="bold")
                        ax3.set_xlim(0, 110)
                        ax3.legend(fontsize=9)
                        ax3.grid(axis="x", alpha=0.3, linestyle="--")
                        ax3.set_facecolor("#fafafa")
                        fig3.patch.set_facecolor("white")
                        st.pyplot(fig3, use_container_width=True)
                        plt.close()

                        st.markdown("**Year-by-Year Summary for this Physician**")
                        merged_summary = phys_trend.merge(pct_df[["Year","Percentile Rank","Dept Mean Score","Physicians in Dept"]],
                                                          on="Year", how="left")
                        st.dataframe(merged_summary, use_container_width=True, hide_index=True)

                    # ── PEER COMPARISON ───────────────────────────────────────
                    st.markdown("---")
                    st.markdown('<div class="section-header">👥 Peer Comparison — All Physicians in Department</div>', unsafe_allow_html=True)

                    # Build full cross-physician trend table using all available years
                    peer_rows = []
                    for pid in all_phys_ids:
                        pid_data = raw_d[raw_d["physician_id"] == pid]
                        if pid_data.empty:
                            continue
                        row = {"Physician ID": pid, "Years Active": 0}
                        all_scores = []
                        for yr in years_avail:
                            yr_scores = pid_data[pid_data["year"] == yr]["overall_score"].dropna()
                            avg = round(yr_scores.mean(), 3) if not yr_scores.empty else np.nan
                            row[str(yr)] = avg
                            if not np.isnan(avg):
                                all_scores.append(avg)
                                row["Years Active"] += 1
                        row["Overall Avg"] = round(np.mean(all_scores), 3) if all_scores else np.nan
                        # Trend direction: last year minus first year with data
                        valid = [row[str(yr)] for yr in years_avail if not np.isnan(row.get(str(yr), np.nan))]
                        row["Trend"] = round(valid[-1] - valid[0], 3) if len(valid) >= 2 else np.nan
                        peer_rows.append(row)

                    peer_df = pd.DataFrame(peer_rows).sort_values("Overall Avg", ascending=False).reset_index(drop=True)

                    # Add rank and highlight selected physician
                    peer_df.insert(0, "Rank", range(1, len(peer_df)+1))
                    peer_df["Selected"] = peer_df["Physician ID"] == selected_phys

                    # ── Heatmap-style multi-year comparison chart ─────────────
                    st.markdown("**Score Heatmap — All Physicians Across Years**")
                    year_cols = [str(yr) for yr in years_avail if str(yr) in peer_df.columns]
                    heatmap_data = peer_df.set_index("Physician ID")[year_cols].astype(float)

                    fig_h, ax_h = plt.subplots(figsize=(max(6, len(year_cols)*2), max(5, len(peer_df)*0.38)))
                    im = ax_h.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto",
                                     vmin=0, vmax=4)

                    # Axis labels
                    ax_h.set_xticks(range(len(year_cols)))
                    ax_h.set_xticklabels(year_cols, fontsize=11, fontweight="600")
                    ax_h.set_yticks(range(len(heatmap_data)))
                    ax_h.set_yticklabels(heatmap_data.index.tolist(), fontsize=9)

                    # Annotate each cell with score value
                    for i in range(len(heatmap_data)):
                        for j in range(len(year_cols)):
                            val = heatmap_data.values[i, j]
                            if not np.isnan(val):
                                ax_h.text(j, i, f"{val:.2f}", ha="center", va="center",
                                          fontsize=8, fontweight="600",
                                          color="white" if val < 1.5 or val > 3.2 else "#1f2937")

                    # Highlight selected physician row with a border
                    if selected_phys in heatmap_data.index:
                        sel_row = heatmap_data.index.tolist().index(selected_phys)
                        for j in range(len(year_cols)):
                            ax_h.add_patch(plt.Rectangle(
                                (j - 0.5, sel_row - 0.5), 1, 1,
                                fill=False, edgecolor="#2563eb", linewidth=3, zorder=10
                            ))
                        ax_h.set_yticklabels(
                            ["▶ " + pid if pid == selected_phys else pid
                             for pid in heatmap_data.index.tolist()],
                            fontsize=9
                        )

                    plt.colorbar(im, ax=ax_h, label="Avg Behaviour Score (0–4)", shrink=0.6)
                    ax_h.set_title(f"{trend_dept} — Physician Score Heatmap (Selected: {selected_phys})",
                                   fontsize=12, fontweight="bold", pad=12)
                    fig_h.patch.set_facecolor("white")
                    plt.tight_layout()
                    st.pyplot(fig_h, use_container_width=True)
                    plt.close()

                    # ── Multi-line trend chart — all peers ────────────────────
                    st.markdown("**Score Trajectory — Selected Physician vs All Peers**")
                    fig_l, ax_l = plt.subplots(figsize=(9, 5))

                    for _, prow in peer_df.iterrows():
                        pid = prow["Physician ID"]
                        yr_scores = [prow.get(str(yr), np.nan) for yr in years_avail]
                        valid_yrs  = [yr for yr, sc in zip(years_avail, yr_scores) if not (isinstance(sc, float) and np.isnan(sc))]
                        valid_scs  = [sc for sc in yr_scores if not (isinstance(sc, float) and np.isnan(sc))]
                        if not valid_yrs:
                            continue
                        if pid == selected_phys:
                            # Selected physician — bold blue on top
                            ax_l.plot(valid_yrs, valid_scs, "o-",
                                      color="#2563eb", linewidth=3, markersize=9,
                                      zorder=10, label=f"▶ {pid} (selected)")
                            for vyr, vsc in zip(valid_yrs, valid_scs):
                                ax_l.annotate(f"{vsc:.2f}", (vyr, vsc),
                                              textcoords="offset points", xytext=(0, 10),
                                              ha="center", fontsize=9, fontweight="700",
                                              color="#2563eb")
                        else:
                            # All other physicians — thin grey
                            ax_l.plot(valid_yrs, valid_scs, "o-",
                                      color="#d1d5db", linewidth=1, markersize=4,
                                      alpha=0.6, zorder=1)

                    # Department mean line
                    dept_mean_line = [raw_d[raw_d["year"]==yr]["overall_score"].mean() for yr in years_avail]
                    ax_l.plot(years_avail, dept_mean_line, "s--",
                              color="#6b7280", linewidth=2, markersize=6,
                              label="Dept Mean", zorder=5)

                    ax_l.set_xticks(years_avail)
                    ax_l.set_xlabel("Year", fontsize=10)
                    ax_l.set_ylabel("Avg Behaviour Score (0–4)", fontsize=10)
                    ax_l.set_title(f"{trend_dept} — All Physician Trajectories (Selected highlighted)",
                                   fontsize=11, fontweight="bold")
                    ax_l.legend(fontsize=9, loc="lower right")
                    ax_l.grid(alpha=0.25, linestyle="--")
                    ax_l.set_facecolor("#fafafa")
                    fig_l.patch.set_facecolor("white")
                    plt.tight_layout()
                    st.pyplot(fig_l, use_container_width=True)
                    plt.close()

                    # ── Full peer comparison table ─────────────────────────────
                    st.markdown("**Full Peer Comparison Table**")
                    st.caption("Sorted by Overall Avg ↓ · Selected physician highlighted · Trend = last year minus first year")

                    # Style: highlight selected physician row
                    display_peer = peer_df.drop(columns=["Selected"]).copy()
                    if "Trend" in display_peer.columns:
                        display_peer["Trend"] = display_peer["Trend"].apply(
                            lambda x: f"▲ {x:+.3f}" if (not isinstance(x, float) or not np.isnan(x)) and x > 0
                            else (f"▼ {x:+.3f}" if (not isinstance(x, float) or not np.isnan(x)) and x < 0
                            else ("— 0.000" if x == 0 else "—"))
                        )

                    def highlight_selected(row):
                        if row["Physician ID"] == selected_phys:
                            return ["background-color: #dbeafe; font-weight: bold"] * len(row)
                        return [""] * len(row)

                    styled = display_peer.style.apply(highlight_selected, axis=1).format(
                        {str(yr): lambda x: f"{x:.3f}" if pd.notna(x) else "—" for yr in years_avail}
                    ).format({"Overall Avg": lambda x: f"{x:.3f}" if pd.notna(x) else "—"})

                    st.dataframe(styled, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DEPARTMENTS & DIVISIONS
# ═══════════════════════════════════════════════════════════════════════════════

DEPT_DIVISION_MAP = {
    "Anesthesia and Pain Medicine": [],
    "Dentofacial Medicine":         ["Orthodontics"],
    "Dermatology":                  [],
    "Diagnostic Radiology":         [],
    "Emergency Medicine":           [],
    "Family Medicine":              [],
    "Internal Medicine": [
        "Cardiology",
        "Endocrinology and Metabolism",
        "Gastroenterology",
        "General Internal Medicine and Geriatrics",
        "Geriatrics",
        "Hematology-Oncology",
        "Infectious Diseases",
        "Nephrology and Hypertension",
        "Pulmonary and Critical Care",
        "Rheumatology",
    ],
    "Neurology":        [],
    "Ob/Gyn": [
        "Obstetrics and Gynecology",
        "Maternal-Fetal Medicine",
        "Reproductive Endocrinology and Infertility",
        "Gynecologic Oncology",
    ],
    "Ophthalmology": [
        "Ophthalmology",
        "Vitreo-Retinal surgery",
        "Corneal/Refractive Surgery",
        "Oculoplastics/Orbital/Lacrimal surgery",
        "Pediatric Ophthalmology & Motility",
    ],
    "Otolaryngology, Head & Neck Surgery": [],
    "Pathology and Lab":  [],
    "Pediatrics": [
        "Pediatrics and Adolescent Medicine",
        "Neonatology",
        "Pediatric Cardiology",
        "Pediatric Critical Care",
        "Pediatric Endocrinology",
        "Pediatric Gastroenterology",
        "Pediatric Hematology-Oncology",
        "Pediatric Infectious Diseases",
        "Pediatric Nephrology",
        "Pediatric Neurology",
        "Pediatric Pulmonology",
    ],
    "Psychiatry":        [],
    "Radiation Oncology":[],
    "Surgery": [
        "General Surgery",
        "Pediatric Surgery Service",
        "Cardiothoracic Surgery",
        "Neurosurgery",
        "Orthopaedic Surgery",
        "Plastic Surgery",
        "Urology",
        "Vascular Surgery",
    ],
}

# Exact mapping: every Division value in the CSV -> parent Department label
DIV_TO_DEPT = {
    # Internal Medicine (13 rows in CSV)
    "Cardiology":                               "Internal Medicine",
    "Endocrinology":                            "Internal Medicine",
    "Endocrinology and Metabolism":             "Internal Medicine",
    "Gastroenterology":                         "Internal Medicine",
    "General Internal Medicine and Geriatrics": "Internal Medicine",
    "Geriatrics":                               "Internal Medicine",
    "Hematology-Oncology":                      "Internal Medicine",
    "Infectious Diseases":                      "Internal Medicine",
    "Internal Medicine":                        "Internal Medicine",
    "Nephrology":                               "Internal Medicine",
    "Nephrology and Hypertension":              "Internal Medicine",
    "Pulmonary and Critical Care":              "Internal Medicine",
    "Rheumatology":                             "Internal Medicine",
    # Surgery (8)
    "General Surgery":                          "Surgery",
    "Pediatric Surgery Service":                "Surgery",
    "Cardiothoracic Surgery":                   "Surgery",
    "Neurosurgery":                             "Surgery",
    "Orthopaedic Surgery":                      "Surgery",
    "Plastic Surgery":                          "Surgery",
    "Urology":                                  "Surgery",
    "Vascular Surgery":                         "Surgery",
    # Ob/Gyn (4)
    "Obstetrics and Gynecology":                "Ob/Gyn",
    "Maternal-Fetal Medicine":                  "Ob/Gyn",
    "Reproductive Endocrinology and Infertility": "Ob/Gyn",
    "Gynecologic Oncology":                     "Ob/Gyn",
    # Ophthalmology (5)
    "Ophthalmology":                            "Ophthalmology",
    "Vitreo-Retinal surgery":                   "Ophthalmology",
    "Corneal/Refractive Surgery":               "Ophthalmology",
    "Oculoplastics/Orbital/Lacrimal surgery":   "Ophthalmology",
    "Pediatric Ophthalmology & Motility":       "Ophthalmology",
    # Pediatrics (11)
    "Pediatrics and Adolescent Medicine":       "Pediatrics",
    "Neonatology":                              "Pediatrics",
    "Pediatric Cardiology":                     "Pediatrics",
    "Pediatric Critical Care":                  "Pediatrics",
    "Pediatric Endocrinology":                  "Pediatrics",
    "Pediatric Gastroenterology":               "Pediatrics",
    "Pediatric Hematology-Oncology":            "Pediatrics",
    "Pediatric Infectious Diseases":            "Pediatrics",
    "Pediatric Nephrology":                     "Pediatrics",
    "Pediatric Neurology":                      "Pediatrics",
    "Pediatric Pulmonology":                    "Pediatrics",
    # Standalone departments (each maps to itself)
    "Anesthesia and Pain Medicine":             "Anesthesia and Pain Medicine",
    "Dentofacial Medicine":                     "Dentofacial Medicine",
    "Orthodontics":                             "Dentofacial Medicine",
    "Dermatology":                              "Dermatology",
    "Diagnostic Radiology":                     "Diagnostic Radiology",
    "Emergency Medicine":                       "Emergency Medicine",
    "Family Medicine":                          "Family Medicine",
    "Neurology":                                "Neurology",
    "Otorhinolaryngology - Head and Neck Surgery": "Otolaryngology, Head & Neck Surgery",
    "Pathology and Lab":                        "Pathology and Lab",
    "Psychiatry":                               "Psychiatry",
    "Radiation Oncology":                       "Radiation Oncology",
}


with tab6:
    st.markdown('<div class="section-header">🏢 Departments & Divisions — Indicators Analysis</div>', unsafe_allow_html=True)
    st.markdown("Upload the **Physicians Indicators CSV** to explore performance by department and division.")

    ind_file = st.file_uploader(
        "📂 Upload Physicians_Indicators CSV", type="csv", key="ind_upload",
        help="Expected columns: Aubnetid, FiscalCycle, Division, ClinicVisits, ClinicWaitingTime, PatientComplaints"
    )

    if ind_file is None:
        st.info("👆 Upload your indicators file to begin.")
        st.markdown("---")
        st.markdown('<div class="section-header">📋 AUBMC Organisational Structure</div>', unsafe_allow_html=True)
        org_cols = st.columns(2)
        dept_list = list(DEPT_DIVISION_MAP.keys())
        half = len(dept_list) // 2
        for col, depts in zip(org_cols, [dept_list[:half], dept_list[half:]]):
            with col:
                for dept in depts:
                    divs = DEPT_DIVISION_MAP[dept]
                    if divs:
                        with st.expander(f"**{dept}** — {len(divs)} divisions"):
                            for d in divs:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;• {d}")
                    else:
                        st.markdown(f"**{dept}**")
    else:

        @st.cache_data
        def load_indicators(file):
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            # Department and Division columns come directly from the file
            if "Division" in df.columns:
                df["Division_norm"] = df["Division"].str.strip()
            if "Department" in df.columns:
                df["Department"] = df["Department"].str.strip()
            for col in ["ClinicVisits", "ClinicWaitingTime", "PatientComplaints"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df


        ind_df = load_indicators(ind_file)

        # Show detected columns so user can verify mapping
        with st.expander("🔍 Detected columns in your file", expanded=False):
            st.write(list(ind_df.columns))
            expected = ["Aubnetid","FiscalCycle","Division","ClinicVisits","ClinicWaitingTime","PatientComplaints"]
            missing  = [c for c in expected if c not in ind_df.columns]
            if missing:
                st.warning(f"Missing expected columns: {missing} — some metrics will show 0")
            else:
                st.success("All expected columns found ✅")

        # ── Cycle filter ──────────────────────────────────────────────────────
        cycles = ["All"] + sorted(ind_df["FiscalCycle"].dropna().unique().tolist(), reverse=True) \
                 if "FiscalCycle" in ind_df.columns else ["All"]
        fc1, fc2 = st.columns([1, 3])
        with fc1:
            sel_cycle = st.selectbox("Fiscal Cycle", cycles, key="ind_cycle")
        df_filt = ind_df if sel_cycle == "All" else ind_df[ind_df["FiscalCycle"] == sel_cycle]
        n_dept = df_filt["Department"].nunique() if "Department" in df_filt.columns else "—"
        n_div  = df_filt["Division_norm"].nunique() if "Division_norm" in df_filt.columns else "—"
        with fc2:
            st.markdown(
                f"<div style='padding-top:28px; color:#6b7280; font-size:13px'>"
                f"{len(df_filt):,} physicians &nbsp;·&nbsp; {n_dept} departments &nbsp;·&nbsp; {n_div} divisions"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── KPI cards ─────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        total_physicians = len(df_filt)
        total_visits     = int(df_filt["ClinicVisits"].sum())       if "ClinicVisits"      in df_filt.columns else 0
        total_complaints = int(df_filt["PatientComplaints"].sum())  if "PatientComplaints" in df_filt.columns else 0
        avg_wait         = df_filt["ClinicWaitingTime"].mean()      if "ClinicWaitingTime" in df_filt.columns else np.nan

        with k1:
            st.markdown(f'''<div class="metric-card neutral">
                <div class="metric-label">Physicians</div>
                <div class="metric-value">{total_physicians}</div>
                <div class="metric-sub">{sel_cycle}</div>
            </div>''', unsafe_allow_html=True)
        with k2:
            st.markdown(f'''<div class="metric-card success">
                <div class="metric-label">Total Clinic Visits</div>
                <div class="metric-value">{total_visits:,}</div>
                <div class="metric-sub">all departments</div>
            </div>''', unsafe_allow_html=True)
        with k3:
            wt_class = "success" if pd.isna(avg_wait) or avg_wait < 20 else ("warning" if avg_wait < 40 else "danger")
            wt_val   = f"{avg_wait:.1f} min" if pd.notna(avg_wait) else "—"
            st.markdown(f'''<div class="metric-card {wt_class}">
                <div class="metric-label">Avg Waiting Time</div>
                <div class="metric-value">{wt_val}</div>
                <div class="metric-sub">clinic waiting time</div>
            </div>''', unsafe_allow_html=True)
        with k4:
            cmp_class = "success" if total_complaints == 0 else ("warning" if total_complaints < 20 else "danger")
            st.markdown(f'''<div class="metric-card {cmp_class}">
                <div class="metric-label">Patient Complaints</div>
                <div class="metric-value">{total_complaints:,}</div>
                <div class="metric-sub">total reported</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Department overview ───────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Department Overview</div>', unsafe_allow_html=True)

        if "Department" in df_filt.columns:
            # Build agg spec only from columns that actually exist in the file
            agg_spec = {}
            id_col = "Aubnetid" if "Aubnetid" in df_filt.columns else (df_filt.columns[0])
            agg_spec["Physicians"] = (id_col, "nunique")
            if "Division_norm"    in df_filt.columns: agg_spec["Divisions"]        = ("Division_norm",    "nunique")
            if "ClinicVisits"     in df_filt.columns: agg_spec["Total_Visits"]     = ("ClinicVisits",     "sum")
            if "ClinicWaitingTime"in df_filt.columns: agg_spec["Avg_Wait"]         = ("ClinicWaitingTime","mean")
            if "PatientComplaints"in df_filt.columns: agg_spec["Total_Complaints"] = ("PatientComplaints","sum")
            if "PatientComplaints"in df_filt.columns: agg_spec["Avg_Complaints"]   = ("PatientComplaints","mean")

            dept_summary = (
                df_filt.groupby("Department", as_index=False)
                .agg(**agg_spec)
                .reset_index(drop=True)
            )
            # Add missing columns as zeros so downstream code doesn't break
            for c in ["Total_Visits", "Total_Complaints", "Divisions", "Avg_Wait", "Avg_Complaints"]:
                if c not in dept_summary.columns:
                    dept_summary[c] = 0
            for c in ["Total_Visits", "Total_Complaints"]:
                dept_summary[c] = dept_summary[c].fillna(0).astype(int)
            dept_summary["Avg_Wait"]       = dept_summary["Avg_Wait"].round(1)
            dept_summary["Avg_Complaints"] = dept_summary["Avg_Complaints"].round(2)
            dept_summary = dept_summary.sort_values("Total_Visits", ascending=False).reset_index(drop=True)

            dv1, dv2 = st.columns(2)
            with dv1:
                st.markdown("**Clinic Visits by Department**")
                fig, ax = plt.subplots(figsize=(7, max(4, len(dept_summary) * 0.42)))
                colours = ["#ef4444" if v == dept_summary["Total_Visits"].max()
                           else "#3b82f6" for v in dept_summary["Total_Visits"]]
                bars = ax.barh(dept_summary["Department"], dept_summary["Total_Visits"],
                               color=colours, edgecolor="white", linewidth=0.8, alpha=0.85)
                mx = dept_summary["Total_Visits"].max()
                for bar, val in zip(bars, dept_summary["Total_Visits"]):
                    ax.text(val + mx * 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{val:,}", va="center", fontsize=8, fontweight="600")
                ax.set_xlabel("Total Clinic Visits", fontsize=10)
                ax.set_title("Clinic Visits by Department", fontsize=11, fontweight="bold")
                ax.grid(axis="x", alpha=0.3, linestyle="--")
                ax.set_facecolor("#fafafa")
                fig.patch.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with dv2:
                st.markdown("**Patient Complaints by Department**")
                dept_cmp = dept_summary[dept_summary["Total_Complaints"] > 0].sort_values("Total_Complaints", ascending=False)
                if dept_cmp.empty:
                    st.success("No complaints recorded for this cycle.")
                else:
                    fig2, ax2 = plt.subplots(figsize=(7, max(4, len(dept_cmp) * 0.42)))
                    c2 = ["#ef4444" if v == dept_cmp["Total_Complaints"].max()
                          else "#f59e0b" for v in dept_cmp["Total_Complaints"]]
                    bars2 = ax2.barh(dept_cmp["Department"], dept_cmp["Total_Complaints"],
                                     color=c2, edgecolor="white", linewidth=0.8, alpha=0.85)
                    for bar, val in zip(bars2, dept_cmp["Total_Complaints"]):
                        ax2.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                                 str(int(val)), va="center", fontsize=8, fontweight="600")
                    ax2.set_xlabel("Total Patient Complaints", fontsize=10)
                    ax2.set_title("Patient Complaints by Department", fontsize=11, fontweight="bold")
                    ax2.grid(axis="x", alpha=0.3, linestyle="--")
                    ax2.set_facecolor("#fafafa")
                    fig2.patch.set_facecolor("white")
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close()

            st.markdown("**Department Summary Table**")
            dept_display = dept_summary.copy()
            dept_display.columns = ["Department", "Physicians", "Divisions", "Total Visits",
                                     "Avg Wait (min)", "Total Complaints", "Avg Complaints/Physician"]
            st.dataframe(dept_display, use_container_width=True, hide_index=True,
                         column_config={
                             "Total Visits": st.column_config.ProgressColumn(
                                 min_value=0, max_value=int(dept_display["Total Visits"].max()), format="%d"),
                             "Total Complaints": st.column_config.ProgressColumn(
                                 min_value=0, max_value=max(1, int(dept_display["Total Complaints"].max())), format="%d"),
                         })

        st.markdown("---")

        # ── Division drill-down ───────────────────────────────────────────────
        st.markdown('<div class="section-header">🔬 Division Drill-Down</div>', unsafe_allow_html=True)

        dd1, dd2 = st.columns([1, 2])
        with dd1:
            dept_opts = ["All Departments"] + sorted(df_filt["Department"].dropna().unique().tolist()) \
                        if "Department" in df_filt.columns else ["All Departments"]
            sel_dept = st.selectbox("Filter by Department", dept_opts, key="div_dept")

        df_div = df_filt if sel_dept == "All Departments" else df_filt[df_filt["Department"] == sel_dept]

        if "Division_norm" in df_div.columns:
            div_agg = {}
            id_col2 = "Aubnetid" if "Aubnetid" in df_div.columns else df_div.columns[0]
            div_agg["Physicians"] = (id_col2, "nunique")
            if "ClinicVisits"     in df_div.columns: div_agg["Total_Visits"]     = ("ClinicVisits",     "sum")
            if "ClinicWaitingTime"in df_div.columns: div_agg["Avg_Wait"]         = ("ClinicWaitingTime","mean")
            if "PatientComplaints"in df_div.columns: div_agg["Total_Complaints"] = ("PatientComplaints","sum")

            div_summary = (
                df_div.groupby("Division_norm", as_index=False)
                .agg(**div_agg)
                .sort_values("Total_Visits" if "Total_Visits" in div_agg else "Physicians", ascending=False)
                .reset_index(drop=True)
            )
            for c in ["Total_Visits", "Total_Complaints", "Avg_Wait"]:
                if c not in div_summary.columns:
                    div_summary[c] = 0
            for c in ["Total_Visits", "Total_Complaints"]:
                div_summary[c] = div_summary[c].fillna(0).astype(int)
            div_summary["Avg_Wait"] = div_summary["Avg_Wait"].round(1)

            with dd2:
                st.markdown(f"**{len(div_summary)} divisions** shown")

            fig3, ax3 = plt.subplots(figsize=(9, max(5, len(div_summary) * 0.38)))
            div_colours = ["#3b82f6" if c == 0 else "#f59e0b" if c < 3 else "#ef4444"
                           for c in div_summary["Total_Complaints"]]
            bars3 = ax3.barh(div_summary["Division_norm"], div_summary["Total_Visits"],
                             color=div_colours, edgecolor="white", linewidth=0.8, alpha=0.85)
            mx3 = div_summary["Total_Visits"].max()
            for bar, val, cmp in zip(bars3, div_summary["Total_Visits"], div_summary["Total_Complaints"]):
                label = f"{val:,}" + (f"  ⚠ {int(cmp)}" if cmp > 0 else "")
                col_txt = "#ef4444" if cmp > 0 else "#374151"
                ax3.text(val + mx3 * 0.005, bar.get_y() + bar.get_height() / 2,
                         label, va="center", fontsize=8, color=col_txt, fontweight="600")
            ax3.set_xlabel("Clinic Visits (⚠ = has complaints)", fontsize=10)
            ax3.set_title(f"Division Performance — {sel_dept}", fontsize=11, fontweight="bold")
            ax3.grid(axis="x", alpha=0.3, linestyle="--")
            ax3.set_facecolor("#fafafa")
            fig3.patch.set_facecolor("white")
            ax3.legend(handles=[
                mpatches.Patch(color="#3b82f6", alpha=0.85, label="No complaints"),
                mpatches.Patch(color="#f59e0b", alpha=0.85, label="1–2 complaints"),
                mpatches.Patch(color="#ef4444", alpha=0.85, label="3+ complaints"),
            ], fontsize=8, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close()

            st.markdown("**Division Detail Table**")
            div_display = div_summary.copy()
            div_display.columns = ["Division", "Physicians", "Total Visits", "Avg Wait (min)", "Total Complaints"]
            st.dataframe(div_display, use_container_width=True, hide_index=True,
                         column_config={
                             "Total Visits": st.column_config.ProgressColumn(
                                 min_value=0, max_value=int(div_display["Total Visits"].max()), format="%d"),
                             "Total Complaints": st.column_config.ProgressColumn(
                                 min_value=0, max_value=max(1, int(div_display["Total Complaints"].max())), format="%d"),
                         })

        st.markdown("---")

        # ── Physician-level explorer ──────────────────────────────────────────
        st.markdown('<div class="section-header">👤 Physician-Level Explorer</div>', unsafe_allow_html=True)

        pe1, pe2, pe3 = st.columns(3)
        with pe1:
            dept_opts2 = ["All"] + sorted(df_filt["Department"].dropna().unique().tolist()) \
                         if "Department" in df_filt.columns else ["All"]
            sel_dept2 = st.selectbox("Department", dept_opts2, key="pe_dept")
        df_pe = df_filt if sel_dept2 == "All" else df_filt[df_filt["Department"] == sel_dept2]
        with pe2:
            div_opts = ["All"] + sorted(df_pe["Division_norm"].dropna().unique().tolist()) \
                       if "Division_norm" in df_pe.columns else ["All"]
            sel_div = st.selectbox("Division", div_opts, key="pe_div")
        df_pe = df_pe if sel_div == "All" else df_pe[df_pe["Division_norm"] == sel_div]
        with pe3:
            sort_opts = ["Clinic Visits ↓", "Patient Complaints ↓", "Waiting Time ↓"]
            sel_sort  = st.selectbox("Sort by", sort_opts, key="pe_sort")

        sort_col_map = {
            "Clinic Visits ↓":      "ClinicVisits",
            "Patient Complaints ↓": "PatientComplaints",
            "Waiting Time ↓":       "ClinicWaitingTime",
        }
        sort_col = sort_col_map[sel_sort]
        if sort_col in df_pe.columns:
            df_pe = df_pe.sort_values(sort_col, ascending=False)

        show_cols  = ["Aubnetid", "Division_norm", "Department", "FiscalCycle",
                      "ClinicVisits", "ClinicWaitingTime", "PatientComplaints"]
        avail_show = [c for c in show_cols if c in df_pe.columns]
        show_renamed = df_pe[avail_show].rename(columns={
            "Aubnetid":          "Physician ID",
            "Division_norm":     "Division",
            "FiscalCycle":       "Cycle",
            "ClinicVisits":      "Clinic Visits",
            "ClinicWaitingTime": "Wait Time (min)",
            "PatientComplaints": "Complaints",
        }).reset_index(drop=True)

        max_visits = int(df_filt["ClinicVisits"].max())     if "ClinicVisits"      in df_filt.columns else 100
        max_cmp    = int(df_filt["PatientComplaints"].max())if "PatientComplaints" in df_filt.columns else 10
        st.dataframe(show_renamed, use_container_width=True, hide_index=True,
                     column_config={
                         "Clinic Visits": st.column_config.ProgressColumn(
                             min_value=0, max_value=max_visits, format="%d"),
                         "Complaints":    st.column_config.ProgressColumn(
                             min_value=0, max_value=max(1, max_cmp), format="%d"),
                     })

        csv_ind = show_renamed.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export filtered table as CSV", csv_ind,
                           "division_physicians.csv", "text/csv")

        # ══════════════════════════════════════════════════════════════════════
        # COMPLAINTS × SENTIMENT CROSS-ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<div class="section-header">🔗 Complaints × Sentiment — Combined Outlier Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            "Physicians flagged here have **both** a high patient complaint count (IQR outlier) "
            "**and** negative peer sentiment from survey comments. "
            "Convergence of both signals = strongest evidence for review."
        )

        # ── Check if survey sentiment data is available ───────────────────────
        sent_frames = []
        for dept_name in available_depts:
            _, _, sent_raw = data[dept_name]
            if sent_raw is not None and not sent_raw.empty:
                sent_raw_copy = sent_raw.copy()
                sent_raw_copy["dept_source"] = dept_name
                sent_frames.append(sent_raw_copy)

        if not sent_frames:
            st.warning("No survey comment data loaded yet. Upload your behaviour survey CSVs in the sidebar to enable this analysis.")
        else:
            # Combine all sentiment data
            all_sent = pd.concat(sent_frames, ignore_index=True)

            # Build per-physician sentiment summary across all departments
            sent_cross = (
                all_sent.assign(is_neg=(all_sent["sentiment"] == "NEGATIVE"))
                .groupby("physician_id", as_index=False)
                .agg(
                    total_comments   =("is_neg",   "count"),
                    negative_comments=("is_neg",   "sum"),
                    avg_compound     =("compound", "mean"),
                )
            )
            sent_cross["negative_ratio"] = sent_cross["negative_comments"] / sent_cross["total_comments"]

            # Sentiment flag: IQR upper fence on negative_ratio (min 3 comments)
            Q1s, Q3s = sent_cross["negative_ratio"].quantile(0.25), sent_cross["negative_ratio"].quantile(0.75)
            sent_ub  = Q3s + 1.5 * (Q3s - Q1s)
            sent_cross["sentiment_flag"] = (
                (sent_cross["negative_ratio"] > sent_ub) &
                (sent_cross["total_comments"] >= 3)
            )

            # ── Complaints: IQR upper fence per physician ─────────────────────
            if "PatientComplaints" not in df_filt.columns or "Aubnetid" not in df_filt.columns:
                st.warning("PatientComplaints or Aubnetid column not found in indicators file.")
            else:
                complaints = (
                    df_filt.groupby("Aubnetid", as_index=False)
                    .agg(
                        total_complaints =("PatientComplaints", "sum"),
                        department       =("Department",        lambda x: x.mode()[0] if not x.empty else "—"),
                        division         =("Division",          lambda x: x.mode()[0] if not x.empty else "—"),
                    )
                )
                complaints["total_complaints"] = pd.to_numeric(complaints["total_complaints"], errors="coerce").fillna(0)

                # IQR outlier on complaints
                Q1c, Q3c = complaints["total_complaints"].quantile(0.25), complaints["total_complaints"].quantile(0.75)
                IQRc     = Q3c - Q1c
                complaints_ub = Q3c + 1.5 * IQRc
                complaints["complaints_flag"] = complaints["total_complaints"] > complaints_ub

                # ── Merge complaints + sentiment on physician ID ───────────────
                # Indicators uses Aubnetid; sentiment uses physician_id — same field
                merged = complaints.merge(
                    sent_cross.rename(columns={"physician_id": "Aubnetid"}),
                    on="Aubnetid", how="left"
                )
                merged["sentiment_flag"]  = merged["sentiment_flag"].fillna(False)
                merged["negative_ratio"]  = merged["negative_ratio"].fillna(0)
                merged["avg_compound"]    = merged["avg_compound"].fillna(0)
                merged["total_comments"]  = merged["total_comments"].fillna(0).astype(int)

                # ── Combined outlier badge ────────────────────────────────────
                def combined_badge(row):
                    if row["complaints_flag"] and row["sentiment_flag"]:
                        return "Priority"
                    elif row["complaints_flag"] or row["sentiment_flag"]:
                        return "Monitor"
                    return "Clear"

                merged["combined_status"] = merged.apply(combined_badge, axis=1)

                # ── Summary KPIs ──────────────────────────────────────────────
                n_priority = (merged["combined_status"] == "Priority").sum()
                n_monitor  = (merged["combined_status"] == "Monitor").sum()
                n_clear    = (merged["combined_status"] == "Clear").sum()
                n_total    = len(merged)

                cx1, cx2, cx3, cx4 = st.columns(4)
                with cx1:
                    st.markdown(f'''<div class="metric-card danger">
                        <div class="metric-label">⚠ Priority (Both Flags)</div>
                        <div class="metric-value">{n_priority}</div>
                        <div class="metric-sub">{n_priority/n_total*100:.1f}% of physicians</div>
                    </div>''', unsafe_allow_html=True)
                with cx2:
                    st.markdown(f'''<div class="metric-card warning">
                        <div class="metric-label">👁 Monitor (One Flag)</div>
                        <div class="metric-value">{n_monitor}</div>
                        <div class="metric-sub">{n_monitor/n_total*100:.1f}% of physicians</div>
                    </div>''', unsafe_allow_html=True)
                with cx3:
                    st.markdown(f'''<div class="metric-card success">
                        <div class="metric-label">✓ Clear</div>
                        <div class="metric-value">{n_clear}</div>
                        <div class="metric-sub">{n_clear/n_total*100:.1f}% of physicians</div>
                    </div>''', unsafe_allow_html=True)
                with cx4:
                    st.markdown(f'''<div class="metric-card neutral">
                        <div class="metric-label">IQR Complaint Threshold</div>
                        <div class="metric-value">{complaints_ub:.0f}</div>
                        <div class="metric-sub">complaints to trigger flag</div>
                    </div>''', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Side-by-side scatter: complaints vs sentiment ─────────────
                st.markdown("**Complaints vs Sentiment Score — All Physicians**")
                st.caption("Top-left = high complaints + negative sentiment = Priority · Dashed lines = IQR thresholds")

                fig_s, ax_s = plt.subplots(figsize=(10, 6))

                colour_map = {"Priority": "#ef4444", "Monitor": "#f59e0b", "Clear": "#10b981"}
                for status, grp in merged.groupby("combined_status"):
                    ax_s.scatter(
                        grp["total_complaints"],
                        grp["avg_compound"],
                        c=colour_map.get(status, "#6b7280"),
                        s=70, alpha=0.75, zorder=3,
                        label=f"{status} (n={len(grp)})"
                    )

                # Threshold lines
                ax_s.axvline(complaints_ub, color="#ef4444", linestyle="--",
                             linewidth=1.5, alpha=0.7, label=f"Complaint IQR fence ({complaints_ub:.0f})")
                ax_s.axhline(-0.05, color="#f59e0b", linestyle=":",
                             linewidth=1.5, alpha=0.7, label="Sentiment neutral threshold")

                # Quadrant labels
                y_range = merged["avg_compound"].max() - merged["avg_compound"].min()
                y_top   = merged["avg_compound"].max() - y_range * 0.05
                y_bot   = merged["avg_compound"].min() + y_range * 0.05
                x_right = merged["total_complaints"].max() * 0.98

                ax_s.text(complaints_ub + 0.3, y_bot,
                          "⚠ HIGH RISK\nComplaints + Negative",
                          fontsize=8, color="#ef4444", fontweight="700",
                          va="bottom", ha="left",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", alpha=0.8))

                ax_s.set_xlabel("Total Patient Complaints", fontsize=11)
                ax_s.set_ylabel("Avg VADER Compound Score (−1=negative, +1=positive)", fontsize=11)
                ax_s.set_title("Patient Complaints vs Peer Sentiment — Combined Outlier View",
                               fontsize=12, fontweight="bold")
                ax_s.legend(fontsize=9, loc="upper right")
                ax_s.grid(alpha=0.25, linestyle="--")
                ax_s.set_facecolor("#fafafa")
                fig_s.patch.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig_s, use_container_width=True)
                plt.close()

                # ── Department peer comparison bar chart ──────────────────────
                st.markdown("**Priority Physicians vs Department Peers**")

                dept_cross = (
                    merged.groupby("department", as_index=False)
                    .agg(
                        Total        =("Aubnetid",         "count"),
                        Priority     =("combined_status",  lambda x: (x == "Priority").sum()),
                        Monitor      =("combined_status",  lambda x: (x == "Monitor").sum()),
                        Avg_Complaints=("total_complaints","mean"),
                        Avg_Compound  =("avg_compound",    "mean"),
                    )
                )
                dept_cross["Priority_pct"] = (dept_cross["Priority"] / dept_cross["Total"] * 100).round(1)
                dept_cross = dept_cross.sort_values("Priority_pct", ascending=False)

                fig_d, ax_d = plt.subplots(figsize=(10, max(4, len(dept_cross)*0.45)))
                bar_c = ["#ef4444" if p > 0 else "#10b981" for p in dept_cross["Priority_pct"]]
                bars_d = ax_d.barh(dept_cross["department"], dept_cross["Priority_pct"],
                                   color=bar_c, edgecolor="white", linewidth=0.8, alpha=0.85)
                for bar, pct, n in zip(bars_d, dept_cross["Priority_pct"], dept_cross["Priority"]):
                    label = f"{pct:.1f}%  ({int(n)} physicians)" if n > 0 else "0%"
                    ax_d.text(max(pct + 0.2, 0.5), bar.get_y() + bar.get_height()/2,
                              label, va="center", fontsize=9, fontweight="600",
                              color="#ef4444" if n > 0 else "#6b7280")
                ax_d.set_xlabel("% Physicians with Priority Flag (High Complaints + Negative Sentiment)", fontsize=10)
                ax_d.set_title("Priority Flag Rate by Department", fontsize=11, fontweight="bold")
                ax_d.grid(axis="x", alpha=0.3, linestyle="--")
                ax_d.set_facecolor("#fafafa")
                fig_d.patch.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig_d, use_container_width=True)
                plt.close()

                # ── Physician table (Priority first) ──────────────────────────
                st.markdown("**Physician Combined Outlier Table**")
                st.caption("Sorted by status → complaints → sentiment score")

                status_order = {"Priority": 0, "Monitor": 1, "Clear": 2}
                merged["_sort"] = merged["combined_status"].map(status_order)
                table_out = (
                    merged.sort_values(["_sort", "total_complaints", "avg_compound"],
                                       ascending=[True, False, True])
                    .drop(columns=["_sort"])
                    [["Aubnetid", "department", "division",
                      "total_complaints", "complaints_flag",
                      "total_comments", "negative_ratio", "avg_compound",
                      "sentiment_flag", "combined_status"]]
                    .rename(columns={
                        "Aubnetid":          "Physician ID",
                        "department":        "Department",
                        "division":          "Division",
                        "total_complaints":  "Complaints",
                        "complaints_flag":   "Complaint Flag",
                        "total_comments":    "Comments Scored",
                        "negative_ratio":    "Neg. Ratio",
                        "avg_compound":      "VADER Score",
                        "sentiment_flag":    "Sentiment Flag",
                        "combined_status":   "Status",
                    })
                    .reset_index(drop=True)
                )
                table_out["Neg. Ratio"] = table_out["Neg. Ratio"].apply(lambda x: f"{x:.1%}")
                table_out["VADER Score"] = table_out["VADER Score"].apply(lambda x: f"{x:.3f}")

                max_cmp2 = int(table_out["Complaints"].max()) if len(table_out) > 0 else 10
                st.dataframe(
                    table_out,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Complaints": st.column_config.ProgressColumn(
                            min_value=0, max_value=max_cmp2, format="%d"),
                        "Status": st.column_config.SelectboxColumn(
                            options=["Priority", "Monitor", "Clear"]),
                    }
                )

                # Export
                csv_cross = table_out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Export combined outlier table", csv_cross,
                                   "complaints_sentiment_outliers.csv", "text/csv")
