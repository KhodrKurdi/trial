import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import io
import warnings
warnings.filterwarnings("ignore")

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
    df["z_score"]           = (scores - pop_mean) / pop_std
    df["low_z_outlier"]     = df["z_score"] <= -2
    Q1, Q3                  = scores.quantile(0.25), scores.quantile(0.75)
    IQR                     = Q3 - Q1
    df["low_iqr_outlier"]   = scores < (Q1 - 1.5 * IQR)
    df["low_bottom10"]      = scores <= scores.quantile(0.10)
    df["se"]                = pop_std / np.sqrt(df["n_forms"])
    df["lower_funnel_95"]   = pop_mean - 1.96 * df["se"]
    df["low_funnel_outlier"]= scores < df["lower_funnel_95"]
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
    df["risk_score"] = df["low_funnel_outlier"].astype(int) + df["negative_outlier"].astype(int)
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
                📊 Funnel Plot (95% LCL)<br>
                📏 IQR Outlier Detection<br>
                📉 Z-Score Analysis<br>
                🔢 Bottom 10% Threshold
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Executive Summary",
    "🎯 Flagged Physicians",
    "📊 Department View",
    "💬 Sentiment Explorer",
    "📈 Trends (2023–2025)"
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
            "Funnel Outliers": int(phys['low_funnel_outlier'].sum()),
            "Sentiment Flags": int(phys['negative_outlier'].sum()),
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
        "low_funnel_outlier": "Funnel Flag",
        "low_iqr_outlier":    "IQR Flag",
        "low_z_outlier":      "Z-Flag",
        "negative_ratio":     "Neg. Ratio",
        "avg_compound":       "VADER Score",
        "negative_outlier":   "Sentiment Flag",
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
            funnel = "🔴 FLAGGED" if row.get("low_funnel_outlier", False) else "🟢 Clear"
            st.metric("Funnel Plot", funnel)
        with dc7:
            neg_r = row.get("negative_ratio", np.nan)
            st.metric("Neg. Comment Ratio", f"{neg_r:.1%}" if pd.notna(neg_r) else "—")
        with dc8:
            vader_s = row.get("avg_compound", np.nan)
            st.metric("VADER Compound", f"{vader_s:.3f}" if pd.notna(vader_s) else "—")

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
        with d3: st.metric("Funnel Outliers", int(phys_d["low_funnel_outlier"].sum()))
        with d4: st.metric("Priority Flags", int((phys_d["risk_score"]==2).sum()))

        col_l, col_r = st.columns(2)

        # Funnel plot
        with col_l:
            st.markdown("**Funnel Plot — Score vs. Evaluations**")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            df_s = phys_d.sort_values("n_forms")
            ax.scatter(df_s["n_forms"], df_s["avg_behavior_score"],
                       alpha=0.6, color="#3b82f6", s=55, label="Physicians", zorder=3)
            ax.plot(df_s["n_forms"], df_s["lower_funnel_95"],
                    color="#ef4444", linewidth=2, linestyle="--", label="95% LCL")
            outliers = df_s[df_s["low_funnel_outlier"]]
            ax.scatter(outliers["n_forms"], outliers["avg_behavior_score"],
                       color="#ef4444", s=100, zorder=5, label=f"Low Outliers (n={len(outliers)})")
            ax.set_xlabel("Number of Evaluations", fontsize=10)
            ax.set_ylabel("Avg Behaviour Score", fontsize=10)
            ax.set_title(f"{dept_sel} Funnel Plot", fontsize=11, fontweight="bold")
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

            # Colour funnel outliers red in the histogram
            funnel_thresh = phys_d["lower_funnel_95"].min()
            for patch, left_edge in zip(patches, bins[:-1]):
                if left_edge < funnel_thresh:
                    patch.set_facecolor("#ef4444")
                    patch.set_alpha(0.8)

            ax2.axvline(scores.mean(), color="#1d4ed8", linewidth=2,
                        linestyle="-", label=f"Mean ({scores.mean():.2f})")
            ax2.axvline(scores.quantile(0.10), color="#f59e0b", linewidth=1.5,
                        linestyle=":", label=f"10th pct ({scores.quantile(.1):.2f})")

            red_patch   = mpatches.Patch(color="#ef4444", alpha=0.8, label="Below funnel LCL")
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
            "Method":       ["Funnel Plot (95%)", "IQR Lower Fence", "Z-Score (≤−2)", "Bottom 10%"],
            "Flag Column":  ["low_funnel_outlier","low_iqr_outlier","low_z_outlier","low_bottom10"],
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
            "low_funnel_outlier","negative_outlier","risk_score"
        ]].copy()
        rank_df = rank_df.sort_values("avg_behavior_score")
        rank_df["Percentile"] = (rank_df["avg_behavior_score"].rank(pct=True)*100).round(1).astype(str) + "%"
        rank_df["avg_behavior_score"] = rank_df["avg_behavior_score"].round(3)
        rank_df["z_score"] = rank_df["z_score"].round(2)
        rank_df.columns = ["Physician ID","Avg Score","Evaluations","Z-Score",
                           "Funnel Flag","Sentiment Flag","Risk Score","Percentile"]
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

    trend_dept = st.selectbox("Department", available_depts, key="trend_dept")
    raw_d, phys_d, _ = data[trend_dept]

    if raw_d is None or "year" not in raw_d.columns or raw_d["year"].isna().all():
        st.warning("Year data not available. Check that your files include a Fillout Date column.")
    else:
        years_avail = sorted(raw_d["year"].dropna().unique().astype(int))

        # Year-level physician stats
        trend_rows = []
        for yr in years_avail:
            df_yr    = raw_d[raw_d["year"] == yr]
            phys_yr  = aggregate_physician(df_yr)
            phys_yr, _, _ = add_outlier_flags(phys_yr)
            trend_rows.append({
                "Year":                yr,
                "Physicians":         len(phys_yr),
                "Avg Score":          round(phys_yr["avg_behavior_score"].mean(), 3),
                "Funnel Outliers":    int(phys_yr["low_funnel_outlier"].sum()),
                "% Flagged":          round(phys_yr["low_funnel_outlier"].mean()*100, 1),
                "Median Score":       round(phys_yr["avg_behavior_score"].median(), 3),
                "Score Std":          round(phys_yr["avg_behavior_score"].std(), 3),
            })
        trend_df = pd.DataFrame(trend_rows)

        # Trend metrics
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
            st.markdown("**% Physicians Flagged by Funnel Plot**")
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
            ax2.set_ylabel("% Physicians Below Funnel LCL", fontsize=10)
            ax2.set_title(f"{trend_dept} — Flagged Rate Over Time", fontsize=11, fontweight="bold")
            ax2.grid(axis="y", alpha=0.3, linestyle="--")
            ax2.set_facecolor("#fafafa")
            fig2.patch.set_facecolor("white")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Year-over-year score distributions (violin / box per year)
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
