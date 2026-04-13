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


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AUBMC Physician Performance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── AUBMC BRAND THEME ───────────────────────────────────────────────────────
# Palette: Navy #1a365d | AUBMC Blue #2b7bc8 | Sky #e8f4fd | White #ffffff
st.markdown("""
<style>
    /* ── Global ── */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: #f0f6fc !important;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    [data-testid="stHeader"]  { background-color: #f0f6fc !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    header[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }
    .block-container { padding-top: 2rem !important; max-width: 1200px !important; }

    /* ── Metric Cards ── */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(43,123,200,0.10);
        border-left: 4px solid #2b7bc8;
        margin-bottom: 8px;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover { box-shadow: 0 4px 20px rgba(43,123,200,0.18); }
    .metric-card.warning { border-left-color: #f59e0b; box-shadow: 0 2px 12px rgba(245,158,11,0.10); }
    .metric-card.danger  { border-left-color: #e53e3e; box-shadow: 0 2px 12px rgba(229,62,62,0.10); }
    .metric-card.success { border-left-color: #38a169; box-shadow: 0 2px 12px rgba(56,161,105,0.10); }
    .metric-card.neutral { border-left-color: #6b7280; }
    .metric-label { font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { font-size: 34px; font-weight: 800; color: #1a365d; line-height: 1.15; }
    .metric-sub   { font-size: 12px; color: #94a3b8; margin-top: 3px; }

    /* ── Section Headers ── */
    .section-header {
        font-size: 17px; font-weight: 700; color: #1a365d;
        border-left: 4px solid #2b7bc8;
        padding: 6px 0 6px 14px;
        margin-bottom: 18px; margin-top: 4px;
        background: linear-gradient(90deg, rgba(43,123,200,0.06) 0%, transparent 100%);
        border-radius: 0 6px 6px 0;
    }

    /* ── Pills ── */
    .pill-red    { background:#fff0f0; color:#c53030; padding:3px 11px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid #fed7d7; }
    .pill-yellow { background:#fffbeb; color:#b7791f; padding:3px 11px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid #fef3c7; }
    .pill-green  { background:#f0fff4; color:#276749; padding:3px 11px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid #c6f6d5; }
    .pill-grey   { background:#f1f5f9; color:#475569; padding:3px 11px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid #e2e8f0; }

    /* ── Tabs ── */
    div[data-testid="stSidebarNav"] { display: none; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: white !important;
        border-radius: 12px 12px 0 0;
        padding: 6px 6px 0 6px;
        border-bottom: 2px solid #2b7bc8;
        box-shadow: 0 2px 8px rgba(43,123,200,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 13px;
        color: #64748b !important;
        padding: 8px 16px !important;
        transition: all 0.15s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #e8f4fd !important;
        color: #2b7bc8 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #2b7bc8 !important;
        color: white !important;
        font-weight: 700 !important;
    }

    /* ── Comment Cards ── */
    .comment-card {
        background: white; border-radius: 10px; padding: 14px 18px;
        margin-bottom: 10px; border-left: 3px solid #cbd5e0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .comment-card.neg { border-left-color: #e53e3e; background: #fff8f8; }
    .comment-card.pos { border-left-color: #38a169; background: #f8fff9; }
    .comment-card.neu { border-left-color: #cbd5e0; }

    /* ── Chat messages — force dark text ── */
    [data-testid="stChatMessage"] { background: white !important; border-radius: 12px !important; }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div { color: #1a365d !important; }
    [data-testid="stChatMessage"] table {
        border-collapse: collapse; width: 100%;
    }
    [data-testid="stChatMessage"] th {
        background: #2b7bc8 !important; color: white !important;
        padding: 8px 12px; font-size: 13px;
    }
    [data-testid="stChatMessage"] td {
        padding: 7px 12px; border-bottom: 1px solid #e2e8f0;
        font-size: 13px; color: #1a365d !important;
    }
    [data-testid="stChatMessage"] tr:nth-child(even) td {
        background: #f0f6fc !important;
    }

    /* ── Streamlit native ── */
    .stMetric {
        background: white; border-radius: 10px; padding: 14px 16px;
        box-shadow: 0 1px 6px rgba(43,123,200,0.08);
    }
    .stMetric label { color: #64748b !important; font-size: 12px !important; font-weight: 600 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #1a365d !important; font-weight: 700 !important; }
    div[data-testid="stDataFrame"] { background: white; border-radius: 10px; box-shadow: 0 1px 6px rgba(43,123,200,0.08); }
    .stSelectbox label, .stSlider label, .stRadio label { color: #1a365d !important; font-weight: 600 !important; font-size: 13px !important; }
    .stSelectbox > div > div { background: white !important; color: #1a365d !important; border-color: #bfdbfe !important; border-radius: 8px !important; }
    h1 { color: #1a365d !important; font-weight: 800 !important; }
    h2, h3 { color: #1a365d !important; font-weight: 700 !important; }
    p, li { color: #374151; }
    .stMarkdown p { color: #374151; }
    hr { border-color: #e2e8f0 !important; }
    .stAlert { background: white !important; border-radius: 10px !important; }
    [data-testid="stInfo"]    { background: #eff6ff !important; border-color: #2b7bc8 !important; }
    [data-testid="stWarning"] { background: #fffbeb !important; border-color: #f59e0b !important; }
    [data-testid="stError"]   { background: #fff5f5 !important; border-color: #e53e3e !important; }
    [data-testid="stSuccess"] { background: #f0fff4 !important; border-color: #38a169 !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    /* ── Division Cards (Tab 6) ── */
    .div-card {
        background: white; border-radius: 14px; padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(43,123,200,0.09);
        border-top: 4px solid #2b7bc8; margin-bottom: 14px;
        transition: box-shadow 0.2s, transform 0.15s;
    }
    .div-card:hover { box-shadow: 0 6px 22px rgba(43,123,200,0.16); transform: translateY(-1px); }
    .div-card.alert { border-top-color: #e53e3e; }
    .div-card.warn  { border-top-color: #f59e0b; }
    .div-name  { font-size: 15px; font-weight: 700; color: #1a365d; margin-bottom: 14px; line-height: 1.3; }
    .div-stats { display: flex; gap: 20px; flex-wrap: wrap; }
    .div-stat  { font-size: 12px; color: #64748b; line-height: 1.8; }
    .div-stat span { font-weight: 800; color: #1a365d; font-size: 16px; display: block; }

    /* ── Dept Banner ── */
    .dept-banner {
        background: linear-gradient(135deg, #1a365d 0%, #2b7bc8 100%);
        border-radius: 12px; padding: 18px 24px; margin-bottom: 16px;
        color: white;
    }
    .dept-banner-name { font-size: 20px; font-weight: 800; letter-spacing: -0.3px; }
    .dept-banner-stats { font-size: 13px; opacity: 0.85; margin-top: 4px; }
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
    # Normalize newlines that BLUE Explorance embeds in column headers
    c = re.sub(r"[\r\n]+", " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    m = re.search(r"_(.*?)_First Scale", c)
    if not m:
        core = c
    else:
        full = m.group(1).strip()
        # Strip BLUE Explorance form header: everything up to and including ], MD or [S$LN]
        cleaned = re.sub(r"^.*?\[S\$LN\],?\s*(?:MD)?\s*_?\s*", "", full).strip().strip("_").strip()
        core = cleaned if cleaned else full
    # If it's a long sentence, take first 7 words for the key
    words = core.split()
    if len(words) > 7:
        core = " ".join(words[:7]).rstrip(".,;:")
    core = core.lower()
    core = re.sub(r"[^a-z0-9]+", "_", core).strip("_")
    return f"q_{core}"

def clean_headers(df):
    df = df.copy()
    df.columns = [clean_question_col(c) for c in df.columns]
    return df

def q_display_label(q_key):
    """Convert q_ snake_case key to a readable display label for charts."""
    label = q_key.replace("q_", "").replace("_", " ").title()
    # Truncate to ~40 chars for chart readability
    return label[:42] + "…" if len(label) > 42 else label

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
        # Fill NaN years from explicit tag if available
        if "_explicit_year" in df.columns:
            df["year"] = df["year"].fillna(df["_explicit_year"])
    elif "year" not in df.columns:
        # Fallback: try to detect year column by name variants
        year_candidates = [c for c in df.columns if "year" in c.lower() or "date" in c.lower()]
        if year_candidates:
            df["year"] = pd.to_datetime(df[year_candidates[0]], errors="coerce").dt.year
        else:
            df["year"] = np.nan  # year unknown — will be excluded from 2025 filter
    return df

def aggregate_physician(df):
    # Exclude self-evaluations — not peer assessments
    if "raters_group" in df.columns:
        df = df[df["raters_group"] != "Faculty Self-Evaluation"].copy()

    q_cols = [c for c in df.columns if c.startswith("q_")]

    # Flat mean: stack all question responses across all forms per physician,
    # drop NaN, then take a single mean — avoids mean-of-means distortion
    # where forms with fewer answered questions get equal weight to full forms
    def flat_mean(grp):
        vals = grp[q_cols].values.flatten()
        vals = vals[~pd.isnull(vals)]
        return float(vals.mean()) if len(vals) > 0 else np.nan

    grouped = df.groupby("physician_id")
    scores  = grouped.apply(flat_mean).reset_index()
    scores.columns = ["physician_id", "avg_behavior_score"]
    n_forms = grouped.size().reset_index(name="n_forms")
    return scores.merge(n_forms, on="physician_id")

def add_outlier_flags(phys_df):
    df = phys_df.copy()
    scores   = df["avg_behavior_score"]
    pop_mean = scores.mean()
    pop_std  = scores.std(ddof=0) if len(scores) > 1 else 0

    df["z_score"]         = (scores - pop_mean) / pop_std if pop_std > 0 else 0.0
    df["low_z_outlier"]   = df["z_score"] <= -2

    Q1, Q3                = scores.quantile(0.25), scores.quantile(0.75)
    IQR                   = Q3 - Q1
    df["low_iqr_outlier"] = scores < (Q1 - 1.5 * IQR) if IQR > 0 else False
    df["low_bottom10"]    = scores <= scores.quantile(0.10)
    return df, pop_mean, pop_std

vader = SentimentIntensityAnalyzer()

# ── Custom medical/professional domain lexicon ────────────────────────────────
# VADER was built for social media — it misses clinical/professional negative terms.
# We extend its lexicon with domain-specific words scored -1.0 to -3.0 (negative)
# and +1.0 to +2.0 (positive). Scale: ±4.0 max in VADER's system.
_MEDICAL_LEXICON = {
    # Strong negatives (-2.5 to -3.0)
    "sexism":           -3.0,
    "harassment":       -3.0,
    "abuse":            -3.0,
    "abusive":          -3.0,
    "discriminatory":   -2.8,
    "discrimination":   -2.8,
    "malpractice":      -2.8,
    "negligent":        -2.8,
    "negligence":       -2.8,
    "incompetent":      -2.8,
    "incompetence":     -2.8,
    "unethical":        -2.8,
    "misconduct":       -2.8,

    # Moderate negatives (-1.5 to -2.5)
    "disrespects":      -2.5,
    "disrespectful":    -2.5,
    "disrespect":       -2.3,
    "suboptimal":       -2.0,
    "unprofessional":   -2.3,
    "triggering":       -1.8,
    "dismissive":       -2.0,
    "condescending":    -2.3,
    "arrogant":         -2.0,
    "rude":             -2.0,
    "intimidating":     -1.8,
    "demeaning":        -2.3,
    "belittling":       -2.3,
    "humiliates":       -2.5,
    "humiliating":      -2.5,
    "inadequate":       -1.8,
    "inconsiderate":    -1.8,
    "inattentive":      -1.5,
    "unapproachable":   -1.8,
    "unavailable":      -1.2,
    "unhelpful":        -1.8,
    "uncompassionate":  -2.0,
    "impatient":        -1.5,
    "impolite":         -1.8,
    "inappropriate":    -2.0,
    "unprepared":       -1.5,
    "overconfident":    -1.5,
    "careless":         -2.0,
    "rushed":           -1.3,
        "disorganized":     -1.5,
        "bullying":         -2.8,
    "bully":            -2.5,

    # Mild negatives (-0.5 to -1.5)
                    "complaints":       -1.0,
    "complaint":        -1.0,
                
    # Positives (+1.5 to +2.5)
    "compassionate":    +2.5,
    "empathetic":       +2.5,
    "empathy":          +2.3,
    "supportive":       +2.0,
    "knowledgeable":    +2.0,
    "dedicated":        +2.0,
    "thorough":         +2.0,
    "approachable":     +2.0,
    "attentive":        +2.0,
    "collaborative":    +1.8,
    "professional":     +1.8,
    "respectful":       +2.0,
    "responsive":       +1.8,
    "excellent":        +2.5,
    "outstanding":      +2.5,
    "exceptional":      +2.5,
    "commendable":      +2.3,
}
vader.lexicon.update(_MEDICAL_LEXICON)

def score_vader(text, threshold=-0.05):
    try:
        s = vader.polarity_scores(str(text))
        c = s["compound"]
        # Notebook uses standard VADER thresholds fixed at ±0.05
        label = "POSITIVE" if c >= 0.05 else ("NEGATIVE" if c <= -0.05 else "NEUTRAL")
        return {"compound": c, "sentiment": label}
    except:
        return {"compound": 0.0, "sentiment": "NEUTRAL"}

# Non-informative comment patterns — excluded from scoring and display
_NO_INFO_PREFIXES = (
    "no comment", "no interaction", "no opportunity", "no contact",
    "no direct", "no significant", "no exposure", "no encounter",
    "no working", "no personal", "no experience",
    "d/a", "n/a", "i have never", "never had the chance",
    "haven't had the chance", "i have not", "i did not", "i don't",
    "i do not have", "haven't worked", "have not worked",
    "not had the", "not worked with", "not interacted",
    "we have no contact", "we have no", "no working relationship",
    "not working with", "not work with", "don't work with",
    "do not work with", "never worked with", "never work with",
    "have not had", "i have not had", "unable to comment",
    "cannot comment", "can't comment", "no opportunity to",
    "i have not interacted", "no enough interaction",
    "not enough interaction", "limited interaction",
    "minimal interaction", "no sufficient",
)
_NO_INFO_EXACT = {
    "d/a","n/a","na","n.a","n.a.","-","--","---","none","nil",
    ".","..","...","no comment","no comments","no interaction",
    "no interactions","not applicable","not available","no opportunity"
}

def _is_no_info(text):
    t = str(text).strip().lower().rstrip(".,;:!?/ ")
    if t in _NO_INFO_EXACT:
        return True
    if any(t.startswith(p) for p in _NO_INFO_PREFIXES):
        return True
    # Keyword-based: short comments (<= 12 words) containing no-contact phrases
    words = t.split()
    if len(words) <= 12:
        no_contact_phrases = (
            "no contact", "no interaction", "no direct", "no working",
            "not work", "not working", "no opportunity", "no exposure",
            "no experience with", "no sufficient", "no enough",
            "not interacted", "never worked", "never interacted",
            "have no direct", "have no contact", "no relationship",
        )
        if any(p in t for p in no_contact_phrases):
            return True
    return False

def run_sentiment(df, threshold=-0.05):
    # Match notebook: score all non-empty, non-self-eval comments — no skip list
    df_s = df[
        (df.get("raters_group", pd.Series(dtype=str)) != "Faculty Self-Evaluation") &
        (df["comments"].notna()) &
        (df["comments"].astype(str).str.strip() != "")
    ].copy()
    df_s["comments"] = df_s["comments"].astype(str).str.strip()
    # Exclude non-informative comments from scoring (no contact, no interaction etc.)
    df_s = df_s[~df_s["comments"].apply(_is_no_info)].copy()
    results = df_s["comments"].apply(lambda t: score_vader(t, threshold))
    df_s = pd.concat([df_s, pd.DataFrame(results.tolist(), index=df_s.index)], axis=1)
    return df_s

def sentiment_summary(df_sent, min_comments=5, threshold=-0.05):
    # Check if any comment has compound < 0 (catches comments VADER labels NEUTRAL but are still negative-leaning)
    has_any_negative = (
        df_sent[df_sent["compound"] <= -0.05]
        .groupby("physician_id")
        .size()
        .reset_index(name="n_neg_compound")
    )
    has_any_negative["has_negative"] = True

    s = (
        df_sent.assign(is_neg=(df_sent["sentiment"]=="NEGATIVE"))
        .groupby("physician_id", as_index=False)
        .agg(total_comments=("is_neg","count"),
             negative_comments=("is_neg","sum"),
             avg_compound=("compound","mean"))
    )
    s["negative_ratio"] = s["negative_comments"] / s["total_comments"]

    # Merge in the any-negative flag
    s = s.merge(has_any_negative[["physician_id","has_negative"]], on="physician_id", how="left")
    s["has_negative"] = s["has_negative"].fillna(False)

    # Flag if physician has ANY comment with compound <= -0.05 (VADER negative threshold)
    s["negative_outlier"] = s["has_negative"]
    s = s.drop(columns=["has_negative"])
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
    # 4 independent flags — each worth 1 point
    f1 = df["low_iqr_outlier"].fillna(False).astype(bool).astype(int)  if "low_iqr_outlier"  in df.columns else pd.Series(0, index=df.index)
    f2 = df["low_z_outlier"].fillna(False).astype(bool).astype(int)    if "low_z_outlier"    in df.columns else pd.Series(0, index=df.index)
    f3 = df["low_bottom10"].fillna(False).astype(bool).astype(int)     if "low_bottom10"     in df.columns else pd.Series(0, index=df.index)
    f4 = df["negative_outlier"].fillna(False).astype(bool).astype(int) if "negative_outlier" in df.columns else pd.Series(0, index=df.index)
    df["risk_score"] = f1 + f2 + f3 + f4
    # 3-4 = Priority, 1-2 = Monitor, 0 = Clear
    df["final_flag"] = df["risk_score"] >= 3
    return df

def risk_pill(score):
    if score >= 3:   return '<span class="pill-red">⚠ Priority</span>'
    if score >= 1:   return '<span class="pill-yellow">👁 Monitor</span>'
    return '<span class="pill-green">✓ Clear</span>'

def process_dept(df_raw, dept_name, threshold=-0.05, min_f=1):
    df = clean_headers(df_raw)
    df = map_ratings(df)
    df = compute_score(df)
    df = add_year(df)
    # Aggregate and flag on 2025 only — consistent with notebook methodology
    df_2025 = df[df["year"] == 2025] if ("year" in df.columns and not df[df["year"] == 2025].empty) else df
    phys = aggregate_physician(df_2025)
    # Apply min_forms BEFORE outlier detection so thresholds are computed
    # on the same population that will appear in results (matches notebook)
    phys = phys[phys["n_forms"] >= min_f].copy().reset_index(drop=True)
    phys, mean, std = add_outlier_flags(phys)
    sent_raw = run_sentiment(df, threshold) if "comments" in df.columns else pd.DataFrame()
    if not sent_raw.empty:
        # 2025 primary, all-years fallback for physicians with no 2025 comments
        if "year" in sent_raw.columns and not sent_raw[sent_raw["year"] == 2025].empty:
            sent_2025 = sent_raw[sent_raw["year"] == 2025]
        else:
            sent_2025 = sent_raw
        sent_s_2025 = sentiment_summary(sent_2025)

        # Supplement with all-years for physicians who have no 2025 comments
        sent_s_all = sentiment_summary(sent_raw)
        missing    = ~sent_s_all["physician_id"].isin(sent_s_2025["physician_id"])
        sent_s     = pd.concat([sent_s_2025, sent_s_all[missing]], ignore_index=True)

        phys = merge_sentiment(phys, sent_s)
    else:
        phys["total_comments"]   = 0
        phys["negative_ratio"]   = np.nan
        phys["avg_compound"]     = np.nan
        phys["negative_outlier"] = False
    phys = add_risk(phys)
    phys["department"] = dept_name
    return df, phys, sent_raw

# ─── FIXED SETTINGS ──────────────────────────────────────────────────────────
min_forms   = 1
sent_thresh = -0.01

# ─── GITHUB DATA SOURCES ─────────────────────────────────────────────────────
# Replace each value below with your raw GitHub URL
# Raw URL format: https://raw.githubusercontent.com/<user>/<repo>/main/<path>.csv

GITHUB_URLS = {
    # ── Behaviour survey CSVs (3 departments × 3 years) ──────────────────────
    "aubmc_23": "AUBMC, Behavior survey responses, 2023.csv",
    "aubmc_24": "AUBMC, Behavior survey responses, 2024.csv",
    "aubmc_25": "AUBMC, Behavior raw data 2025.csv",
    "ed_23":    "ED, Behavior survey responses, 2023.csv",
    "ed_24":    "ED, Behavior survey responses, 2024.csv",
    "ed_25":    "ED, Behavior raw data 2025.csv",
    "patho_23": "Patho & Lab, Behavior survey responses, 2023.csv",
    "patho_24": "Patho,lab behavior survey responses, 2024.csv",
    "patho_25": "Patho,Lab, Behavior raw data 2025.csv",

    # ── Physicians Indicators CSV (Tab 6 — Departments & Divisions) ───────────
    "indicators": "Physicians indicators.csv",

    # ── Physician lookup CSVs (Name, Department, Division per year) ──────────
    "lookup_2023": "Datasource, cycle 2023.csv",
    "lookup_2024": "Datasource, cycle 2024.csv",
    "lookup_2025": "Datasource, cycle 2025.csv",
}

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_from_github(urls, min_f, threshold, _version="v5.4"):
    def fetch(url):
        if not url or url.startswith("REPLACE"):
            return None
        try:
            return pd.read_csv(url)
        except Exception as e:
            st.warning(f"Could not load {url}: {e}")
            return None

    # Year map — explicit year tag per file key (matches notebook tag_year approach)
    # This ensures rows with unparseable fillout_date still get the correct year
    KEY_YEAR = {
        "aubmc_23": 2023, "aubmc_24": 2024, "aubmc_25": 2025,
        "ed_23":    2023, "ed_24":    2024, "ed_25":    2025,
        "patho_23": 2023, "patho_24": 2024, "patho_25": 2025,
    }

    def load_dept(keys, name):
        frames = []
        for k in keys:
            df = fetch(urls[k])
            if df is not None:
                # Tag year explicitly — overrides fillout_date parsing for rows with bad dates
                df["_explicit_year"] = KEY_YEAR.get(k, None)
                frames.append(df)
        if not frames: return None, None, None
        raw = pd.concat(frames, ignore_index=True)
        return process_dept(raw, name, threshold, min_f=min_f)

    aubmc_raw, aubmc_phys, aubmc_sent = load_dept(["aubmc_23","aubmc_24","aubmc_25"], "AUBMC")
    ed_raw,    ed_phys,    ed_sent    = load_dept(["ed_23","ed_24","ed_25"],           "ED")
    patho_raw, patho_phys, patho_sent = load_dept(["patho_23","patho_24","patho_25"], "Pathology")

    return {
        "AUBMC":     (aubmc_raw, aubmc_phys, aubmc_sent),
        "ED":        (ed_raw,    ed_phys,    ed_sent),
        "Pathology": (patho_raw, patho_phys, patho_sent),
    }

# ── PROCESS DATA ─────────────────────────────────────────────────────────────
with st.spinner("Loading data from GitHub and running VADER sentiment analysis..."):
    data = load_from_github(
        GITHUB_URLS,
        min_forms,
        sent_thresh,
        _version="v5.4"
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

# ── Load physician lookup (Name, Department, Division) ───────────────────────
@st.cache_data(show_spinner=False)
def load_physician_lookup(urls, _version="v1.1"):
    """Load and merge the three annual lookup CSVs into one physician→dept/div map."""
    frames = []
    for key in ["lookup_2023", "lookup_2024", "lookup_2025"]:
        url = urls.get(key, "")
        if not url or url.startswith("REPLACE"):
            continue
        try:
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip()
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["physician_id", "FullName", "Department", "Division"])
    lookup = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["OriginalID"])
    lookup["OriginalID"] = lookup["OriginalID"].astype(str).str.strip().str.lower()
    lookup["Division"] = lookup["DIVISION"].fillna(lookup["DEPARTMENT"])
    lookup["Division"] = lookup["Division"].where(
        lookup["Division"].astype(str).str.strip() != "", lookup["DEPARTMENT"])
    lookup = lookup.rename(columns={"DEPARTMENT": "Department", "FullName": "FullName"})
    lookup["physician_id_key"] = lookup["OriginalID"]
    return lookup[["physician_id_key", "FullName", "Department", "Division"]].drop_duplicates(subset=["physician_id_key"])

physician_lookup = load_physician_lookup(GITHUB_URLS, _version="v1.1")

# Merge Department + Division onto all_phys using suffix key (Data29_aa → aa)
if not physician_lookup.empty:
    all_phys["_key"] = all_phys["physician_id"].astype(str).str.split("_", n=1).str[-1].str.lower()
    all_phys = all_phys.merge(
        physician_lookup[["physician_id_key", "FullName", "Department", "Division"]],
        left_on="_key", right_on="physician_id_key", how="left"
    ).drop(columns=["_key", "physician_id_key"], errors="ignore")
    all_phys["Division"] = all_phys["Division"].fillna(all_phys["Department"])
else:
    all_phys["FullName"]   = ""
    all_phys["Department"] = ""
    all_phys["Division"]   = ""


# ─── MAIN HEADER ─────────────────────────────────────────────────────────────
_logo_b64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCACUAPwDASIAAhEBAxEB/8QAGwABAAMBAQEBAAAAAAAAAAAAAAQFBgIDAQf/xAA3EAABBAIBAwIFAgMHBQEAAAABAAIDBAUREgYhMRNBFCJRYXEygRVCkQcjM1JisdEWJFNyk6L/xAAaAQEAAwEBAQAAAAAAAAAAAAAAAQIDBAUH/8QALBEAAgECAwYGAgMAAAAAAAAAAAECAxESITEEQVGRsfATYYGhweEU8TJx0f/aAAwDAQACEQMRAD8AhIiL6MfOgiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCJ4Xo2GZzeTYZHDW9hhI0gPNF9cC13FwLSPYjRXxAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAXtSq2LtqOrUidLPK7ixjR5P/H3XitJ00xlPDW8m6VteWeUUoZyd+kCNvdod+40NqlSeCN0aU4Y5WZMrVMfidsqRxX7rAWvuSs5RcjsBsUZ/Udgjkd+D28KTFkskx0osZO7GNubFGyyyEcNjjto8dg7t91VZe7M1teCgxzp7DAyARMJe2Ini1rAO/J/k67nYC/Tuh+hMJQxdGt1BiH28tmWODmWYy11dg7k/qcAQS0dtO7+B315tepGlDHUzb5+Z6VCnKrPBTyS5eRhZrMlqLhlII8lzDSWytayU6/ySt8nxoFZ/PYhtONl2lI+ahK4tBc3ToX/APjf9Hf7qz6nxNzozqGTG23RzUpHEtbHJyHDf37tePbeide6l0jDZuSYSV8chvRugDPU1ylA3DKB/KN6G/ufC2hLAlODvF5+hjOONuE1aSy9fsy2Eo/xLLV6XMxskd/eSBu/TYBtzte+mglXmN6RfZ6gzWBnsOhvUBIIRoFszmn5R9uQ0R+VE6SqmzFnW/EmuYMVLZeRC15eyNzSWAkgsJJb3H0WgxFm7l8gyYZmtRyMtOGaOxLUYxrpWyGOKMvHcbLQOR+wIV69Sabwuy+eRnQpwaWJXfwRKXRtaW509UmvTtly0M0knFjdQuYCePfz4UbHdMVrtrI1o33nT06AtiFgjLnuL2tDAd67h7SDv6hXeOOZGRjjtW308hiaVq3wloRl0Tt6kZ3Pfly/V91B5WKuHrZGvdYzHZLFTR2Hx0I2uhLJO8PY/qMhj+YaPzNPgLFVKl7Yu7t8OBu6VJL+PdkuPEg2el6sMvVETMi+wcJIyON8bQGzOc8RkHfcFrjo/wDqVC6ixNDC3rmLntWH3qrW8iGD0i8tDi0e+gCO60fUT7sNE3bV9z29Q06ctiT+HxD1HSEkMJB3zaI3O5a7kDxtSeocRax+bqw3M3TbPatfBGxfosjfJHG0ES7JPJm+LNn30NlTCvJNYpdpK+7++ZWVCNnhj227b/65Gau9MPoPwLrckvo5RwZIWM+aF/Pi5vfyRsH7rnP9ONxNWa18S+1VkMbsfaiA9Kwx3Le/cObx0W+xIV/PBla8WYrXJpK78M6PJtgtUI3ble8N2Hb1x3ogt2CB4VbfhpQdJQazl5+Lv3LM0MHwbQBNAA3YHP5eYk1oeNDf1F4VZtq8t/3w4FZ0YJOy9/TqZFFpct0vHVmrx07z7rcg2N+Me1jW/EtI+cuHI8OBBaR37hV9jA3a+LdlJHwOqNAdyjc4l4+3y/76XTGtCVrM5ZUZxvdFUi2mc6Jr4yxCz+JTStfkK1J2oWnQmYHB/Zx463ri7Rd3IPYrhnR9V3WMfTbr1qOR89mMSOhaeTInuY2QAO8OdG8aOiNDztUW1Umrp5al3stVSwtZ6GORa3EdIwZLpqPNNvz14nCcyPkgBjhEY2Obge3LYH5+qh3cDUqtxDjZsuGRxzruuDQWaY5wb57/AKdb7efCstopttXIez1Ek7a/P7M8i0U2FwzOnm5lmUuOhfdlpR7rAbeyESBx+bYaS4N+o8/ZZ1aQmp6Gc4OGoREVigREQBERAEREAREQBa7pqYt6frvieGOhyBDyzs4B0fYnsfx4WRVx0vfr1p56V97mUbrAyVwG/TeDtj/2Pn7FY144oG1CWGeZ7Wo6By2PkyzbBpPhYyUQhnIlnykfP8vkd9+3sV+618nhMhXw8OIvVWV6dhrppq0kbpYiWHhvhpoDtEH20O40vxXJVjWmdj8hEZXSO9Q6eNvPvLG7wd9vlHnX187jofKdBYbGWRHWyPqWQWTmTkdkNc0tGh2/WV5e3Q8SEZK7a0toersU/DqSTsk9b6lb/bjnMfkr7aFbI5e7ZY4HgbEZqR79g1g+Z33O/wAqvxDZBlcduZohjvxvkaT8uoWfO49vAAO+4/Cq8k3pyjd9bAVLcULHemx1h+/TPg8Qe7nD6a7L0yczcPjpHu5R5C3B6NeMkEwwHfNzvcOd4APfRK3hSUaUacfcxnUcqkqk/bvMi9AV57/UktOrbhqOt15YSJYucMwfpvpPGwQ1xLRsdwdePKs6OBs26GMbckirOzVk481xWJ+EEfztH6wQdnffZ7+TtZjCTXoXzjHwB8sjWs5hu3MPMObxPs7k0a/CurHU/UkdmGxZqRGaK+++xz6+9TOaNn8cS3t+FvVhUc3ga7++hz0ZU1BY0/ju1+Zd1hkTl70drKySWquGlZI6zUDnursdx49n/wAw7hx79vZVVfD2X9MTQDKSjDHHzZ2JrYQXOfEWxPYQT2cNgeSDrar2Z/L1HySDH14jbhkicXViPUZI7bvPnv8A0XVLM5yUWYoqsToRipahiMOmR1v1PDR7EkAk+VRUqkc1bdwLurTlln7k/qKzLjp/4ZlH1Bk6WOrxBrKhLI2hrZWQc+e9hrxtwHfZG9KfkJc9ZfhJMlLSkizFwyd6ocK8zgGEaJ/maWkjx7jus7eyOVFiM3aUFm2a0cZmkrl0jmcAGh31dx03fnsp8uU6lihgD6kVjdv46NnoueYJQdAa/l1/l/Ch03aOl/rcSqivLW33v6HneyOWizNvpmq6vYbOY8Sxr4yBwa8NjY3biWjlo7JJ+pXdmtWms1OjX5HZpZI143tq6HOV/GVzXF2yOQb2IHYdteD4ZKHOXzFcfWr153XS/lAzjJ67iNk/ggfjS5jy2ZmzxnixtV+RktNldIK+jJIx3LffQG3DZ8bIV0sk42y66X5FG83ivZ9Nbcy1vYt9rL0On7eVFB+OjnjqyyV/TDGskdpxcHns5wcd+R9NKFPhrAt1sTn8lfhtWbDYjWkj9RhfzDAd8htujvY9j2XFmTP2WyzS4muXXHTGTUHzH1Dt7vPjZOvpslRMjlM3HksfHceLVjDabA5zeegHcgCfJAPjaiEZ6Jr271JnKGri/fy+NDXzPyGSyOYigyLZrdCSGxZiioBpsywyiKFp2/uG8muA2B533TBQXLtiTqDH36sU4y1hr7BoaAkdE6R80h5nTe7hodgSSB7rIjLZdkOYyTqkIiycjYbbnREN5E+oA3v27t3+ykUOqM/AY4aNSFn/AHYstjjqkhz/AE/T4692lnbXv+Vm9nnhajb24Ly7Rotop4k5X9+OW/tkWh1VlaFSCpUFOOvAZCxno8g4Sdng7PdpGu32CiyZu7JThrSNruFeJ0NeQxn1IY3b2xp3rXcjuCQDoELq9hMjFHJakrRwx+oxoaz9O3+A0D2+3kKLTx1q1eZTYzhI9xbuQFrQQD7/ALfuV2KNL+St3+zjbq5Rd+/0dvyll+CjwpZD8JHZdabph5+oW8Sd7/ygDXjsPfuoKsKuGyM4e74Z8bGMLy6RpaDrj2//AECq8eFpHDnhM5KWWIIiKxQIiIAiIgCIiAIiIAiIgLbGZ2zVqto2YIMhQaSRWstJa0n3a4ac0779jo+4Ksauew0EAihqZyo3RJjhyDXMLj5/UzYH9SswiylRhLcbRrzjozQSdQVq0r5cRiWV7Dncvi7UvxEwce5I2A0d/wDSSPqqKeWWed888jpJZDye9x2XH6lcK56PxlfLZOxBZDzHDTksaZIGbLS0AFx7AfMloUouRF51ZKJV17FiuXGvPJEXDTuDiN/0Ul+XybgB8dM0DwGO4jwB7fYAfstHN0pRnimsY7Ih1aB8nrvAdKYwyGB7mANHzkPlLdjzrx2XMnQtlkvwxyVf4l4mfDGY3APbG2NxJPhvaRvY999ln+RReb6Gv49ZadShjzWSbOJ3WPUkbGWNLxviD7j7/deL8lkJGOY+9Zc14LXAykgg+QVqz/Z7aFiaM5OANiADiY3ba4uLe49m7G+X0Xjj+i/irT6DcjE+56dR/Zjg2MTuAHt83YjwoW0bPquhZ7PtGj6majyWRjDWsvWWhvgeodD9l6jNZUOe4Xpdv/USd/sPp+yu8f0ab1aO7XysLqUsMUzZzC4BrXmZu3A92gGF3f7hc9PdN1Mnj8bbkush+JmsMex8ga9wjZscAfJ+qmVWjnfd5evwVjSrO1t/n6fJnpb12UtMtud/E7bykJ0fquzk8kRo37X/ANXf8rSz9CW4LMMMl1hLmOdLwicfTADTy/1N+cDY91IsdE16lWQW7crLNeJxkDIy5srxfkrdj/KOLW/gnZ7KHtNDL/Cfx693/pkhk8iCT8dY2fJ9QruHK34a0kMVh7fUdydICef45edfZaLK9Gw15L9qvk+WOqzzxF3w73SNMcoj48QNu/U3bh28lc/9DWm44XprzIImxvfKJIXcow1od499ghPHoNX+B4FdPLqZ1uTv+pG6SzJO1juXpzOL2O7EEEHyCCQfsSu35nKundMb9jk55f8A4h1snfj6Kw6n6ZnwNWvNPaildK4tLGtIIPEO2N+R38hUC1h4dRYo5oxn4lN4ZZMlOyF5zmOfbmkLHBzRI4uAI8HRXRyuTJJOQt7Pf/Gd/wAqGi0wR4FMcuJLlyeQlYY33J+GgOAeQ3Q8DQ7ewURERJLQhtvUIiKSAiIgCIiAIiIAiIgCIiAIiIAu45JIuXpvczk0tdo620+QfsuEQHvVuW6paa1mWHg4ubwcRokAE/uAB+y+vu3Hv5vtzud83zF5382uX9dDf4UdFGFa2JxPS5M/iuS9QyfH2OZZwLuZ2W/Q/Vecd67HL6sdudkmmjkHkHTf09/toaUdEwx4E45cS2wnUOTw8T46ckXFzWtHqxh/ADkQG78d3uP7quisTxMjZHNI0RbMYDj8pPY6/K8kUKEU27ahzk0lfQmNymSbw437A9NpYz+8PZp8j8dh/RDlMkd7v2TsPB3Ie/Igu/qQCfv3UNEwR4DHLiS4slkIpC+K9YY4lxJEh7l3d39fdH5LIyRelJesPj1x4mQka+iiIpwR4DHLie89u1YiiinsSSsiGo2udsNH2XgiKUktCG29QiIhAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQH/9k="
_hc1, _hc2 = st.columns([1, 4])
with _hc1:
    st.markdown(
        f'''<img src="data:image/png;base64,{_logo_b64}" style="width:160px; margin-top:4px;">''',
        unsafe_allow_html=True
    )
with _hc2:
    st.markdown("# Physician Performance Dashboard")
    st.markdown(f"**Projects active:** {'  ·  '.join(available_depts)}  &nbsp;&nbsp; **Physicians:** {len(all_phys):,}  &nbsp;&nbsp; **Years:** 2023–2025")
st.markdown("---")


# ── Department / Division filter helpers (from lookup) ────────────────────────
def get_dept_options(df):
    if "Department" in df.columns:
        return ["All"] + sorted(df["Department"].dropna().unique().tolist())
    return ["All"]

def get_div_options(df, dept="All"):
    if "Division" not in df.columns:
        return ["All"]
    if dept != "All":
        df = df[df["Department"] == dept]
    return ["All"] + sorted(df["Division"].dropna().unique().tolist())

def apply_dept_div_filter(df, dept, div):
    if dept != "All" and "Department" in df.columns:
        df = df[df["Department"] == dept]
    if div != "All" and "Division" in df.columns:
        df = df[df["Division"] == div]
    return df

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Summary",
    "🎯 Flagged",
    "📊 Scores",
    "💬 Sentiment",
    "📈 Trends",
    "🏢 Dept & Div",
    "🤖 Ask MC"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🔑 Key Performance Indicators</div>', unsafe_allow_html=True)

    # Project + Department + Division filters
    t1f1, t1f2, t1f3 = st.columns(3)
    with t1f1:
        t1_proj = st.selectbox("Project", ["All"] + available_depts, key="t1_proj")
    t1_pool = all_phys if t1_proj == "All" else all_phys[all_phys["department"] == t1_proj]
    with t1f2:
        t1_dept = st.selectbox("Department", get_dept_options(t1_pool), key="t1_dept")
    with t1f3:
        t1_div  = st.selectbox("Division", get_div_options(t1_pool, t1_dept), key="t1_div")
    t1_phys = apply_dept_div_filter(t1_pool, t1_dept, t1_div)

    total      = len(t1_phys)
    priority   = (t1_phys["risk_score"] >= 3).sum()
    monitor    = (t1_phys["risk_score"].between(1, 2)).sum()
    clear      = (t1_phys["risk_score"] == 0).sum()
    avg_score  = t1_phys["avg_behavior_score"].mean()
    pct_neg    = (t1_phys["negative_outlier"] == True).sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
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
    with c6:
        st.markdown(f"""<div class="metric-card {'danger' if pct_neg > 0 else 'success'}">
            <div class="metric-label">Neg. Sentiment Flags</div>
            <div class="metric-value">{pct_neg}</div>
            <div class="metric-sub">{pct_neg/total*100:.1f}% of physicians</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Score — big and prominent ───────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Risk Score Breakdown</div>', unsafe_allow_html=True)
    risk_vals = [
        int((t1_phys["risk_score"] == 0).sum()),
        int(t1_phys["risk_score"].between(1, 2).sum()),
        int((t1_phys["risk_score"] >= 3).sum()),
    ]
    total_phys_r = sum(risk_vals)
    fig_risk, ax_risk = plt.subplots(figsize=(10, 4.5))
    risk_labels_r = ["Clear (0)", "Monitor (1–2)", "Priority (3–4)"]
    risk_colors_r = ["#38a169", "#f59e0b", "#e53e3e"]
    bars_r = ax_risk.bar(risk_labels_r, risk_vals, color=risk_colors_r,
                         edgecolor="white", linewidth=2, width=0.45)
    for bar, val in zip(bars_r, risk_vals):
        pct = val / total_phys_r * 100 if total_phys_r > 0 else 0
        ax_risk.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", fontweight="900", fontsize=24, color="#1f2937")
        if val > 0:
            ax_risk.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                         f"{pct:.1f}%", ha="center", va="center",
                         fontweight="700", fontsize=15, color="white")
    ax_risk.set_ylabel("Number of Physicians", fontsize=12)
    ax_risk.set_title("Composite Risk Distribution — All Physicians", fontsize=14, fontweight="bold", pad=14)
    ax_risk.tick_params(axis="x", labelsize=14)
    ax_risk.grid(axis="y", alpha=0.3, linestyle="--")
    ax_risk.set_facecolor("white"); fig_risk.patch.set_facecolor("white")
    ax_risk.spines["top"].set_visible(False); ax_risk.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig_risk, use_container_width=True); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Department Risk Comparison ────────────────────────────────────────────
    st.markdown('<div class="section-header">🏥 Department Risk Comparison</div>', unsafe_allow_html=True)

    dept_risk_rows = []
    for dept in available_depts:
        _, phys, _ = data[dept]
        if phys is None: continue
        total_d   = len(phys)
        priority_d = int((phys["risk_score"] >= 3).sum())
        monitor_d  = int((phys["risk_score"].between(1,2)).sum())
        clear_d    = int((phys["risk_score"] == 0).sum())
        dept_risk_rows.append({
            "dept": dept, "Priority": priority_d,
            "Monitor": monitor_d, "Clear": clear_d, "total": total_d,
            "avg": round(phys["avg_behavior_score"].mean(), 2),
        })
    dept_risk_df = pd.DataFrame(dept_risk_rows)

    x      = np.arange(len(dept_risk_df))
    width  = 0.25
    fig_dr, ax_dr = plt.subplots(figsize=(10, 4.5))
    b1 = ax_dr.bar(x - width, dept_risk_df["Priority"], width, color="#e53e3e", alpha=0.88, label="Priority", edgecolor="white")
    b2 = ax_dr.bar(x,          dept_risk_df["Monitor"],  width, color="#f59e0b", alpha=0.88, label="Monitor",  edgecolor="white")
    b3 = ax_dr.bar(x + width,  dept_risk_df["Clear"],    width, color="#38a169", alpha=0.88, label="Clear",    edgecolor="white")
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_dr.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                           str(int(h)), ha="center", va="bottom", fontsize=10, fontweight="700")
    # Avg score annotation per dept
    for i, row in dept_risk_df.iterrows():
        ax_dr.text(i, dept_risk_df[["Priority","Monitor","Clear"]].iloc[i].max() + 2.5,
                   f"Avg: {row['avg']}", ha="center", fontsize=9,
                   color="#64748b", fontweight="600")
    ax_dr.set_xticks(x)
    ax_dr.set_xticklabels(dept_risk_df["dept"], fontsize=12)
    ax_dr.set_ylabel("Number of Physicians", fontsize=10)
    ax_dr.set_title("Priority · Monitor · Clear by Project", fontsize=13, fontweight="bold", pad=12)
    ax_dr.legend(fontsize=10, loc="upper right")
    ax_dr.grid(axis="y", alpha=0.3, linestyle="--")
    ax_dr.set_facecolor("white"); fig_dr.patch.set_facecolor("white")
    ax_dr.spines["top"].set_visible(False); ax_dr.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig_dr, use_container_width=True); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Flagged Physicians + Sentiment Snapshot ───────────────────────────
    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown('<div class="section-header">⚠️ Top Physicians Needing Attention</div>', unsafe_allow_html=True)
        top_flagged = (
            t1_phys[t1_phys["risk_score"] >= 1]
            .sort_values(["risk_score", "avg_behavior_score"], ascending=[False, True])
            .head(10)
        )
        if top_flagged.empty:
            st.success("No physicians flagged across all projects.")
        else:
            for _, fp in top_flagged.iterrows():
                rs = int(fp["risk_score"])
                bg     = "#fef2f2" if rs >= 3 else "#fffbeb"
                border = "#e53e3e" if rs >= 3 else "#f59e0b"
                label  = "⚠ Priority" if rs >= 3 else "👁 Monitor"
                flags  = []
                if fp.get("low_iqr_outlier", False): flags.append("IQR")
                if fp.get("low_z_outlier",   False): flags.append("Z")
                if fp.get("low_bottom10",    False): flags.append("P10")
                if fp.get("negative_outlier",False): flags.append("Sent.")
                neg_r  = f"{fp['negative_ratio']:.0%}" if pd.notna(fp.get("negative_ratio")) else "—"
                st.markdown(f"""
                <div style="background:{bg}; border-left:4px solid {border}; border-radius:8px;
                            padding:10px 16px; margin-bottom:8px; box-shadow:0 1px 3px rgba(0,0,0,0.05)">
                    <div style="display:flex; justify-content:space-between; align-items:center">
                        <span style="font-size:14px; font-weight:700; color:#111827">{fp["physician_id"]}</span>
                        <span style="font-size:12px; font-weight:700; color:{border}">{label} &nbsp;|&nbsp; Score {fp["avg_behavior_score"]:.2f}</span>
                    </div>
                    <div style="font-size:11px; color:#6b7280; margin-top:4px">
                        {fp.get("department","—")} &nbsp;·&nbsp; {int(fp["n_forms"])} evals
                        &nbsp;·&nbsp; Flags: <b>{"  ".join(flags) if flags else "—"}</b>
                        &nbsp;·&nbsp; Neg. comments: <b>{neg_r}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">💬 Sentiment Snapshot</div>', unsafe_allow_html=True)

        # Gather all sentiment data
        all_sent_frames = []
        for dn in available_depts:
            _, _, sr = data[dn]
            if sr is not None and not sr.empty:
                all_sent_frames.append(sr)

        if all_sent_frames:
            sent_all = pd.concat(all_sent_frames, ignore_index=True)
            # Filter out noise comments from display counts (same as deep-dive display filter)
            sent_all = sent_all[~sent_all["comments"].astype(str).apply(_is_no_info)].copy()
            total_c  = len(sent_all)
            neg_c    = (sent_all["sentiment"] == "NEGATIVE").sum()
            pos_c    = (sent_all["sentiment"] == "POSITIVE").sum()
            neu_c    = (sent_all["sentiment"] == "NEUTRAL").sum()
            neg_pct  = neg_c / total_c * 100 if total_c > 0 else 0
            pos_pct  = pos_c / total_c * 100 if total_c > 0 else 0
            neu_pct  = neu_c / total_c * 100 if total_c > 0 else 0

            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:20px 22px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-bottom:12px">
                <div style="font-size:12px; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:.05em">Total Comments</div>
                <div style="font-size:32px; font-weight:700; color:#111827">{total_c:,}</div>
                <div style="font-size:12px; color:#9ca3af; margin-top:2px">across all departments</div>
            </div>""", unsafe_allow_html=True)

            # Sentiment donut chart
            fig_sent, ax_sent = plt.subplots(figsize=(4, 3.2))
            sizes  = [neg_pct, neu_pct, pos_pct]
            colors = ["#e53e3e", "#9ca3af", "#38a169"]
            labels = [f"Negative\n{neg_pct:.1f}%", f"Neutral\n{neu_pct:.1f}%", f"Positive\n{pos_pct:.1f}%"]
            wedges, texts = ax_sent.pie(
                sizes, colors=colors, startangle=90,
                wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
                pctdistance=0.75,
            )
            # Labels outside
            for i, (wedge, label) in enumerate(zip(wedges, labels)):
                angle = (wedge.theta2 + wedge.theta1) / 2
                x = 1.25 * np.cos(np.radians(angle))
                y = 1.25 * np.sin(np.radians(angle))
                ha = "left" if x > 0 else "right"
                ax_sent.text(x, y, label, ha=ha, va="center",
                             fontsize=9, fontweight="700", color=colors[i])
            # Centre label
            ax_sent.text(0, 0, f"{total_c:,}\ncomments", ha="center", va="center",
                         fontsize=9, fontweight="700", color="#1a365d")
            ax_sent.set_title("Comment Sentiment Split", fontsize=10, fontweight="bold", color="#1a365d", pad=8)
            fig_sent.patch.set_facecolor("white")
            plt.tight_layout(); st.pyplot(fig_sent, use_container_width=True); plt.close()

            # Neg sentiment outlier physicians
            neg_flag_n = int(t1_phys["negative_outlier"].sum()) if "negative_outlier" in t1_phys.columns else 0
            st.markdown(f"""
            <div style="background:#fef2f2; border-left:4px solid #ef4444; border-radius:8px;
                        padding:12px 16px; margin-top:8px">
                <div style="font-size:12px; color:#6b7280; font-weight:600">Negative Sentiment Outliers</div>
                <div style="font-size:28px; font-weight:700; color:#ef4444">{neg_flag_n}</div>
                <div style="font-size:11px; color:#9ca3af">physicians above IQR upper fence on negative ratio</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No comment data available.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Department Summary Table ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Department Summary</div>', unsafe_allow_html=True)
    summary_rows = []
    for dept in available_depts:
        _, phys, _ = data[dept]
        if phys is None: continue
        row = {
            "Project":         dept,
            "Physicians":      len(phys),
            "Avg Score":       f"{phys['avg_behavior_score'].mean():.2f}",
            "Priority (3-4)":  int((phys['risk_score']>=3).sum()),
            "Monitor (1-2)":   int((phys['risk_score'].between(1,2)).sum()),
            "Clear (0)":       int((phys['risk_score']==0).sum()),
            "Sentiment Flags": int(phys['negative_outlier'].sum()) if 'negative_outlier' in phys.columns else 0,
        }
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FLAGGED PHYSICIANS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🎯 Physician Risk Register</div>', unsafe_allow_html=True)

    # ── Methodology definitions ───────────────────────────────────────────────
    _mdef_col, _mdef_btn = st.columns([4, 1])
    with _mdef_col:
        pass
    with _mdef_btn:
        with st.popover("📖 Method Definitions"):
            st.markdown("""
**🔵 IQR Lower Fence**
Flags physicians below Q1 − 1.5×IQR. Robust, non-parametric, resistant to extreme values.

---

**🟣 Z-Score (≤ −2)**
Flags physicians more than 2 standard deviations below the group mean (~2.3% of population).

---

**🟠 Bottom 10% (P10)**
Flags the lowest-scoring 10% regardless of absolute value. Always flags exactly 10% of the group.

---

**🔴 Negative Sentiment (VADER)**
Flags physicians with any 2025 peer comment scoring compound ≤ −0.05 using VADER + medical lexicon.

---

**⚠️ Composite Risk Score (0–4)**
Sum of all 4 flags:
- **0** = Clear ✓
- **1–2** = Monitor 👁
- **3–4** = Priority ⚠️
""")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dept_filter = st.selectbox("Project", ["All"] + available_depts, key="flag_dept")
    with col_f2:
        risk_filter = st.selectbox("Risk Level", ["All","Priority (3-4)","Monitor (1-2)","Clear (0)"], key="flag_risk")
    with col_f3:
        sort_by = st.selectbox("Sort by", ["Risk Score ↓", "Avg Score ↑", "Neg. Ratio ↓"], key="flag_sort")

    # Department / Division filters
    t2f1, t2f2 = st.columns(2)
    with t2f1:
        t2_dept = st.selectbox("Department", get_dept_options(all_phys), key="t2_dept")
    with t2f2:
        t2_div  = st.selectbox("Division",   get_div_options(all_phys, t2_dept), key="t2_div")

    df_view = all_phys.copy()
    if dept_filter != "All":
        df_view = df_view[df_view["department"] == dept_filter]
    df_view = apply_dept_div_filter(df_view, t2_dept, t2_div)
    if risk_filter == "Priority (3-4)":
        df_view = df_view[df_view["risk_score"] >= 3]
    elif risk_filter == "Monitor (1-2)":
        df_view = df_view[df_view["risk_score"].between(1, 2)]
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
        "department":         "Project",
        "avg_behavior_score": "Avg Score",
        "n_forms":            "Evaluations",
        "z_score":            "Z-Score",
        "low_iqr_outlier":    "IQR Flag",
        "low_z_outlier":      "Z-Flag",
        "low_bottom10":       "Bottom 10%",
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
    if "Avg Compound" in show_df.columns:
        show_df["Avg Compound"] = show_df["Avg Compound"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

    st.dataframe(
        show_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=4, format="%d"
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

    # Row 1: Project + Year
    dd1, dd2 = st.columns(2)
    with dd1:
        dd_dept = st.selectbox("Project", available_depts, key="deep_dept")
    with dd2:
        raw_dd, phys_dd_all, sent_dd = data[dd_dept]
        if raw_dd is not None and "year" in raw_dd.columns:
            yr_opts = ["All Years"] + sorted(raw_dd["year"].dropna().unique().astype(int).tolist(), reverse=True)
        else:
            yr_opts = ["All Years"]
        dd_year = st.selectbox("Year", yr_opts, key="deep_year")

    # Row 2: Department + Division + Physician (cascading)
    dd3, dd4, dd5 = st.columns(3)
    with dd3:
        dd_dept_filter = st.selectbox("Department", get_dept_options(all_phys), key="deep_dept_f")
    with dd4:
        dd_div_filter  = st.selectbox("Division", get_div_options(all_phys, dd_dept_filter), key="deep_div_f")

    # Build physician list filtered by year + department + division
    if raw_dd is not None:
        dd_raw_filt = raw_dd if dd_year == "All Years" else raw_dd[raw_dd["year"] == int(dd_year)]
        # Apply dept/div filter via all_phys lookup
        phys_pool = apply_dept_div_filter(all_phys, dd_dept_filter, dd_div_filter)["physician_id"].unique()
        dd_raw_filt = dd_raw_filt[dd_raw_filt["physician_id"].isin(phys_pool)] if dd_dept_filter != "All" or dd_div_filter != "All" else dd_raw_filt
        phys_in_yr = sorted(dd_raw_filt["physician_id"].dropna().unique().tolist())
    else:
        phys_in_yr = []
    with dd5:
        if phys_in_yr:
            selected_id = st.selectbox("Physician ID", ["— Select —"] + phys_in_yr, key="deep_id")
            if selected_id == "— Select —":
                selected_id = None
        else:
            selected_id = None
            st.info("No physicians for this selection.")

    if selected_id and raw_dd is not None:
        # Get aggregated row — use year-filtered data if year selected
        if dd_year == "All Years":
            phys_src = phys_dd_all
        else:
            yr_raw = raw_dd[raw_dd["year"] == int(dd_year)]
            if not yr_raw.empty:
                phys_src = aggregate_physician(yr_raw)
                phys_src, _, _ = add_outlier_flags(phys_src)
                # Re-merge 2025 sentiment for this year's risk computation
                if sent_dd is not None and not sent_dd.empty and int(dd_year) == 2025:
                    sent_yr = sentiment_summary(sent_dd)
                    phys_src = merge_sentiment(phys_src, sent_yr)
                else:
                    phys_src["negative_outlier"] = False
                phys_src = add_risk(phys_src)
                phys_src["department"] = dd_dept
            else:
                phys_src = phys_dd_all

        # Merge Department + Division from all_phys lookup onto phys_src
        if phys_src is not None and "Department" not in phys_src.columns:
            lookup_cols = all_phys[["physician_id","Department","Division","FullName"]].drop_duplicates("physician_id")
            phys_src = phys_src.merge(lookup_cols, on="physician_id", how="left")

        row_mask = phys_src["physician_id"] == selected_id if phys_src is not None else pd.Series(False)
        if phys_src is None or not row_mask.any():
            st.warning(f"No data found for {selected_id}.")
        else:
            row = phys_src[row_mask].iloc[0]
            year_label = f" — {dd_year}" if dd_year != "All Years" else " — All Years"

            dc1, dc2, dc3, dc4 = st.columns(4)
            with dc1: st.metric("Project", dd_dept)
            with dc2: st.metric(f"Avg Score{year_label}", f"{row['avg_behavior_score']:.3f} / 4.0")
            with dc3: st.metric("Evaluations", int(row["n_forms"]))
            with dc4: st.metric("Risk Score (0–4)", f"{int(row['risk_score'])} / 4")

            dept_val = row.get("Department", "") or "—"
            div_val  = row.get("Division",   "") or "—"
            name_val = row.get("FullName",   "") or "—"
            dd_info1, dd_info2, dd_info3 = st.columns(3)
            with dd_info1: st.metric("Physician Name", name_val)
            with dd_info2: st.metric("Department",     dept_val)
            with dd_info3: st.metric("Division",       div_val)

            dc5, dc6, dc7, dc8 = st.columns(4)
            with dc5: st.metric("Z-Score", f"{row.get('z_score', 0):.2f}")
            with dc6:
                iqr_dd = "🔴 YES" if row.get("low_iqr_outlier", False) else "🟢 No"
                st.metric("IQR Outlier", iqr_dd)
            with dc7:
                neg_r = row.get("negative_ratio", np.nan)
                st.metric("Neg. Comment Ratio", f"{neg_r:.1%}" if pd.notna(neg_r) else "—")
            with dc8:
                neg_s = "🔴 YES" if row.get("negative_outlier", False) else "🟢 No"
                st.metric("Neg. Sentiment", neg_s)

            # Comments — filter by year if selected
            if sent_dd is not None and not sent_dd.empty and "physician_id" in sent_dd.columns:
                phys_comments = sent_dd[sent_dd["physician_id"] == selected_id].copy()
                if dd_year != "All Years" and "year" in phys_comments.columns:
                    phys_comments = phys_comments[phys_comments["year"] == int(dd_year)]
                if not phys_comments.empty:
                    yr_suffix = f", {dd_year}" if dd_year != "All Years" else ""
                    phys_comments_display = phys_comments[~phys_comments["comments"].astype(str).apply(_is_no_info)]
                    # Check if flag is triggered by a comment hidden from display
                    hidden_neg = phys_comments[
                        phys_comments["comments"].astype(str).apply(_is_no_info) &
                        (phys_comments["compound"] <= -0.05)
                    ]
                    if not hidden_neg.empty:
                        st.warning(f"⚠️ Negative sentiment flag is triggered by {len(hidden_neg)} non-informative comment(s) being scored (e.g. 'Not working with him'). These are excluded from display but were scored before filtering. Consider clearing the cache to recompute.")
                    st.markdown(f"**Peer Comments** ({len(phys_comments_display)} total{yr_suffix}):")
                    for _, crow in phys_comments_display.sort_values("compound").iterrows():
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
                    st.info("No comments available for this selection.")
    elif not selected_id:
        st.info("No physicians match the current filter.")



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEPARTMENT VIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📊 Project-Level Analysis</div>', unsafe_allow_html=True)

    dept_sel = st.selectbox("Select Project", available_depts, key="dept_view")

    # Department / Division filters — cascade under project
    t3f1, t3f2 = st.columns(2)
    # Only show depts/divs that belong to the selected project
    proj_phys = all_phys[all_phys["department"] == dept_sel] if "department" in all_phys.columns else all_phys
    with t3f1:
        t3_dept = st.selectbox("Department", get_dept_options(proj_phys), key="t3_dept")
    with t3f2:
        t3_div  = st.selectbox("Division",   get_div_options(proj_phys, t3_dept), key="t3_div")

    # Filter all_phys by project + department + division — single source of truth
    phys_d = proj_phys.copy()
    phys_d = apply_dept_div_filter(phys_d, t3_dept, t3_div)

    if phys_d is None or phys_d.empty:
        st.warning("No data available for this project.")
    else:
        # Recompute outlier flags on the filtered subset so IQR/Z thresholds
        # reflect the currently visible population
        if len(phys_d) >= 4:
            phys_d, _, _ = add_outlier_flags(phys_d)
            phys_d = add_risk(phys_d)

        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Physicians", len(phys_d))
        with d2: st.metric("Project Mean Score", f"{phys_d['avg_behavior_score'].mean():.3f}")
        with d3: st.metric("IQR Outliers", int(phys_d["low_iqr_outlier"].sum()) if "low_iqr_outlier" in phys_d.columns else 0)
        with d4: st.metric("Priority Flags", int((phys_d["risk_score"]>=3).sum()))

        col_l, col_r = st.columns(2)

        # IQR scatter plot
        with col_l:
            st.markdown("**IQR Outlier View — Score Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            scores_d  = phys_d["avg_behavior_score"]
            Q1d, Q3d  = scores_d.quantile(0.25), scores_d.quantile(0.75)
            iqr_fence = Q1d - 1.5 * (Q3d - Q1d)
            normal    = phys_d[~phys_d["low_iqr_outlier"]] if "low_iqr_outlier" in phys_d.columns else phys_d
            outliers  = phys_d[phys_d["low_iqr_outlier"]]  if "low_iqr_outlier" in phys_d.columns else phys_d.iloc[0:0]
            ax.scatter(normal.index,  normal["avg_behavior_score"],
                       alpha=0.6, color="#2b7bc8", s=55, label="Within range", zorder=3)
            ax.scatter(outliers.index, outliers["avg_behavior_score"],
                       color="#e53e3e", s=100, zorder=5, label=f"IQR Outliers (n={len(outliers)})")
            ax.axhline(iqr_fence, color="#e53e3e", linewidth=2, linestyle="--",
                       label=f"IQR Lower Fence ({iqr_fence:.2f})")
            ax.axhline(scores_d.mean(), color="#1a365d", linewidth=1.5, linestyle=":",
                       label=f"Mean ({scores_d.mean():.2f})")
            ax.set_xlabel("Physician Index", fontsize=10)
            ax.set_ylabel("Avg Behaviour Score", fontsize=10)
            ax.set_title(f"{dept_sel}{" · " + t3_dept if t3_dept != "All" else ""}{" / " + t3_div if t3_div != "All" else ""} — IQR Score Outliers", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, linestyle="--")
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Within-dept colleague comparison histogram
        with col_r:
            st.markdown("**Score Distribution — Colleague Comparison**")
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            scores = phys_d["avg_behavior_score"]
            n, bins, patches = ax2.hist(scores, bins=20, edgecolor="white",
                                         linewidth=0.8, color="#2b7bc8", alpha=0.75)

            # Colour IQR outliers red in the histogram
            Q1h, Q3h   = scores.quantile(0.25), scores.quantile(0.75)
            iqr_thresh = Q1h - 1.5 * (Q3h - Q1h)
            for patch, left_edge in zip(patches, bins[:-1]):
                if left_edge < iqr_thresh:
                    patch.set_facecolor("#e53e3e")
                    patch.set_alpha(0.8)

            ax2.axvline(scores.mean(), color="#1a365d", linewidth=2,
                        linestyle="-", label=f"Mean ({scores.mean():.2f})")
            ax2.axvline(scores.quantile(0.10), color="#f59e0b", linewidth=1.5,
                        linestyle=":", label=f"10th pct ({scores.quantile(.1):.2f})")

            red_patch   = mpatches.Patch(color="#e53e3e", alpha=0.8, label="Below IQR fence")
            blue_patch  = mpatches.Patch(color="#2b7bc8", alpha=0.75, label="Within range")
            ax2.legend(handles=[red_patch, blue_patch] +
                       [plt.Line2D([0],[0],color="#1a365d",linewidth=2,label=f"Mean ({scores.mean():.2f})"),
                        plt.Line2D([0],[0],color="#f59e0b",linewidth=1.5,linestyle=":",label=f"10th pct")],
                       fontsize=8)

            ax2.set_xlabel("Avg Behaviour Score (0–4)", fontsize=10)
            ax2.set_ylabel("Number of Physicians", fontsize=10)
            ax2.set_title(f"{dept_sel}{" · " + t3_dept if t3_dept != "All" else ""}{" / " + t3_div if t3_div != "All" else ""} — Colleague Comparison", fontsize=11, fontweight="bold")
            ax2.grid(axis="y", alpha=0.3, linestyle="--")
            ax2.set_facecolor("white")
            fig2.patch.set_facecolor("white")
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Outlier method comparison table
        mc_title, mc_help = st.columns([3, 1])
        with mc_title:
            st.markdown("**Outlier Method Comparison**")
        with mc_help:
            with st.popover("ℹ️ Method definitions"):
                st.markdown("""
**IQR Lower Fence**
Flags scores below Q1 − 1.5×IQR. Robust to extreme values.

**Z-Score ≤ −2**
More than 2 standard deviations below the mean (~2.3% of population).

**Bottom 10%**
Lowest-scoring 10% regardless of absolute value. Always flags exactly 10%.

**Neg. Sentiment**
Any 2025 peer comment with VADER compound score ≤ −0.05.

**Risk Score**
Sum of all 4 flags (0–4). Priority = 3 or 4.
""")
        method_df = pd.DataFrame({
                "Method":       ["IQR Lower Fence", "Z-Score (≤−2)", "Bottom 10%", "Neg. Sentiment"],
                "Flag Column":  ["low_iqr_outlier", "low_z_outlier", "low_bottom10", "negative_outlier"],
            })
        method_df["Physicians Flagged"] = method_df["Flag Column"].apply(
            lambda c: int(phys_d[c].sum()) if c in phys_d.columns else 0
        )
        method_df["% of Project"] = (
            method_df["Physicians Flagged"] / max(len(phys_d), 1) * 100
        ).round(1).astype(str) + "%"
        st.dataframe(method_df[["Method","Physicians Flagged","% of Project"]],
                     use_container_width=True, hide_index=True)

        # Within-dept ranking table
        st.markdown("**Physician Ranking within Department**")
        rank_cols = ["physician_id","avg_behavior_score","n_forms","z_score",
                     "low_iqr_outlier","low_z_outlier","low_bottom10",
                     "negative_outlier","risk_score"]
        rank_cols = [c for c in rank_cols if c in phys_d.columns]
        rank_df = phys_d[rank_cols].copy()
        rank_df = rank_df.sort_values("avg_behavior_score")
        rank_df["Percentile"] = (rank_df["avg_behavior_score"].rank(pct=True)*100).round(1).astype(str) + "%"
        rank_df["avg_behavior_score"] = rank_df["avg_behavior_score"].round(3)
        if "z_score" in rank_df.columns:
            rank_df["z_score"] = rank_df["z_score"].round(2)
        col_rename = {
            "physician_id":       "Physician ID",
            "avg_behavior_score": "Avg Score",
            "n_forms":            "Evaluations",
            "z_score":            "Z-Score",
            "low_iqr_outlier":    "IQR Flag",
            "low_z_outlier":      "Z-Flag",
            "low_bottom10":       "Bottom 10%",
            "negative_outlier":   "Neg. Sentiment",
            "risk_score":         "Risk Score",
        }
        rank_df = rank_df.rename(columns={k:v for k,v in col_rename.items() if k in rank_df.columns})
        st.dataframe(rank_df.reset_index(drop=True), use_container_width=True, hide_index=True,
                     column_config={"Risk Score": st.column_config.ProgressColumn(min_value=0, max_value=4, format="%d"),
                                    "Avg Score":  st.column_config.ProgressColumn(min_value=0, max_value=4, format="%.3f")})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENTIMENT EXPLORER (placeholder to close the with tab3 block)
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENTIMENT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">💬 Sentiment Analysis</div>', unsafe_allow_html=True)

    # Gather sentiment data across all departments
    sent_frames = []
    for dn in available_depts:
        _, _, sr = data[dn]
        if sr is not None and not sr.empty:
            sr2 = sr.copy()
            sr2["dept"] = dn
            sent_frames.append(sr2)

    if not sent_frames:
        st.info("No comment data available. Upload behaviour survey CSVs to enable sentiment analysis.")
    else:
        all_sent_raw = pd.concat(sent_frames, ignore_index=True)

        # ── No-comment / no-meaningful-comment stats ──────────────────────────
        total_raw_comments   = len(all_sent_raw)
        no_info_mask         = all_sent_raw["comments"].astype(str).apply(_is_no_info)
        empty_mask           = all_sent_raw["comments"].isna() | (all_sent_raw["comments"].astype(str).str.strip() == "")
        no_info_count        = no_info_mask.sum()
        empty_count          = empty_mask.sum()
        meaningful_count     = (~no_info_mask & ~empty_mask).sum()
        no_info_pct          = no_info_count / total_raw_comments * 100 if total_raw_comments > 0 else 0
        empty_pct            = empty_count  / total_raw_comments * 100 if total_raw_comments > 0 else 0
        meaningful_pct       = meaningful_count / total_raw_comments * 100 if total_raw_comments > 0 else 0

        st.markdown('<div class="section-header">📊 Comment Coverage Overview</div>', unsafe_allow_html=True)
        cov1, cov2, cov3, cov4 = st.columns(4)
        with cov1:
            st.markdown(f'''<div class="metric-card neutral">
                <div class="metric-label">Total Comment Fields</div>
                <div class="metric-value">{total_raw_comments:,}</div>
                <div class="metric-sub">all survey responses</div>
            </div>''', unsafe_allow_html=True)
        with cov2:
            st.markdown(f'''<div class="metric-card success">
                <div class="metric-label">Meaningful Comments</div>
                <div class="metric-value">{meaningful_count:,}</div>
                <div class="metric-sub">{meaningful_pct:.1f}% — scored by VADER</div>
            </div>''', unsafe_allow_html=True)
        with cov3:
            st.markdown(f'''<div class="metric-card warning">
                <div class="metric-label">No-Contact Comments</div>
                <div class="metric-value">{no_info_count:,}</div>
                <div class="metric-sub">{no_info_pct:.1f}% — e.g. "N/A", "Not working with"</div>
            </div>''', unsafe_allow_html=True)
        with cov4:
            st.markdown(f'''<div class="metric-card neutral">
                <div class="metric-label">Empty / Blank</div>
                <div class="metric-value">{empty_count:,}</div>
                <div class="metric-sub">{empty_pct:.1f}% — no response provided</div>
            </div>''', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Filter noise comments for sentiment analysis
        all_sent = all_sent_raw[~no_info_mask & ~empty_mask].copy()

        # Project + Department + Division filters
        t4f1, t4f2, t4f3 = st.columns(3)
        with t4f1:
            t4_proj = st.selectbox("Project", ["All"] + available_depts, key="t4_proj")
        with t4f2:
            t4_dept = st.selectbox("Department", get_dept_options(all_phys), key="t4_dept")
        with t4f3:
            t4_div  = st.selectbox("Division", get_div_options(all_phys, t4_dept), key="t4_div")
        # Apply project filter first
        if t4_proj != "All":
            all_sent = all_sent[all_sent["dept"] == t4_proj]
        # Filter by dept/div via physician lookup
        phys_filtered = apply_dept_div_filter(all_phys, t4_dept, t4_div)
        all_sent = all_sent[all_sent["physician_id"].isin(phys_filtered["physician_id"])] if t4_dept != "All" or t4_div != "All" else all_sent

        # ── KPI row ───────────────────────────────────────────────────────────
        total_c = len(all_sent)
        neg_c   = (all_sent["sentiment"] == "NEGATIVE").sum()
        pos_c   = (all_sent["sentiment"] == "POSITIVE").sum()
        neu_c   = (all_sent["sentiment"] == "NEUTRAL").sum()
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f'''<div class="metric-card neutral">
                <div class="metric-label">Total Comments</div>
                <div class="metric-value">{total_c:,}</div>
                <div class="metric-sub">scored by VADER</div>
            </div>''', unsafe_allow_html=True)
        with sc2:
            st.markdown(f'''<div class="metric-card danger">
                <div class="metric-label">🔴 Negative</div>
                <div class="metric-value">{neg_c:,}</div>
                <div class="metric-sub">{neg_c/total_c*100:.1f}% of comments</div>
            </div>''', unsafe_allow_html=True)
        with sc3:
            st.markdown(f'''<div class="metric-card success">
                <div class="metric-label">🟢 Positive</div>
                <div class="metric-value">{pos_c:,}</div>
                <div class="metric-sub">{pos_c/total_c*100:.1f}% of comments</div>
            </div>''', unsafe_allow_html=True)
        with sc4:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-label">⚪ Neutral</div>
                <div class="metric-value">{neu_c:,}</div>
                <div class="metric-sub">{neu_c/total_c*100:.1f}% of comments</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("---")

        # ── Chart 1: Sentiment breakdown by department (stacked bar) ─────────
        st.markdown('<div class="section-header">📊 Sentiment Breakdown by Department</div>', unsafe_allow_html=True)

        dept_sent = (
            all_sent.groupby(["dept","sentiment"], as_index=False)
            .size()
            .rename(columns={"size":"count"})
        )
        dept_totals = dept_sent.groupby("dept")["count"].transform("sum")
        dept_sent["pct"] = dept_sent["count"] / dept_totals * 100

        depts_order  = dept_sent.groupby("dept")["count"].sum().sort_values(ascending=False).index.tolist()
        neg_pct  = dept_sent[dept_sent["sentiment"]=="NEGATIVE"].set_index("dept")["pct"].reindex(depts_order).fillna(0)
        pos_pct  = dept_sent[dept_sent["sentiment"]=="POSITIVE"].set_index("dept")["pct"].reindex(depts_order).fillna(0)
        neu_pct  = dept_sent[dept_sent["sentiment"]=="NEUTRAL"].set_index("dept")["pct"].reindex(depts_order).fillna(0)
        neg_cnt  = dept_sent[dept_sent["sentiment"]=="NEGATIVE"].set_index("dept")["count"].reindex(depts_order).fillna(0)
        pos_cnt  = dept_sent[dept_sent["sentiment"]=="POSITIVE"].set_index("dept")["count"].reindex(depts_order).fillna(0)

        fig_sb, ax_sb = plt.subplots(figsize=(10, max(4.5, len(depts_order)*0.55)))
        y = list(range(len(depts_order)))
        ax_sb.barh(y, neg_pct.values, color="#e53e3e", alpha=0.88, label="Negative", height=0.6)
        ax_sb.barh(y, neu_pct.values, left=neg_pct.values, color="#9ca3af", alpha=0.75, label="Neutral", height=0.6)
        ax_sb.barh(y, pos_pct.values, left=(neg_pct+neu_pct).values, color="#38a169", alpha=0.88, label="Positive", height=0.6)

        # Label all 3 segments
        for i in range(len(depts_order)):
            nv = neg_pct.values[i]
            uv = neu_pct.values[i]
            pv = pos_pct.values[i]
            if nv >= 4:
                ax_sb.text(nv/2, i, f"{nv:.1f}%", va="center", ha="center",
                           fontsize=8, fontweight="700", color="white")
            if uv >= 6:
                ax_sb.text(nv + uv/2, i, f"{uv:.1f}%", va="center", ha="center",
                           fontsize=8, fontweight="700", color="white")
            if pv >= 4:
                ax_sb.text(nv + uv + pv/2, i, f"{pv:.1f}%", va="center", ha="center",
                           fontsize=8, fontweight="700", color="white")

        ax_sb.set_yticks(y)
        ax_sb.set_yticklabels(depts_order, fontsize=10)
        ax_sb.set_xlabel("% of Comments", fontsize=10, color="#64748b")
        ax_sb.set_title("Sentiment Breakdown by Project (% of comments)", fontsize=12, fontweight="bold", color="#1a365d")
        ax_sb.legend(fontsize=9, loc="lower right", framealpha=0.9)
        ax_sb.set_xlim(0, 100)
        ax_sb.grid(axis="x", alpha=0.2, linestyle="--", color="#bfdbfe")
        ax_sb.tick_params(colors="#64748b")
        for sp in ax_sb.spines.values(): sp.set_edgecolor("#e2e8f0")
        ax_sb.set_facecolor("white")
        fig_sb.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig_sb, use_container_width=True)
        plt.close()

        st.markdown("---")

        # ── Chart 2: Yearly sentiment trend (2023-2025) ───────────────────────
        st.markdown('<div class="section-header">📈 Yearly Sentiment Trend (2023–2025)</div>', unsafe_allow_html=True)

        tr1, tr2, tr3 = st.columns(3)
        with tr1:
            trend_proj_sent = st.selectbox("Project", ["All Projects"] + available_depts, key="sent_trend_proj")
        with tr2:
            trend_dept_sent = st.selectbox("Department", get_dept_options(all_phys), key="sent_trend_dept")
        with tr3:
            trend_div_sent  = st.selectbox("Division", get_div_options(all_phys, trend_dept_sent), key="sent_trend_div")

        # Apply all filters to trend data
        df_trend_sent = all_sent.copy()
        if trend_proj_sent != "All Projects":
            df_trend_sent = df_trend_sent[df_trend_sent["dept"] == trend_proj_sent]
        if trend_dept_sent != "All" or trend_div_sent != "All":
            phys_trend_filtered = apply_dept_div_filter(all_phys, trend_dept_sent, trend_div_sent)
            df_trend_sent = df_trend_sent[df_trend_sent["physician_id"].isin(phys_trend_filtered["physician_id"])]

        if "year" not in df_trend_sent.columns or df_trend_sent["year"].isna().all():
            st.warning("Year data not available in sentiment data.")
        else:
            yr_sent = (
                df_trend_sent.groupby(["year","sentiment"], as_index=False)
                .size()
                .rename(columns={"size":"count"})
            )
            yr_totals = yr_sent.groupby("year")["count"].transform("sum")
            yr_sent["pct"] = yr_sent["count"] / yr_totals * 100
            years_s = sorted(df_trend_sent["year"].dropna().unique().astype(int))

            neg_yr = yr_sent[yr_sent["sentiment"]=="NEGATIVE"].set_index("year")["pct"].reindex(years_s).fillna(0)
            pos_yr = yr_sent[yr_sent["sentiment"]=="POSITIVE"].set_index("year")["pct"].reindex(years_s).fillna(0)
            neu_yr = yr_sent[yr_sent["sentiment"]=="NEUTRAL"].set_index("year")["pct"].reindex(years_s).fillna(0)
            total_yr = yr_sent.groupby("year")["count"].sum().reindex(years_s).fillna(0)

            fig_yr, (ax_yr1, ax_yr2) = plt.subplots(1, 2, figsize=(12, 4.5))

            # Left: stacked bar by year — label all 3 segments
            w = 0.5
            x = list(range(len(years_s)))
            ax_yr1.bar(x, neg_yr.values, width=w, color="#e53e3e", alpha=0.88, label="Negative")
            ax_yr1.bar(x, neu_yr.values, width=w, bottom=neg_yr.values, color="#9ca3af", alpha=0.75, label="Neutral")
            ax_yr1.bar(x, pos_yr.values, width=w, bottom=(neg_yr+neu_yr).values, color="#38a169", alpha=0.88, label="Positive")
            for xi in x:
                nv = neg_yr.values[xi]
                uv = neu_yr.values[xi]
                pv = pos_yr.values[xi]
                if nv >= 3:
                    ax_yr1.text(xi, nv/2, f"{nv:.1f}%", ha="center", va="center",
                                fontsize=9, fontweight="700", color="white")
                if uv >= 5:
                    ax_yr1.text(xi, nv + uv/2, f"{uv:.1f}%", ha="center", va="center",
                                fontsize=9, fontweight="700", color="white")
                if pv >= 3:
                    ax_yr1.text(xi, nv + uv + pv/2, f"{pv:.1f}%", ha="center", va="center",
                                fontsize=9, fontweight="700", color="white")
            ax_yr1.set_xticks(x)
            ax_yr1.set_xticklabels([str(y) for y in years_s], fontsize=10)
            ax_yr1.set_ylabel("% of Comments", fontsize=10, color="#64748b")
            ax_yr1.set_title("Sentiment Mix by Year", fontsize=11, fontweight="bold", color="#1a365d")
            ax_yr1.legend(fontsize=9, framealpha=0.9)
            ax_yr1.set_ylim(0, 105)
            ax_yr1.grid(axis="y", alpha=0.2, linestyle="--", color="#bfdbfe")
            ax_yr1.tick_params(colors="#64748b")
            for sp in ax_yr1.spines.values(): sp.set_edgecolor("#e2e8f0")
            ax_yr1.set_facecolor("white")

            # Right: all 3 sentiment trend lines with clear labels
            ax_yr2.plot(years_s, neg_yr.values, color="#e53e3e", linewidth=2.5,
                        marker="o", markersize=8, label="Negative %")
            ax_yr2.plot(years_s, pos_yr.values, color="#38a169", linewidth=2.5,
                        marker="s", markersize=8, label="Positive %")
            ax_yr2.plot(years_s, neu_yr.values, color="#9ca3af", linewidth=2,
                        marker="^", markersize=7, label="Neutral %", linestyle="--")
            for yr_v, nv, pv, uv in zip(years_s, neg_yr.values, pos_yr.values, neu_yr.values):
                ax_yr2.annotate(f"{nv:.1f}%", (yr_v, nv), textcoords="offset points",
                                xytext=(0, 10), ha="center", fontsize=9, color="#e53e3e", fontweight="700")
                ax_yr2.annotate(f"{pv:.1f}%", (yr_v, pv), textcoords="offset points",
                                xytext=(0, 10), ha="center", fontsize=9, color="#38a169", fontweight="700")
                ax_yr2.annotate(f"{uv:.1f}%", (yr_v, uv), textcoords="offset points",
                                xytext=(0,-15), ha="center", fontsize=9, color="#64748b", fontweight="600")
            ax_yr2.set_xticks(years_s)
            ax_yr2.set_xticklabels([str(y) for y in years_s], fontsize=10)
            ax_yr2.set_ylabel("% of Comments", fontsize=10)
            ax_yr2.set_title("Negative vs Positive Trend", fontsize=11, fontweight="bold")
            ax_yr2.legend(fontsize=9)
            ax_yr2.grid(alpha=0.3, linestyle="--")
            ax_yr2.set_facecolor("white")

            fig_yr.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig_yr, use_container_width=True)
            plt.close()

            # Summary table
            st.markdown("**Year-by-Year Summary**")
            yr_table = pd.DataFrame({
                "Year":             [str(y) for y in years_s],
                "Total Comments":   total_yr.astype(int).values,
                "Negative %":       neg_yr.round(1).values,
                "Neutral %":        neu_yr.round(1).values,
                "Positive %":       pos_yr.round(1).values,
            })
            st.dataframe(yr_table, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRENDS (2023–2025)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📈 Year-on-Year Trends (2023–2025)</div>', unsafe_allow_html=True)

    # ── Filters row 1: Project + View mode ──────────────────────────────────────
    tf1, tf2 = st.columns([1, 2])
    with tf1:
        trend_dept = st.selectbox("Project", available_depts, key="trend_dept")
    raw_d, phys_d, _ = data[trend_dept]
    if raw_d is not None and "raters_group" in raw_d.columns:
        raw_d = raw_d[raw_d["raters_group"] != "Faculty Self-Evaluation"].copy()
    with tf2:
        view_mode = st.radio("View", ["Project Overall", "Individual Physician"],
                             horizontal=True, key="trend_mode")

    # ── Filters row 2: Department + Division ─────────────────────────────────
    tf3, tf4 = st.columns(2)
    proj_phys_t5 = all_phys[all_phys["department"] == trend_dept] if "department" in all_phys.columns else all_phys
    with tf3:
        t5_dept = st.selectbox("Department", get_dept_options(proj_phys_t5), key="t5_dept")
    with tf4:
        t5_div  = st.selectbox("Division",   get_div_options(proj_phys_t5, t5_dept), key="t5_div")

    # Apply dept/div filter to physician pool
    phys_pool_t5 = apply_dept_div_filter(proj_phys_t5, t5_dept, t5_div)["physician_id"].unique()
    if (t5_dept != "All" or t5_div != "All") and raw_d is not None:
        raw_d = raw_d[raw_d["physician_id"].isin(phys_pool_t5)].copy()

    if raw_d is None or raw_d.empty or "year" not in raw_d.columns or raw_d["year"].isna().all():
        st.warning("Year data not available. Check that your files include a Fillout Date column.")
    else:
        years_avail = sorted(raw_d["year"].dropna().unique().astype(int))

        all_phys_ids = sorted(raw_d["physician_id"].dropna().unique().tolist())
        if view_mode == "Individual Physician":
            selected_phys = st.selectbox("Physician ID", ["— Select —"] + all_phys_ids, key="trend_phys")
            if selected_phys == "— Select —":
                selected_phys = None
        else:
            selected_phys = None

        st.markdown("---")

        # ── DEPARTMENT OVERALL VIEW ───────────────────────────────────────────
        if view_mode == "Project Overall":

            trend_rows = []
            for yr in years_avail:
                df_yr   = raw_d[raw_d["year"] == yr]
                phys_yr = aggregate_physician(df_yr)
                phys_yr = phys_yr[phys_yr["n_forms"] >= min_forms].copy().reset_index(drop=True)
                phys_yr, _, _ = add_outlier_flags(phys_yr)
                trend_rows.append({
                    "Year":             yr,
                    "Physicians":       len(phys_yr),
                    "Avg Score":        round(phys_yr["avg_behavior_score"].mean(), 3),
                    "IQR Outliers":     int(phys_yr["low_iqr_outlier"].sum()) if "low_iqr_outlier" in phys_yr.columns else 0,
                    "% Flagged":        round(phys_yr["low_iqr_outlier"].mean()*100, 1) if "low_iqr_outlier" in phys_yr.columns else 0,
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
                        color="#2b7bc8", linewidth=2.5, markersize=8, label="Mean")
                ax.fill_between(
                    trend_df["Year"],
                    trend_df["Avg Score"] - trend_df["Score Std"],
                    trend_df["Avg Score"] + trend_df["Score Std"],
                    alpha=0.15, color="#2b7bc8", label="±1 SD"
                )
                ax.plot(trend_df["Year"], trend_df["Median Score"], "s--",
                        color="#6366f1", linewidth=1.5, markersize=6, label="Median")
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
                ax.set_facecolor("white")
                fig.patch.set_facecolor("white")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col_t2:
                st.markdown("**% Physicians Flagged by IQR**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                bar_cols = ["#38a169" if p < 10 else ("#f59e0b" if p < 20 else "#e53e3e")
                            for p in trend_df["% Flagged"]]
                bars = ax2.bar(trend_df["Year"], trend_df["% Flagged"],
                               color=bar_cols, edgecolor="white", linewidth=1.5, width=0.5)
                for bar, val in zip(bars, trend_df["% Flagged"]):
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.3, f"{val}%",
                             ha="center", va="bottom", fontsize=10, fontweight="700")
                ax2.set_xticks(years_avail)
                ax2.set_xlabel("Year", fontsize=10)
                ax2.set_ylabel("% Physicians Below IQR Fence", fontsize=10)
                ax2.set_title(f"{trend_dept} — IQR Flagged Rate Over Time", fontsize=11, fontweight="bold")
                ax2.grid(axis="y", alpha=0.3, linestyle="--")
                ax2.set_facecolor("white")
                fig2.patch.set_facecolor("white")
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                st.pyplot(fig2, use_container_width=True)
                plt.close()

            st.markdown("**Year-over-Year Summary Table**")
            st.dataframe(trend_df, use_container_width=True, hide_index=True)

            # ── Question-by-Question Analysis ────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header">📝 Question-by-Question Analysis</div>', unsafe_allow_html=True)
            q_cols_all = [c for c in raw_d.columns if c.startswith("q_")]
            if q_cols_all:
                q_rows = []
                for q in q_cols_all:
                    row = {"Question": q_display_label(q)}
                    for yr in years_avail:
                        yr_df_q = raw_d[raw_d["year"] == yr]
                        if "raters_group" in yr_df_q.columns:
                            yr_df_q = yr_df_q[yr_df_q["raters_group"] != "Faculty Self-Evaluation"]
                        vals = pd.to_numeric(yr_df_q[q], errors="coerce").dropna()
                        row[str(yr)] = round(vals.mean(), 3) if len(vals) > 0 else np.nan
                    yr_vals_q = [row[str(y)] for y in years_avail if not pd.isna(row.get(str(y), np.nan))]
                    row["Trend"] = (f"▲ {yr_vals_q[-1]-yr_vals_q[0]:+.3f}" if len(yr_vals_q)>=2 and yr_vals_q[-1]>yr_vals_q[0]
                                   else f"▼ {yr_vals_q[-1]-yr_vals_q[0]:+.3f}" if len(yr_vals_q)>=2 and yr_vals_q[-1]<yr_vals_q[0]
                                   else "—")
                    q_rows.append(row)
                q_df = pd.DataFrame(q_rows).sort_values(str(years_avail[-1]) if years_avail else "Question")

                # Heatmap-style bar chart per question
                fig_q, ax_q = plt.subplots(figsize=(10, max(4, len(q_cols_all)*0.45)))
                x_q = np.arange(len(q_rows))
                width_q = 0.8 / max(len(years_avail),1)
                colors_q = ["#2b7bc8","#6366f1","#38a169"]
                for i, yr in enumerate(years_avail):
                    vals_q = [r.get(str(yr), np.nan) for r in q_rows]
                    bars_q = ax_q.barh(
                        x_q - (len(years_avail)-1)*width_q/2 + i*width_q,
                        vals_q, height=width_q*0.85,
                        color=colors_q[i % len(colors_q)], alpha=0.85, label=str(yr)
                    )
                ax_q.set_yticks(x_q)
                ax_q.set_yticklabels([r["Question"] for r in q_rows], fontsize=9)
                ax_q.set_xlabel("Average Score (0–4)", fontsize=10, color="#64748b")
                ax_q.set_title(f"{trend_dept} — Average Score per Question by Year", fontsize=11, fontweight="bold", color="#1a365d")
                ax_q.axvline(3.0, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.6, label="Score 3.0")
                ax_q.set_xlim(0, 4.2)
                ax_q.legend(fontsize=9, loc="lower right")
                ax_q.grid(axis="x", alpha=0.25, linestyle="--")
                ax_q.set_facecolor("white"); fig_q.patch.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig_q, use_container_width=True)
                plt.close()

                st.markdown("**Question Averages by Year**")
                st.dataframe(q_df.reset_index(drop=True), use_container_width=True, hide_index=True,
                    column_config={str(yr): st.column_config.ProgressColumn(
                        str(yr), min_value=0, max_value=4, format="%.3f") for yr in years_avail})

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
                    # Flat mean across all question responses — consistent with aggregate_physician
                    flat_vals = yr_data[q_cols].values.flatten()
                    flat_vals = flat_vals[~pd.isnull(flat_vals)]
                    flat_mean = float(flat_vals.mean()) if len(flat_vals) > 0 else np.nan
                    phys_year_rows.append({
                        "Year":           yr,
                        "Forms":          len(yr_data),
                        "Avg Score":      round(flat_mean, 3),
                        "Min Score":      round(yr_data["overall_score"].min(), 3),
                        "Max Score":      round(yr_data["overall_score"].max(), 3),
                        "Score Std":      round(yr_data["overall_score"].std(), 3) if len(yr_data) > 1 else 0.0,
                    })
                phys_trend = pd.DataFrame(phys_year_rows)

                if phys_trend.empty:
                    st.warning("No year-level data available for this physician.")
                else:
                    # Compute dept avg per year for benchmarking — flat mean consistent with aggregate_physician
                    dept_avgs = {}
                    for yr in years_avail:
                        yr_df   = raw_d[raw_d["year"] == yr]
                        phys_yr_bench = aggregate_physician(yr_df)
                        phys_yr_bench = phys_yr_bench[phys_yr_bench["n_forms"] >= min_forms]
                        dept_avgs[yr] = phys_yr_bench["avg_behavior_score"].mean() if not phys_yr_bench.empty else np.nan

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
                                label=f"{trend_dept} Mean (Project)", alpha=0.8)

                        # Physician score line
                        ax.plot(phys_trend["Year"], phys_trend["Avg Score"], "o-",
                                color="#2b7bc8", linewidth=2.5, markersize=9,
                                label=f"Physician {selected_phys}", zorder=5)

                        # Min-max shading for spread
                        if "Min Score" in phys_trend.columns and "Max Score" in phys_trend.columns:
                            ax.fill_between(phys_trend["Year"],
                                            phys_trend["Min Score"], phys_trend["Max Score"],
                                            alpha=0.1, color="#2b7bc8", label="Score range")

                        # Label each data point
                        for _, row in phys_trend.iterrows():
                            ax.annotate(f"{row['Avg Score']:.2f}",
                                        (row["Year"], row["Avg Score"]),
                                        textcoords="offset points", xytext=(0, 10),
                                        ha="center", fontsize=9, fontweight="700",
                                        color="#1a365d")

                        ax.set_xticks(years_avail)
                        ax.set_xlabel("Year", fontsize=10)
                        ax.set_ylabel("Avg Behaviour Score (0–4)", fontsize=10)
                        ax.set_title(f"Physician {selected_phys} — Score Over Time",
                                     fontsize=11, fontweight="bold")
                        ax.legend(fontsize=9)
                        ax.grid(alpha=0.3, linestyle="--")
                        ax.set_facecolor("white")
                        fig.patch.set_facecolor("white")
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                    # Number of forms received per year
                    with col_p2:
                        st.markdown("**Evaluations Received Per Year**")
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        bar_colors = ["#2b7bc8"] * len(phys_trend)
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
                        ax2.set_facecolor("white")
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
                        phys_agg = phys_agg[phys_agg["n_forms"] >= min_forms].copy()
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
                                "Project Mean Score":  round(dept_avg_yr, 3),
                                "Physicians in Dept": len(phys_agg),
                            })
                    pct_df = pd.DataFrame(pct_rows)

                    if not pct_df.empty:
                        # Sort oldest to newest
                        pct_df = pct_df.sort_values("Year").reset_index(drop=True)
                        fig3, ax3 = plt.subplots(figsize=(max(5, len(pct_df)*2), 4.5))
                        colours_pct = ["#38a169" if p >= 50 else ("#f59e0b" if p >= 25 else "#e53e3e")
                                       for p in pct_df["Percentile Rank"]]
                        x_pos = range(len(pct_df))
                        bars3 = ax3.bar(x_pos, pct_df["Percentile Rank"],
                                        color=colours_pct, edgecolor="white",
                                        linewidth=1, width=0.55, alpha=0.88)
                        ax3.axhline(50, color="#64748b", linestyle="--",
                                    linewidth=1.2, label="50th percentile")
                        ax3.axhline(25, color="#e53e3e", linestyle=":",
                                    linewidth=1.2, label="25th percentile (concern)")
                        for bar, val in zip(bars3, pct_df["Percentile Rank"]):
                            ax3.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                                     f"{val:.0f}th",
                                     ha="center", va="bottom", fontsize=11, fontweight="700",
                                     color="#1a365d")
                        ax3.set_xticks(list(x_pos))
                        ax3.set_xticklabels(pct_df["Year"].astype(str), fontsize=11)
                        ax3.set_ylabel("Percentile Rank (100 = best)", fontsize=10, color="#64748b")
                        ax3.set_title(f"Physician {selected_phys} — Percentile Rank Over Time",
                                      fontsize=11, fontweight="bold", color="#1a365d")
                        ax3.set_ylim(0, 115)
                        ax3.legend(fontsize=9, framealpha=0.9)
                        ax3.grid(axis="y", alpha=0.25, linestyle="--", color="#bfdbfe")
                        ax3.tick_params(colors="#64748b")
                        for sp in ax3.spines.values(): sp.set_edgecolor("#e2e8f0")
                        ax3.set_facecolor("white")
                        fig3.patch.set_facecolor("white")
                        plt.tight_layout()
                        st.pyplot(fig3, use_container_width=True)
                        plt.close()

                        st.markdown("**Year-by-Year Summary for this Physician**")
                        merged_summary = phys_trend.merge(pct_df[["Year","Percentile Rank","Project Mean Score","Physicians in Dept"]],
                                                          on="Year", how="left")
                        st.dataframe(merged_summary, use_container_width=True, hide_index=True)

                    # ── Peer ranking table ────────────────────────────────────
                    st.markdown("---")
                    st.markdown("**Peer Ranking Table**")
                    st.caption("Sorted by Overall Avg ↓ · Trend = last year minus first year")

                    peer_rows = []
                    for pid in all_phys_ids:
                        pid_data = raw_d[raw_d["physician_id"] == pid]
                        if pid_data.empty:
                            continue
                        row = {"Physician ID": pid}
                        all_scores = []
                        for yr in years_avail:
                            yr_data_pid = pid_data[pid_data["year"] == yr]
                            if yr_data_pid.empty:
                                avg = np.nan
                            else:
                                q_cols_pid = [c for c in yr_data_pid.columns if c.startswith("q_")]
                                flat_pid = yr_data_pid[q_cols_pid].values.flatten()
                                flat_pid = flat_pid[~pd.isnull(flat_pid)]
                                avg = round(float(flat_pid.mean()), 3) if len(flat_pid) > 0 else np.nan
                            row[str(yr)] = avg
                            if not np.isnan(avg):
                                all_scores.append(avg)
                        row["Overall Avg"] = round(np.mean(all_scores), 3) if all_scores else np.nan
                        valid = [row[str(yr)] for yr in years_avail if not np.isnan(row.get(str(yr), np.nan))]
                        trend_val = round(valid[-1] - valid[0], 3) if len(valid) >= 2 else np.nan
                        row["Trend"] = f"▲ {trend_val:+.3f}" if pd.notna(trend_val) and trend_val > 0 else (
                                       f"▼ {trend_val:+.3f}" if pd.notna(trend_val) and trend_val < 0 else "— 0.000")
                        peer_rows.append(row)

                    peer_df = pd.DataFrame(peer_rows).sort_values("Overall Avg", ascending=False).reset_index(drop=True)
                    peer_df.insert(0, "Rank", range(1, len(peer_df)+1))

                    def highlight_selected(row):
                        if row["Physician ID"] == selected_phys:
                            return ["background-color: #dbeafe; font-weight: bold"] * len(row)
                        return [""] * len(row)

                    yr_fmt = {str(yr): lambda x: f"{x:.3f}" if pd.notna(x) else "—" for yr in years_avail}
                    styled = peer_df.style.apply(highlight_selected, axis=1).format(
                        {**yr_fmt, "Overall Avg": lambda x: f"{x:.3f}" if pd.notna(x) else "—"}
                    )
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    # ── Question-by-Question for this physician ───────────────
                    st.markdown("---")
                    st.markdown('<div class="section-header">📝 Question-by-Question Analysis</div>', unsafe_allow_html=True)
                    q_cols_p = [c for c in raw_d.columns if c.startswith("q_")]
                    if q_cols_p and selected_phys:
                        phys_q_rows = []
                        for q in q_cols_p:
                            row_q = {"Question": q_display_label(q)}
                            for yr in years_avail:
                                yr_phys_df = raw_d[(raw_d["physician_id"]==selected_phys) & (raw_d["year"]==yr)]
                                if "raters_group" in yr_phys_df.columns:
                                    yr_phys_df = yr_phys_df[yr_phys_df["raters_group"] != "Faculty Self-Evaluation"]
                                vals_p = pd.to_numeric(yr_phys_df[q], errors="coerce").dropna()
                                row_q[str(yr)] = round(vals_p.mean(), 3) if len(vals_p) > 0 else np.nan
                                # Department avg for this question this year
                                yr_all_df = raw_d[raw_d["year"]==yr]
                                if "raters_group" in yr_all_df.columns:
                                    yr_all_df = yr_all_df[yr_all_df["raters_group"] != "Faculty Self-Evaluation"]
                                dept_vals = pd.to_numeric(yr_all_df[q], errors="coerce").dropna()
                                row_q[f"Dept {yr}"] = round(dept_vals.mean(), 3) if len(dept_vals) > 0 else np.nan
                            yr_vals_p = [row_q[str(y)] for y in years_avail if not pd.isna(row_q.get(str(y), np.nan))]
                            row_q["Trend"] = (f"▲ {yr_vals_p[-1]-yr_vals_p[0]:+.3f}" if len(yr_vals_p)>=2 and yr_vals_p[-1]>yr_vals_p[0]
                                             else f"▼ {yr_vals_p[-1]-yr_vals_p[0]:+.3f}" if len(yr_vals_p)>=2 and yr_vals_p[-1]<yr_vals_p[0]
                                             else "—")
                            phys_q_rows.append(row_q)

                        # Grouped bar: physician vs dept per question per year
                        fig_pq, ax_pq = plt.subplots(figsize=(10, max(4, len(q_cols_p)*0.5)))
                        x_pq = np.arange(len(phys_q_rows))
                        bar_w = 0.35
                        colors_pq = {"phys": "#2b7bc8", "dept": "#e2e8f0"}
                        # Use last available year for comparison
                        last_yr = years_avail[-1]
                        phys_vals_pq = [r.get(str(last_yr), np.nan) for r in phys_q_rows]
                        dept_vals_pq = [r.get(f"Dept {last_yr}", np.nan) for r in phys_q_rows]
                        ax_pq.barh(x_pq + bar_w/2, phys_vals_pq, height=bar_w, color="#2b7bc8", alpha=0.9, label=f"{selected_phys} ({last_yr})")
                        ax_pq.barh(x_pq - bar_w/2, dept_vals_pq, height=bar_w, color="#94a3b8", alpha=0.7, label=f"Dept Avg ({last_yr})")
                        ax_pq.set_yticks(x_pq)
                        ax_pq.set_yticklabels([r["Question"] for r in phys_q_rows], fontsize=9)
                        ax_pq.set_xlabel("Average Score (0–4)", fontsize=10, color="#64748b")
                        ax_pq.set_title(f"{selected_phys} — Question Scores vs. Department Average ({last_yr})", fontsize=11, fontweight="bold", color="#1a365d")
                        ax_pq.set_xlim(0, 4.4)
                        ax_pq.axvline(3.0, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.6, label="Score 3.0")
                        ax_pq.legend(fontsize=9, loc="lower right")
                        ax_pq.grid(axis="x", alpha=0.25, linestyle="--")
                        ax_pq.set_facecolor("white"); fig_pq.patch.set_facecolor("white")
                        plt.tight_layout()
                        st.pyplot(fig_pq, use_container_width=True)
                        plt.close()

                        # Trend line chart per question (physician only)
                        if len(years_avail) >= 2:
                            fig_qt, ax_qt = plt.subplots(figsize=(10, max(4, len(q_cols_p)*0.4)))
                            cmap = plt.cm.get_cmap("tab10", len(phys_q_rows))
                            for i, row_q in enumerate(phys_q_rows):
                                yr_scores_q = [(yr, row_q[str(yr)]) for yr in years_avail if not pd.isna(row_q.get(str(yr), np.nan))]
                                if len(yr_scores_q) >= 2:
                                    xs, ys = zip(*yr_scores_q)
                                    ax_qt.plot(xs, ys, marker="o", linewidth=1.5, markersize=5,
                                               color=cmap(i), label=row_q["Question"], alpha=0.85)
                            ax_qt.set_xticks(years_avail)
                            ax_qt.set_xlabel("Year", fontsize=10)
                            ax_qt.set_ylabel("Avg Score (0–4)", fontsize=10)
                            ax_qt.set_title(f"{selected_phys} — Question Score Trends Over Time", fontsize=11, fontweight="bold", color="#1a365d")
                            ax_qt.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
                            ax_qt.set_ylim(0, 4.2)
                            ax_qt.grid(alpha=0.25, linestyle="--")
                            ax_qt.set_facecolor("white"); fig_qt.patch.set_facecolor("white")
                            plt.tight_layout()
                            st.pyplot(fig_qt, use_container_width=True)
                            plt.close()

                        st.markdown(f"**Question Scores — {selected_phys} vs. Department Avg**")
                        disp_cols = ["Question"] + [str(yr) for yr in years_avail] + ["Trend"]
                        q_disp_df = pd.DataFrame(phys_q_rows)[disp_cols]
                        st.dataframe(q_disp_df.reset_index(drop=True), use_container_width=True, hide_index=True,
                            column_config={str(yr): st.column_config.ProgressColumn(
                                str(yr), min_value=0, max_value=4, format="%.3f") for yr in years_avail})


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
    st.markdown('<div class="section-header">🏢 Departments & Divisions — Clinical Indicators</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def load_indicators(url, _version="v5.4"):
        if not url or url.startswith("REPLACE"):
            return None
        try:
            for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
                try:
                    df = pd.read_csv(url, encoding=enc); break
                except (UnicodeDecodeError, Exception):
                    continue
            else:
                st.warning("Could not decode indicators file."); return None
        except Exception as e:
            st.warning(f"Could not load indicators file: {e}"); return None
        df.columns = df.columns.str.strip()
        if "Division"    in df.columns: df["Division_norm"] = df["Division"].str.strip()
        if "FiscalCycle" in df.columns: df["FiscalCycle"]   = df["FiscalCycle"].astype(str).str.strip()
        for col in ["ClinicVisits", "ClinicWaitingTime", "PatientComplaints"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        # Derive true parent Department from Division using DIV_TO_DEPT mapping
        if "Division_norm" in df.columns:
            mapped = df["Division_norm"].map(DIV_TO_DEPT)
            if "Department" in df.columns:
                df["Department"] = mapped.fillna(df["Department"].str.strip())
            else:
                df["Department"] = mapped.fillna("Other")
        return df

    ind_df = load_indicators(GITHUB_URLS.get("indicators", ""), _version="v5.4")

    if ind_df is None:
        st.info("Indicators data not available. Add the indicators URL to GITHUB_URLS['indicators'].")
    else:
        # ── Filters ───────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            cycles = ["All"] + sorted(ind_df["FiscalCycle"].dropna().unique().tolist(), reverse=True)                      if "FiscalCycle" in ind_df.columns else ["All"]
            sel_cycle = st.selectbox("📅 Fiscal Cycle", cycles, key="ind_cycle")
        df_filt = ind_df if sel_cycle == "All" else ind_df[ind_df["FiscalCycle"] == sel_cycle]
        with fc2:
            dept_opts_t6 = ["All Departments"] + sorted(df_filt["Department"].dropna().unique().tolist())                            if "Department" in df_filt.columns else ["All Departments"]
            sel_dept_t6 = st.selectbox("🏥 Department", dept_opts_t6, key="ind_dept_filter")
        with fc3:
            df_for_div = df_filt if sel_dept_t6 == "All Departments" else df_filt[df_filt["Department"] == sel_dept_t6]
            div_opts_t6 = ["All Divisions"] + sorted(df_for_div["Division_norm"].dropna().unique().tolist())                           if "Division_norm" in df_for_div.columns else ["All Divisions"]
            sel_div_t6 = st.selectbox("🔬 Division", div_opts_t6, key="ind_div_filter")

        # Apply filters — df_view is used by ALL sections below
        df_view = df_filt.copy()
        if sel_dept_t6 != "All Departments" and "Department" in df_view.columns:
            df_view = df_view[df_view["Department"] == sel_dept_t6]
        if sel_div_t6 != "All Divisions" and "Division_norm" in df_view.columns:
            df_view = df_view[df_view["Division_norm"] == sel_div_t6]

        # ── Top KPIs ──────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        total_visits     = int(df_view["ClinicVisits"].sum())       if "ClinicVisits"      in df_view.columns else 0
        total_complaints = int(df_view["PatientComplaints"].sum())  if "PatientComplaints" in df_view.columns else 0
        avg_wait         = df_view["ClinicWaitingTime"].mean()      if "ClinicWaitingTime" in df_view.columns else np.nan
        n_depts          = df_view["Department"].nunique()           if "Department"        in df_view.columns else 0
        n_divs           = df_view["Division_norm"].nunique()        if "Division_norm"     in df_view.columns else 0

        k1,k3,k4,k5 = st.columns(4)
        def kpi(col, label, val, sub, cls=""):
            col.markdown(f'''<div class="metric-card {cls}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>''', unsafe_allow_html=True)
        kpi(k1, "Departments",     n_depts,          f"{n_divs} divisions",   "neutral")
        # Physicians card removed per request
        kpi(k3, "Clinic Visits",   f"{total_visits:,}", "total",              "success")
        kpi(k4, "Avg Wait Time",   f"{avg_wait:.1f} min" if pd.notna(avg_wait) else "—",
            "per visit", "success" if pd.isna(avg_wait) or avg_wait<20 else "warning" if avg_wait<40 else "danger")
        kpi(k5, "Complaints",      f"{total_complaints:,}", "total reported",
            "success" if total_complaints==0 else "warning" if total_complaints<20 else "danger")

        st.markdown("<br>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 1 — OVERVIEW (dept-level when All, physician-level when filtered)
        # ══════════════════════════════════════════════════════════════════════
        is_filtered = (sel_dept_t6 != "All Departments") or (sel_div_t6 != "All Divisions")
        section_title = "👤 Physicians in Selection" if is_filtered else "🏥 Department Overview"
        st.markdown(f'<div class="section-header">{section_title}</div>', unsafe_allow_html=True)

        if is_filtered:
            # ── Physician-level view when dept/div selected ───────────────────
            if "Physician Name" in df_view.columns and not df_view.empty:
                id_col = "Aubnetid" if "Aubnetid" in df_view.columns else df_view.columns[0]
                phys_agg_spec = {"Physicians_count": (id_col, "nunique")}
                if "ClinicVisits"      in df_view.columns: phys_agg_spec["Total_Visits"]     = ("ClinicVisits",     "sum")
                if "ClinicWaitingTime" in df_view.columns: phys_agg_spec["Avg_Wait"]          = ("ClinicWaitingTime","mean")
                if "PatientComplaints" in df_view.columns: phys_agg_spec["Total_Complaints"]  = ("PatientComplaints","sum")
                phys_sum = df_view.groupby("Physician Name", as_index=False).agg(**phys_agg_spec)
                for c in ["Total_Visits","Total_Complaints","Avg_Wait"]:
                    if c not in phys_sum.columns: phys_sum[c] = 0
                phys_sum["Total_Visits"]     = phys_sum["Total_Visits"].fillna(0).astype(int)
                phys_sum["Total_Complaints"] = phys_sum["Total_Complaints"].fillna(0).astype(int)
                phys_sum["Avg_Wait"]         = phys_sum["Avg_Wait"].round(1)
                phys_sum["Rate"]             = (phys_sum["Total_Complaints"] /
                                                phys_sum["Total_Visits"].replace(0,np.nan)*100).round(2).fillna(0)
                phys_sum = phys_sum.sort_values("Total_Visits", ascending=False).reset_index(drop=True)

                pc1, pc2 = st.columns(2)
                with pc1:
                    fig, ax = plt.subplots(figsize=(7, max(4, len(phys_sum)*0.45)))
                    max_v = phys_sum["Total_Visits"].max() or 1
                    colours = [plt.cm.Blues(0.35 + 0.6*(v/max_v)) for v in phys_sum["Total_Visits"]]
                    bars = ax.barh(phys_sum["Physician Name"], phys_sum["Total_Visits"],
                                   color=colours, edgecolor="white", linewidth=0.5, height=0.6)
                    for bar, val in zip(bars, phys_sum["Total_Visits"]):
                        ax.text(val + max_v*0.015, bar.get_y()+bar.get_height()/2,
                                f"{val:,}", va="center", fontsize=9, fontweight="700", color="#1a365d")
                    ax.set_xlabel("Total Clinic Visits", fontsize=10, color="#64748b")
                    ax.set_title(f"Visits by Physician — {sel_dept_t6}" + (f" / {sel_div_t6}" if sel_div_t6 != "All Divisions" else ""),
                                 fontsize=11, fontweight="800", color="#1a365d", pad=8)
                    ax.tick_params(colors="#64748b", labelsize=9)
                    for sp in ax.spines.values(): sp.set_edgecolor("#e2e8f0")
                    ax.grid(axis="x", alpha=0.25, linestyle="--", color="#bfdbfe")
                    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

                with pc2:
                    fig2, ax2 = plt.subplots(figsize=(7, max(4, len(phys_sum)*0.45)))
                    c2 = ["#e53e3e" if c>=3 else "#f59e0b" if c>=1 else "#38a169"
                          for c in phys_sum["Total_Complaints"]]
                    ax2.barh(phys_sum["Physician Name"], phys_sum["Total_Complaints"],
                             color=c2, edgecolor="white", linewidth=0.5, height=0.6, alpha=0.88)
                    for i, (val, wt) in enumerate(zip(phys_sum["Total_Complaints"], phys_sum["Avg_Wait"])):
                        ax2.text(val + 0.05, i, f"{val} complaints · {wt:.0f} min wait",
                                 va="center", fontsize=9, fontweight="600", color="#1a365d")
                    ax2.set_xlabel("Total Complaints", fontsize=10, color="#64748b")
                    ax2.set_title("Complaints & Wait Time by Physician",
                                  fontsize=11, fontweight="800", color="#1a365d", pad=8)
                    ax2.tick_params(colors="#64748b", labelsize=9)
                    for sp in ax2.spines.values(): sp.set_edgecolor("#e2e8f0")
                    ax2.grid(axis="x", alpha=0.25, linestyle="--", color="#bfdbfe")
                    ax2.set_facecolor("white"); fig2.patch.set_facecolor("white")
                    plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()

                # Physician summary table
                phys_display = phys_sum.rename(columns={
                    "Physician Name":"Physician","Total_Visits":"Visits",
                    "Avg_Wait":"Avg Wait (min)","Total_Complaints":"Complaints","Rate":"Complaint Rate %"
                }).drop(columns=["Physicians_count"], errors="ignore")
                st.dataframe(phys_display, use_container_width=True, hide_index=True,
                    column_config={
                        "Visits":      st.column_config.ProgressColumn(min_value=0, max_value=max(1,int(phys_display["Visits"].max())), format="%d"),
                        "Complaints":  st.column_config.ProgressColumn(min_value=0, max_value=max(1,int(phys_display["Complaints"].max())), format="%d"),
                    })
            else:
                st.info("No physician data available for this selection.")

        else:
            # ── Department-level view when no filter selected ─────────────────
            if "Department" in df_view.columns:
                id_col = "Aubnetid" if "Aubnetid" in df_view.columns else df_view.columns[0]
                dept_agg = {
                    "Physicians":        (id_col,           "nunique"),
                    "Divisions":         ("Division_norm",   "nunique") if "Division_norm" in df_view.columns else None,
                    "Total_Visits":      ("ClinicVisits",    "sum")     if "ClinicVisits"  in df_view.columns else None,
                    "Avg_Wait":          ("ClinicWaitingTime","mean")   if "ClinicWaitingTime" in df_view.columns else None,
                    "Total_Complaints":  ("PatientComplaints","sum")    if "PatientComplaints" in df_view.columns else None,
                }
                dept_agg = {k: v for k, v in dept_agg.items() if v is not None}
                dept_sum = df_view.groupby("Department", as_index=False).agg(**dept_agg)
                for c in ["Total_Visits","Total_Complaints","Divisions"]:
                    if c not in dept_sum.columns: dept_sum[c] = 0
                dept_sum["Total_Visits"]     = dept_sum["Total_Visits"].fillna(0).astype(int)
                dept_sum["Total_Complaints"] = dept_sum["Total_Complaints"].fillna(0).astype(int)
                dept_sum["Avg_Wait"]         = dept_sum["Avg_Wait"].round(1) if "Avg_Wait" in dept_sum.columns else 0
                dept_sum["Rate"]             = (dept_sum["Total_Complaints"] /
                                                dept_sum["Total_Visits"].replace(0, np.nan) * 100).round(2).fillna(0)
                dept_sum = dept_sum.sort_values("Total_Visits", ascending=False).reset_index(drop=True)

                dc1, dc2 = st.columns(2)
                with dc1:
                    fig, ax = plt.subplots(figsize=(7, max(3.5, len(dept_sum)*0.5)))
                    max_v = dept_sum["Total_Visits"].max() or 1
                    colours = [plt.cm.Blues(0.35 + 0.6*(v/max_v)) for v in dept_sum["Total_Visits"]]
                    bars = ax.barh(dept_sum["Department"], dept_sum["Total_Visits"],
                                   color=colours, edgecolor="white", linewidth=0.5, height=0.6)
                    for bar, val in zip(bars, dept_sum["Total_Visits"]):
                        ax.text(val + max_v*0.015, bar.get_y()+bar.get_height()/2,
                                f"{val:,}", va="center", fontsize=9, fontweight="700", color="#1a365d")
                    ax.set_xlabel("Total Clinic Visits", fontsize=10, color="#64748b")
                    ax.set_title("Visits by Department", fontsize=12, fontweight="800", color="#1a365d", pad=8)
                    ax.tick_params(colors="#64748b", labelsize=9)
                    for sp in ax.spines.values(): sp.set_edgecolor("#e2e8f0")
                    ax.grid(axis="x", alpha=0.25, linestyle="--", color="#bfdbfe")
                    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                with dc2:
                    fig2, ax2 = plt.subplots(figsize=(7, max(3.5, len(dept_sum)*0.5)))
                    c2 = ["#e53e3e" if r > dept_sum["Rate"].quantile(0.75) and r > 0
                          else "#f59e0b" if r > 0 else "#38a169"
                          for r in dept_sum["Rate"]]
                    bars2 = ax2.barh(dept_sum["Department"], dept_sum["Rate"],
                                     color=c2, edgecolor="white", linewidth=0.5, height=0.6, alpha=0.88)
                    for bar, val in zip(bars2, dept_sum["Rate"]):
                        ax2.text(val + 0.003, bar.get_y()+bar.get_height()/2,
                                 f"{val:.2f}%", va="center", fontsize=9, fontweight="700", color="#1a365d")
                    ax2.set_xlabel("Complaints per 100 Visits", fontsize=10, color="#64748b")
                    ax2.set_title("Complaint Rate by Department", fontsize=12, fontweight="800", color="#1a365d", pad=8)
                    ax2.tick_params(colors="#64748b", labelsize=9)
                    for sp in ax2.spines.values(): sp.set_edgecolor("#e2e8f0")
                    ax2.grid(axis="x", alpha=0.25, linestyle="--", color="#bfdbfe")
                    ax2.set_facecolor("white"); fig2.patch.set_facecolor("white")
                    ax2.legend(handles=[
                        mpatches.Patch(color="#38a169", alpha=0.88, label="No complaints"),
                        mpatches.Patch(color="#f59e0b", alpha=0.88, label="Low rate"),
                        mpatches.Patch(color="#e53e3e", alpha=0.88, label="High rate"),
                    ], fontsize=8, loc="lower right", framealpha=0.9)
                    plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()

                dept_display = dept_sum.rename(columns={
                    "Physicians":"Physicians","Divisions":"Divisions",
                    "Total_Visits":"Visits","Avg_Wait":"Avg Wait (min)",
                    "Total_Complaints":"Complaints","Rate":"Complaint Rate %"
                })
                st.dataframe(dept_display, use_container_width=True, hide_index=True,
                    column_config={
                        "Visits":      st.column_config.ProgressColumn(min_value=0, max_value=int(dept_display["Visits"].max()), format="%d"),
                        "Complaints":  st.column_config.ProgressColumn(min_value=0, max_value=max(1,int(dept_display["Complaints"].max())), format="%d"),
                    })

        st.markdown("<br>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 2 — DIVISION DRILL-DOWN
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">🔬 Division Drill-Down</div>', unsafe_allow_html=True)

        dd1, dd2 = st.columns([1.5, 1])
        with dd1:
            div_metric = st.selectbox("Metric",
                ["Clinic Visits", "Avg Wait Time (min)", "Patient Complaints"], key="div_metric")
        with dd2:
            top_n = st.slider("Top N", min_value=5, max_value=30, value=10, step=5, key="div_topn")

        # Uses df_view — already filtered by Dept/Div from top filters
        df_div = df_view

        if "Division_norm" in df_div.columns:
            id_col2 = "Aubnetid" if "Aubnetid" in df_div.columns else df_div.columns[0]
            div_agg_spec = {"Physicians": (id_col2, "nunique")}
            if "ClinicVisits"      in df_div.columns: div_agg_spec["Total_Visits"]    = ("ClinicVisits",     "sum")
            if "ClinicWaitingTime" in df_div.columns: div_agg_spec["Avg_Wait"]        = ("ClinicWaitingTime","mean")
            if "PatientComplaints" in df_div.columns: div_agg_spec["Total_Complaints"]= ("PatientComplaints","sum")

            div_sum = df_div.groupby("Division_norm", as_index=False).agg(**div_agg_spec)
            for c in ["Total_Visits","Total_Complaints","Avg_Wait"]:
                if c not in div_sum.columns: div_sum[c] = 0
            div_sum["Total_Visits"]     = div_sum["Total_Visits"].fillna(0).astype(int)
            div_sum["Total_Complaints"] = div_sum["Total_Complaints"].fillna(0).astype(int)
            div_sum["Avg_Wait"]         = div_sum["Avg_Wait"].round(1)
            div_sum["Rate"]             = (div_sum["Total_Complaints"] /
                                           div_sum["Total_Visits"].replace(0,np.nan)*100).round(2).fillna(0)

            metric_map = {"Clinic Visits":"Total_Visits","Avg Wait Time (min)":"Avg_Wait","Patient Complaints":"Total_Complaints"}
            sort_col_d = metric_map.get(div_metric, "Total_Visits")
            div_plot = div_sum.sort_values(sort_col_d, ascending=False).head(top_n).sort_values(sort_col_d, ascending=True)

            # Division bar chart
            bar_c = ["#e53e3e" if c>=3 else "#f59e0b" if c>=1 else "#2b7bc8"
                     for c in div_plot["Total_Complaints"]]
            fig3, ax3 = plt.subplots(figsize=(9, max(4, len(div_plot)*0.5)))
            vals = div_plot[sort_col_d]
            mx3 = vals.max() or 1
            ax3.barh(div_plot["Division_norm"], vals, color=bar_c,
                     edgecolor="white", linewidth=0.5, height=0.6, alpha=0.9)
            for val, cmp, bar in zip(vals, div_plot["Total_Complaints"],
                                     ax3.patches):
                fmt = f"{val:,}" if div_metric=="Clinic Visits" else (
                      f"{val:.1f}" if div_metric=="Avg Wait Time (min)" else str(int(val)))
                warn = f"  ⚠ {int(cmp)}" if cmp>0 and div_metric!="Patient Complaints" else ""
                ax3.text(val+mx3*0.01, bar.get_y()+bar.get_height()/2,
                         fmt+warn, va="center", fontsize=9, fontweight="700",
                         color="#e53e3e" if cmp>0 else "#1a365d")
            ax3.set_xlabel(div_metric, fontsize=10, color="#64748b")
            ax3.set_title(
                f"Top {min(top_n,len(div_plot))} Divisions — {div_metric}"
                + (f"  ·  {sel_dept_t6}" if sel_dept_t6 != "All Departments" else "")                + (f"  /  {sel_div_t6}"  if sel_div_t6  != "All Divisions"  else ""),
                fontsize=12, fontweight="800", color="#1a365d", pad=8)
            ax3.tick_params(colors="#64748b", labelsize=9)
            for sp in ax3.spines.values(): sp.set_edgecolor("#e2e8f0")
            ax3.grid(axis="x", alpha=0.25, linestyle="--", color="#bfdbfe")
            ax3.set_facecolor("white"); fig3.patch.set_facecolor("white")
            ax3.legend(handles=[
                mpatches.Patch(color="#2b7bc8",alpha=0.9,label="No complaints"),
                mpatches.Patch(color="#f59e0b",alpha=0.9,label="1–2 complaints"),
                mpatches.Patch(color="#e53e3e",alpha=0.9,label="3+ complaints"),
            ], fontsize=8, loc="lower right", framealpha=0.9)
            plt.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close()

            # Division detail table
            st.markdown("<br>", unsafe_allow_html=True)
            div_display = div_sum.sort_values("Total_Visits", ascending=False).rename(columns={
                "Division_norm":"Division","Total_Visits":"Visits",
                "Avg_Wait":"Avg Wait (min)","Total_Complaints":"Complaints","Rate":"Rate %"
            })
            st.dataframe(div_display, use_container_width=True, hide_index=True,
                column_config={
                    "Visits":     st.column_config.ProgressColumn(min_value=0, max_value=int(div_display["Visits"].max()) if "Visits" in div_display.columns else 100, format="%d"),
                    "Complaints": st.column_config.ProgressColumn(min_value=0, max_value=max(1,int(div_display["Complaints"].max())) if "Complaints" in div_display.columns else 1, format="%d"),
                })

        st.markdown("<br>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 3 — PHYSICIAN EXPLORER
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">👤 Physician Explorer</div>', unsafe_allow_html=True)

        pe0, pe1, pe2 = st.columns(3)
        with pe0:
            cycle_opts_pe = ["All Cycles"] + sorted(df_view["FiscalCycle"].dropna().unique().tolist(), reverse=True)                             if "FiscalCycle" in df_view.columns else ["All Cycles"]
            sel_cycle_pe = st.selectbox("Fiscal Cycle", cycle_opts_pe, key="pe_cycle")
        with pe1:
            df_pe_pool = df_view if sel_cycle_pe == "All Cycles" else df_view[df_view["FiscalCycle"] == sel_cycle_pe]
            phys_opts = ["All"] + sorted(df_pe_pool["Physician Name"].dropna().unique().tolist())                         if "Physician Name" in df_pe_pool.columns else ["All"]
            sel_phys_pe = st.selectbox("Physician", phys_opts, key="pe_phys")
        with pe2:
            sel_sort_pe = st.selectbox("Sort by",
                ["Clinic Visits ↓","Patient Complaints ↓","Waiting Time ↓"], key="pe_sort")
        # Apply cycle + physician filters on top of df_view (already filtered by Dept/Div)
        df_pe = df_pe_pool if sel_phys_pe == "All" else df_pe_pool[df_pe_pool["Physician Name"] == sel_phys_pe]

        sort_map = {"Clinic Visits ↓":"ClinicVisits","Patient Complaints ↓":"PatientComplaints","Waiting Time ↓":"ClinicWaitingTime"}
        sc = sort_map[sel_sort_pe]
        if sc in df_pe.columns: df_pe = df_pe.sort_values(sc, ascending=False)

        # Exclude Aubnetid from display
        show_c = ["Physician Name","Division_norm","Department","FiscalCycle",
                  "ClinicVisits","ClinicWaitingTime","PatientComplaints"]
        avail  = [c for c in show_c if c in df_pe.columns]
        shown  = df_pe[avail].rename(columns={
            "Physician Name":"Name",
            "Division_norm":"Division","FiscalCycle":"Cycle",
            "ClinicVisits":"Visits","ClinicWaitingTime":"Wait (min)","PatientComplaints":"Complaints",
        }).reset_index(drop=True)

        _mv = df_view["ClinicVisits"].max()      if "ClinicVisits"      in df_view.columns and not df_view.empty else None
        _mc = df_view["PatientComplaints"].max() if "PatientComplaints" in df_view.columns and not df_view.empty else None
        mv = int(_mv) if _mv is not None and pd.notna(_mv) else 100
        mc = int(_mc) if _mc is not None and pd.notna(_mc) else 10
        st.dataframe(shown, use_container_width=True, hide_index=True,
            column_config={
                "Visits":     st.column_config.ProgressColumn(min_value=0, max_value=mv, format="%d"),
                "Complaints": st.column_config.ProgressColumn(min_value=0, max_value=max(1,mc), format="%d"),
            })
        csv_out = shown.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export as CSV", csv_out, "physicians_indicators.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-header">🤖 Ask MC — Your AUBMC Data Assistant</div>', unsafe_allow_html=True)
    st.markdown("Ask MC anything about the physician performance data — risk flags, scores, sentiment, trends, and department comparisons.")

    # ── Build context summary from loaded data ────────────────────────────────
    @st.cache_data(show_spinner=False)
    def build_context(_all_phys, _data, _available_depts, _ind_df):  # v2
        lines = []
        lines.append("=" * 60)
        lines.append("AUBMC PHYSICIAN PERFORMANCE DASHBOARD — DATA CONTEXT")
        lines.append("=" * 60)
        lines.append("")

        # ── IMPORTANT TERMINOLOGY ─────────────────────────────────────────
        lines.append("IMPORTANT TERMINOLOGY:")
        lines.append("- 'Project groups' or 'Survey groups': AUBMC, ED, Pathology")
        lines.append("  These are the 3 groups used in the behavior survey project.")
        lines.append("  AUBMC = main hospital group, ED = Emergency Department group,")
        lines.append("  Pathology = Pathology & Lab group.")
        lines.append("  These are NOT clinical departments — they are project data groups.")
        lines.append("")
        lines.append("- 'Departments': The actual AUBMC clinical departments,")
        lines.append("  e.g. Internal Medicine, Surgery, Ob/Gyn, Pediatrics, etc.")
        lines.append("  These are available per physician via the lookup merge.")
        lines.append("  Each physician in the survey data HAS a Department and Division assigned.")
        lines.append("  You CAN answer questions like 'which department has the most priority physicians'.")
        lines.append("")
        lines.append("- 'Divisions': Sub-units within departments,")
        lines.append("  e.g. Cardiology (under Internal Medicine),")
        lines.append("  General Surgery (under Surgery), etc.")
        lines.append("")

        # ── BEHAVIOR SURVEY SUMMARY ───────────────────────────────────────
        lines.append("=" * 40)
        lines.append("BEHAVIOR SURVEY & PERFORMANCE DATA")
        lines.append("=" * 40)
        # Check if dept/div data is available
        has_dept = "Department" in _all_phys.columns and _all_phys["Department"].notna().any() and (_all_phys["Department"] != "").any()
        lines.append(f"Total physicians evaluated: {len(_all_phys)}")
        lines.append(f"Department/Division data available: {'YES — each physician has dept and division assigned' if has_dept else 'NO — lookup not loaded'}")
        lines.append(f"Years covered: 2023, 2024, 2025")
        lines.append(f"Overall avg behavior score: {_all_phys['avg_behavior_score'].mean():.3f} / 4.0")
        lines.append(f"Risk breakdown:")
        lines.append(f"  Priority (risk 3-4): {(_all_phys['risk_score']>=3).sum()} physicians")
        lines.append(f"  Monitor  (risk 1-2): {(_all_phys['risk_score'].between(1,2)).sum()} physicians")
        lines.append(f"  Clear    (risk 0):   {(_all_phys['risk_score']==0).sum()} physicians")
        lines.append(f"Negative sentiment flags: {_all_phys['negative_outlier'].sum()} physicians")
        lines.append("")

        # Build a lookup from all_phys for dept/division per physician
        dept_lookup  = _all_phys.set_index("physician_id")["Department"].to_dict() if "Department" in _all_phys.columns else {}
        div_lookup   = _all_phys.set_index("physician_id")["Division"].to_dict()   if "Division"   in _all_phys.columns else {}
        name_lookup  = _all_phys.set_index("physician_id")["FullName"].to_dict()   if "FullName"   in _all_phys.columns else {}

        for grp in _available_depts:
            _, phys, _ = _data[grp]
            if phys is None or phys.empty: continue
            lines.append(f"Survey group: {grp}")
            lines.append(f"  Physicians: {len(phys)}")
            lines.append(f"  Avg score: {phys['avg_behavior_score'].mean():.3f}")
            lines.append(f"  Priority: {(phys['risk_score']>=3).sum()}, Monitor: {phys['risk_score'].between(1,2).sum()}, Clear: {(phys['risk_score']==0).sum()}")
            lines.append(f"  Sentiment flags: {phys['negative_outlier'].sum()}")
            # Compute percentile rank within the group
            phys_pct = phys.copy()
            phys_pct["percentile"] = phys_pct["avg_behavior_score"].rank(pct=True).mul(100).round(1)
            # Build year-by-year scores per physician from raw data
            raw_grp, _, _ = _data[grp]
            yr_scores = {}  # pid -> {year: avg_score}
            if raw_grp is not None and "year" in raw_grp.columns:
                for yr in sorted(raw_grp["year"].dropna().unique().astype(int)):
                    yr_df = raw_grp[raw_grp["year"] == yr]
                    yr_agg = aggregate_physician(yr_df)
                    for _, row_yr in yr_agg.iterrows():
                        pid_yr = row_yr["physician_id"]
                        if pid_yr not in yr_scores:
                            yr_scores[pid_yr] = {}
                        yr_scores[pid_yr][yr] = round(row_yr["avg_behavior_score"], 3)

            lines.append(f"  All physicians (ID | Name | Dept | Div | 2023 | 2024 | 2025 | trend | score | percentile | risk | flags):")
            for _, r in phys_pct.sort_values('avg_behavior_score').iterrows():
                pid   = r['physician_id']
                iqr   = "IQR"  if r.get("low_iqr_outlier", False) else ""
                z     = "Z"    if r.get("low_z_outlier",   False) else ""
                b10   = "B10"  if r.get("low_bottom10",    False) else ""
                sent  = "SENT" if r.get("negative_outlier",False) else ""
                flags = " ".join(f for f in [iqr,z,b10,sent] if f) or "none"
                dept  = dept_lookup.get(pid, "Unknown")
                div   = div_lookup.get(pid, dept)
                name  = name_lookup.get(pid, "")
                pct   = r.get("percentile", 0)
                compound     = r.get("avg_compound", float("nan"))
                compound_str = f"{compound:.3f}" if pd.notna(compound) else "n/a"
                neg_ratio    = r.get("negative_ratio", float("nan"))
                neg_ratio_str= f"{neg_ratio:.1%}" if pd.notna(neg_ratio) else "n/a"
                # Year-by-year scores
                pid_yrs = yr_scores.get(pid, {})
                s2023 = f"{pid_yrs[2023]:.3f}" if 2023 in pid_yrs else "—"
                s2024 = f"{pid_yrs[2024]:.3f}" if 2024 in pid_yrs else "—"
                s2025 = f"{pid_yrs[2025]:.3f}" if 2025 in pid_yrs else "—"
                # Trend: last available minus first available
                yr_vals = [pid_yrs[y] for y in sorted(pid_yrs)]
                if len(yr_vals) >= 2:
                    trend_val = yr_vals[-1] - yr_vals[0]
                    trend_str = f"+{trend_val:.3f}" if trend_val > 0 else f"{trend_val:.3f}"
                else:
                    trend_str = "n/a"
                lines.append(f"    {pid} | {name} | {dept} | {div} | 2023={s2023} | 2024={s2024} | 2025={s2025} | trend={trend_str} | score={r['avg_behavior_score']:.3f} | percentile={pct:.0f}th | risk={int(r['risk_score'])} | sentiment={compound_str} | neg_ratio={neg_ratio_str} | flags=[{flags}]")
            lines.append("")

        # Also add a department-level risk summary for quick lookup
        if dept_lookup:
            lines.append("=" * 40)
            lines.append("RISK SUMMARY BY CLINICAL DEPARTMENT")
            lines.append("=" * 40)
            dept_risk = {}
            for _, r in _all_phys.iterrows():
                pid  = r["physician_id"]
                dept = dept_lookup.get(pid, "Unknown")
                if dept not in dept_risk:
                    dept_risk[dept] = {"Priority":0,"Monitor":0,"Clear":0,"Flagged_IDs":[]}
                risk = int(r.get("risk_score",0))
                if risk >= 3:
                    dept_risk[dept]["Priority"] += 1
                    dept_risk[dept]["Flagged_IDs"].append(pid)
                elif risk >= 1:
                    dept_risk[dept]["Monitor"] += 1
                else:
                    dept_risk[dept]["Clear"] += 1
            for dept, counts in sorted(dept_risk.items(), key=lambda x: str(x[0])):
                fids = ", ".join(counts["Flagged_IDs"]) if counts["Flagged_IDs"] else "none"
                lines.append(f"  {dept}: Priority={counts['Priority']}, Monitor={counts['Monitor']}, Clear={counts['Clear']} | Priority IDs: {fids}")
            lines.append("")

        # ── CLINICAL INDICATORS (DEPARTMENTS & DIVISIONS) ─────────────────
        if _ind_df is not None and not _ind_df.empty:
            lines.append("=" * 40)
            lines.append("CLINICAL DEPARTMENTS & DIVISIONS (INDICATORS DATA)")
            lines.append("=" * 40)
            if "Department" in _ind_df.columns:
                depts = _ind_df["Department"].dropna().unique()
                lines.append(f"Clinical departments ({len(depts)}): {', '.join(sorted(depts))}")
            if "Division_norm" in _ind_df.columns:
                divs = _ind_df["Division_norm"].dropna().unique()
                lines.append(f"Divisions ({len(divs)}): {', '.join(sorted(divs))}")
            # Dept-level aggregates
            if "Department" in _ind_df.columns and "ClinicVisits" in _ind_df.columns:
                dept_g = _ind_df.groupby("Department").agg(
                    Visits=("ClinicVisits","sum"),
                    Complaints=("PatientComplaints","sum") if "PatientComplaints" in _ind_df.columns else ("ClinicVisits","count"),
                    AvgWait=("ClinicWaitingTime","mean") if "ClinicWaitingTime" in _ind_df.columns else ("ClinicVisits","count"),
                ).reset_index().sort_values("Visits", ascending=False)
                lines.append("")
                lines.append("Department summary (visits / complaints / avg wait):")
                for _, r in dept_g.iterrows():
                    lines.append(f"  {r['Department']}: {int(r['Visits']):,} visits, {int(r.get('Complaints',0))} complaints, {r.get('AvgWait',0):.1f} min avg wait")
            lines.append("")

        return "\n".join(lines)

    # Load indicators for context (may be None if not configured)
    _ind_for_ctx = load_indicators(GITHUB_URLS.get("indicators", ""), _version="v5.4") if "load_indicators" in dir() else None
    context = build_context(all_phys, data, available_depts, _ind_for_ctx)

    # ── Chat UI ───────────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clear button at top right
    col_title, col_clear = st.columns([5, 1])
    with col_clear:
        if st.session_state.chat_history:
            if st.button("🗑️ Clear", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    # Chat container — display full history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input at bottom
    user_input = st.chat_input("Ask MC about physicians, scores, sentiment, flags...")

    if user_input:
        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build system prompt with live data context
        system_prompt = f"""You are MC, an AI data assistant for the AUBMC Physician Performance Dashboard.
You help medical administrators and stakeholders understand physician performance data.

CRITICAL — TERMINOLOGY YOU MUST FOLLOW:
- When someone says "department", they mean the CLINICAL departments: Internal Medicine, Surgery, Ob/Gyn, Pediatrics, etc.
- AUBMC, ED, and Pathology are NOT departments — they are PROJECT SURVEY GROUPS used in the behavior analysis.
- IMPORTANT: Each physician in the survey data HAS a clinical Department and Division assigned via a lookup merge.
  You CAN answer questions like "which department has the most priority physicians" or "show me Surgery physicians".
  The Department and Division fields appear in the physician data lines below.
- If someone asks about survey scores, outliers, or risk flags by department, look at the Department field in each physician line.
- If someone asks about clinic visits, wait times, or patient complaints — use the indicators data section.

Answer questions clearly, concisely, and accurately using ONLY the data provided below.
Be direct and professional. Do not invent numbers not present in the data.

DATA CONTEXT:
{context}
"""
        # Build message history for API (all turns)
        messages = []
        for h in st.session_state.chat_history:
            messages.append({"role": h["role"], "content": h["content"]})

        # Call Anthropic API and display response
        with st.chat_message("assistant"):
            with st.spinner("MC is thinking..."):
                try:
                    import requests
                    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
                    if not api_key:
                        answer = "⚠️ ANTHROPIC_API_KEY not found in Streamlit secrets."
                    else:
                        response = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={
                                "Content-Type": "application/json",
                                "x-api-key": api_key,
                                "anthropic-version": "2023-06-01",
                            },
                            json={
                                "model": "claude-haiku-4-5-20251001",
                                "max_tokens": 4096,
                                "system": system_prompt,
                                "messages": messages,
                            },
                            timeout=30
                        )
                        result = response.json()
                        if "content" in result and result["content"]:
                            answer = result["content"][0]["text"]
                        else:
                            err = result.get("error", {}).get("message", "Unknown error")
                            answer = f"Sorry, I couldn't get a response. ({err})"
                except Exception as e:
                    answer = f"Connection error: {str(e)}"

            st.markdown(answer)
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
