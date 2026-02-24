import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="AUBMC Dashboard (Step 1)", layout="wide")

@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.map(str).str.strip()  # ✅ fixes your int column-name issue
    return df

def get_question_cols(df):
    return [c for c in df.columns if c.startswith("Q1_") or c.startswith("q_")]

def ensure_overall_score(df):
    df = df.copy()
    if "overall_score" not in df.columns:
        q_cols = get_question_cols(df)
        if len(q_cols) == 0:
            st.error("No overall_score and no Q1_/q_ columns found.")
            return df
        df["overall_score"] = df[q_cols].mean(axis=1, skipna=True)
    df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")
    return df

st.sidebar.header("Data Sources")
aubmc_path = st.sidebar.text_input("AUBMC CSV", "AUBMC_Behavior_survey.csv")
ed_path = st.sidebar.text_input("ED CSV", "ED_Behavior_survey.csv")
patho_path = st.sidebar.text_input("Pathology CSV", "Patho_Behavior_survey.csv")

# ---- Load ----
try:
    aubmc = ensure_overall_score(load_csv(aubmc_path))
    ed = ensure_overall_score(load_csv(ed_path))
    patho = ensure_overall_score(load_csv(patho_path))
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

# Add unit_group if missing
if "unit_group" not in aubmc.columns: aubmc["unit_group"] = "AUBMC_360"
if "unit_group" not in ed.columns: ed["unit_group"] = "ED"
if "unit_group" not in patho.columns: patho["unit_group"] = "PATHO"

behavior_all = pd.concat([aubmc, ed, patho], ignore_index=True)

# ---- Filters ----
st.sidebar.header("Filters")
unit_sel = st.sidebar.selectbox("Unit", sorted(behavior_all["unit_group"].unique()))
df = behavior_all[behavior_all["unit_group"] == unit_sel].copy()

# optional cycle filter if exists
if "cycle" in df.columns:
    df["cycle"] = df["cycle"].astype(str)
    cycles = sorted(df["cycle"].dropna().unique().tolist())
    cycle_sel = st.sidebar.multiselect("Cycle", cycles, default=cycles)
    df = df[df["cycle"].isin(cycle_sel)]

# optional rater group filter if exists
if "raters_group" in df.columns:
    rg = sorted(df["raters_group"].dropna().unique().tolist())
    default_rg = [x for x in rg if x != "Faculty Self-Evaluation"] or rg
    rg_sel = st.sidebar.multiselect("Raters group", rg, default=default_rg)
    df = df[df["raters_group"].isin(rg_sel)]

# ---- Main UI ----
st.title("AUBMC Behavior Dashboard — Step 1 (Load + Preview)")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Physicians", f"{df['physician_id'].nunique():,}" if "physician_id" in df.columns else "missing")
c3.metric("Avg overall score", f"{df['overall_score'].mean():.3f}" if "overall_score" in df.columns else "missing")

st.subheader("Preview")
st.dataframe(df.head(50), use_container_width=True)

with st.expander("Debug: Columns loaded"):
    st.write(df.columns.tolist())
    st.write("Question columns found:", len(get_question_cols(df)))
