import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Page config
st.set_page_config(page_title="AUBMC Dashboard", page_icon="🏥", layout="wide")

# Load data
@st.cache_data
def load_data():
    # Load stats
    stats = pd.read_csv('Doctor_Statistics_2025 (1).csv')
    
    # Load indicators
    try:
        indicators = pd.read_csv('Physicians_Indicators_Anonymized.csv')
        indicators['Year'] = indicators['FiscalCycle'].str.extract(r'(\d{4})').astype(float)
    except:
        indicators = None
    
    # Load yearly behavior data
    yearly_data = []
    for year in range(2020, 2026):
        file = f'All_Departments_{year}.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['Year'] = year
            yearly_data.append(df)
    
    if yearly_data:
        long_data = pd.concat(yearly_data, ignore_index=True)
        behavior_by_year = long_data.groupby(['Subject ID', 'Year', 'Source'])['Response'].mean().reset_index()
        behavior_by_year.columns = ['Subject ID', 'Year', 'Source', 'Avg_Score']
    else:
        behavior_by_year = None
    
    return stats, indicators, behavior_by_year

stats, indicators, behavior_by_year = load_data()

# Main page
st.title("🏥 AUBMC Physician Performance Dashboard")

# Two buttons
col1, col2 = st.columns(2)
with col1:
    individual_btn = st.button("👤 Individual Physician Analysis", use_container_width=True, type="primary")
with col2:
    dept_btn = st.button("🏥 Departmental Analysis", use_container_width=True, type="primary")

# Default: General Overview
if not individual_btn and not dept_btn:
    st.header("📊 General Overview")
    
    # Top cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Physicians", len(stats))
    
    with col2:
        avg_score = stats['Avg_Score'].mean()
        st.metric("Average Score", f"{avg_score:.2f}/5.0")
    
    with col3:
        total_eval = stats['Num_Evaluations'].sum()
        st.metric("Total Evaluations", f"{total_eval:,}")
    
    with col4:
        if 'Sentiment_Negative' in stats.columns:
            neg_pct = (stats['Sentiment_Negative'].sum() / stats['Total_Comments'].sum() * 100)
            st.metric("Negative Comments", f"{neg_pct:.1f}%")
    
    st.markdown("---")
    
    # Visuals
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        fig = px.histogram(stats, x='Avg_Score', nbins=30, title="Physician Behavior Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Performers")
        top_10 = stats.nlargest(10, 'Avg_Score')[['Subject ID', 'Avg_Score']]
        st.dataframe(top_10, hide_index=True, use_container_width=True)
    
    # Summary stats
    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Score Metrics**")
        st.write(f"Mean: {stats['Avg_Score'].mean():.3f}")
        st.write(f"Median: {stats['Avg_Score'].median():.3f}")
        st.write(f"Std Dev: {stats['Avg_Score'].std():.3f}")
    
    with col2:
        st.write("**Evaluation Metrics**")
        st.write(f"Total: {stats['Num_Evaluations'].sum():,}")
        st.write(f"Avg per Physician: {stats['Num_Evaluations'].mean():.0f}")
        st.write(f"Max: {stats['Num_Evaluations'].max():,}")
    
    with col3:
        if 'Source' in stats.columns:
            st.write("**Department Breakdown**")
            dept_counts = stats['Source'].value_counts()
            for dept, count in dept_counts.items():
                st.write(f"{dept}: {count}")

# Individual Physician Analysis
elif individual_btn:
    st.header("👤 Individual Physician Analysis")
    
    # Filters
    with st.sidebar:
        st.subheader("🔍 Filters")
        
        # Physician filter
        physicians = sorted(stats['Subject ID'].unique())
        selected_phys = st.selectbox("Select Physician", physicians)
        
        # Year filter
        if behavior_by_year is not None:
            years = sorted(behavior_by_year['Year'].unique())
            selected_years = st.multiselect("Select Year(s)", years, default=years)
        
        # Source filter
        if 'Source' in stats.columns:
            sources = sorted(stats['Source'].unique())
            selected_source = st.multiselect("Select Source", sources, default=sources)
        
        # Question filter (if you have question-level data)
        st.info("Question-level filtering requires detailed data")
    
    # Get physician data
    phys_stats = stats[stats['Subject ID'] == selected_phys].iloc[0]
    
    # Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = phys_stats['Avg_Score']
        color = "inverse" if score >= 4.5 else "off"
        st.metric("Behavior Score", f"{score:.2f}/5.0", 
                 delta=f"{score - stats['Avg_Score'].mean():.2f} vs avg")
    
    with col2:
        st.metric("Evaluations", f"{phys_stats['Num_Evaluations']:.0f}")
    
    with col3:
        if 'Sentiment_Negative' in phys_stats:
            st.metric("Negative Comments", f"{phys_stats.get('Sentiment_Negative', 0):.0f}")
    
    with col4:
        percentile = (stats['Avg_Score'] < score).sum() / len(stats) * 100
        st.metric("Percentile", f"{percentile:.0f}%")
    
    st.markdown("---")
    
    # Score change by year
    if behavior_by_year is not None:
        st.subheader("📈 Average Score Change Over Years")
        
        phys_yearly = behavior_by_year[behavior_by_year['Subject ID'] == selected_phys]
        
        if len(phys_yearly) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=phys_yearly['Year'],
                y=phys_yearly['Avg_Score'],
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=12)
            ))
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Average Score",
                yaxis_range=[0, 5.5],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.dataframe(phys_yearly[['Year', 'Avg_Score']], hide_index=True)
    
    # Detailed metrics
    st.markdown("---")
    st.subheader("📊 Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Score Statistics**")
        st.write(f"Average: {phys_stats['Avg_Score']:.3f}")
        st.write(f"Std Dev: {phys_stats.get('Std_Dev', 0):.3f}")
        st.write(f"Min: {phys_stats.get('Min_Score', 0):.3f}")
        st.write(f"Max: {phys_stats.get('Max_Score', 0):.3f}")
    
    with col2:
        if 'Sentiment_Positive' in phys_stats:
            st.write("**Comment Sentiment**")
            st.write(f"Positive: {phys_stats.get('Sentiment_Positive', 0):.0f}")
            st.write(f"Neutral: {phys_stats.get('Sentiment_Neutral', 0):.0f}")
            st.write(f"Negative: {phys_stats.get('Sentiment_Negative', 0):.0f}")

# Departmental Analysis
elif dept_btn:
    st.header("🏥 Departmental Analysis")
    
    if indicators is None:
        st.error("❌ Clinical indicators data not found")
        st.stop()
    
    # Filters
    with st.sidebar:
        st.subheader("🔍 Filters")
        
        # Year/Cycle filter
        cycles = sorted(indicators['FiscalCycle'].unique())
        selected_cycles = st.multiselect("Select Cycle(s)", cycles, default=cycles)
        
        # Department filter
        departments = sorted(indicators['Department'].unique())
        selected_depts = st.multiselect("Select Department(s)", departments, default=departments)
    
    # Filter data
    filtered = indicators[
        (indicators['FiscalCycle'].isin(selected_cycles)) &
        (indicators['Department'].isin(selected_depts))
    ]
    
    # Cards
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_visits = filtered['ClinicVisits'].mean()
        st.metric("Avg Clinic Visits", f"{avg_visits:.0f}")
    
    with col2:
        avg_wait = filtered['ClinicWaitingTime'].mean()
        st.metric("Avg Waiting Time", f"{avg_wait:.1f} min")
    
    with col3:
        avg_complaints = filtered['PatientComplaints'].mean()
        st.metric("Avg Complaints", f"{avg_complaints:.1f}")
    
    with col4:
        physician_count = filtered['Aubnetid'].nunique()
        st.metric("Physicians", physician_count)
    
    st.markdown("---")
    
    # Department breakdown
    st.subheader("📈 By Department")
    
    dept_summary = filtered.groupby('Department').agg({
        'ClinicVisits': 'mean',
        'ClinicWaitingTime': 'mean',
        'PatientComplaints': 'mean',
        'Aubnetid': 'nunique'
    }).reset_index()
    
    dept_summary.columns = ['Department', 'Avg Visits', 'Avg Wait (min)', 'Avg Complaints', 'Physician Count']
    
    # Charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.bar(dept_summary, x='Department', y='Avg Visits', title="Average Clinic Visits")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(dept_summary, x='Department', y='Avg Wait (min)', title="Average Waiting Time")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.bar(dept_summary, x='Department', y='Avg Complaints', title="Average Complaints")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.markdown("---")
    st.subheader("📋 Department Summary Table")
    st.dataframe(dept_summary, hide_index=True, use_container_width=True)
    
    # Physician count by department
    st.markdown("---")
    st.subheader("👥 Physician Count by Department")
    
    fig = px.bar(dept_summary, x='Department', y='Physician Count', title="Number of Physicians per Department")
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("AUBMC Physician Performance Dashboard | For internal use only")
