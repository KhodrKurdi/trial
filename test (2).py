import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="AUBMC Physician Performance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .trend-up {
        color: #28a745;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .trend-down {
        color: #dc3545;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load all datasets."""
    
    # Load physician indicators
    try:
        indicators = pd.read_csv('Physicians_Indicators_Anonymized.csv')
        indicators['Year'] = indicators['FiscalCycle'].str.extract(r'(\d{4})-\d{4}').astype(float)
        indicators['ContractEffectiveDate'] = pd.to_datetime(indicators['ContractEffectiveDate'], errors='coerce')
        indicators['Years_of_Service'] = (datetime.now() - indicators['ContractEffectiveDate']).dt.days / 365.25
    except FileNotFoundError:
        indicators = None
    
    # Load behavior survey statistics
    try:
        doctor_stats = pd.read_csv('Doctor_Statistics_2025.csv')
    except FileNotFoundError:
        doctor_stats = None
    
    # Try to load multi-year behavior data - PARQUET OR CSV
    behavior_by_year = None
    has_multi_year = False
    
    # Try Parquet first (faster, smaller)
    if os.path.exists('All_Departments_Long_Numeric.parquet'):
        try:
            behavior_long = pd.read_parquet('All_Departments_Long_Numeric.parquet')
            if 'Fillout Date (mm/dd/yy)' in behavior_long.columns:
                behavior_long['Fillout Date (mm/dd/yy)'] = pd.to_datetime(
                    behavior_long['Fillout Date (mm/dd/yy)'], errors='coerce'
                )
                behavior_long['Year'] = behavior_long['Fillout Date (mm/dd/yy)'].dt.year
            
            behavior_by_year = behavior_long.groupby(['Subject ID', 'Year']).agg({
                'Response': 'mean'
            }).reset_index()
            behavior_by_year.columns = ['Subject ID', 'Year', 'Avg_Score']
            has_multi_year = True
        except Exception:
            pass
    
    # Fall back to CSV if Parquet not available
    if behavior_by_year is None:
        try:
            behavior_long = pd.read_csv('All_Departments_Long_Numeric.csv')
            if 'Fillout Date (mm/dd/yy)' in behavior_long.columns:
                behavior_long['Fillout Date (mm/dd/yy)'] = pd.to_datetime(
                    behavior_long['Fillout Date (mm/dd/yy)'], errors='coerce'
                )
                behavior_long['Year'] = behavior_long['Fillout Date (mm/dd/yy)'].dt.year
            
            behavior_by_year = behavior_long.groupby(['Subject ID', 'Year']).agg({
                'Response': 'mean'
            }).reset_index()
            behavior_by_year.columns = ['Subject ID', 'Year', 'Avg_Score']
            has_multi_year = True
        except FileNotFoundError:
            if doctor_stats is not None:
                behavior_by_year = doctor_stats.copy()
                behavior_by_year['Year'] = 2025
                has_multi_year = False
            else:
                behavior_by_year = None
                has_multi_year = False
    
    return indicators, doctor_stats, behavior_by_year, has_multi_year

# Load data
with st.spinner('Loading data...'):
    indicators, doctor_stats, behavior_by_year, has_multi_year = load_data()

if doctor_stats is None and behavior_by_year is None:
    st.error("❌ No behavior survey data found. Please run your analysis notebook first.")
    st.stop()

# Main header
st.markdown('<div class="main-header">🏥 AUBMC Physician Performance Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Dashboard Mode")
mode = st.sidebar.radio(
    "Select View:",
    ["🏠 Overview", "👤 Individual Physician", "🏥 Departmental Analysis"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# ===========================================================================
# OVERVIEW MODE
# ===========================================================================
if mode == "🏠 Overview":
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_phys = len(doctor_stats) if doctor_stats is not None else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Physicians</div>
            <div class="metric-value">{total_phys}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if doctor_stats is not None:
            avg_score = doctor_stats['Avg_Score'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average Behavior Score</div>
                <div class="metric-value">{avg_score:.2f}/5.0</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if doctor_stats is not None:
            total_eval = doctor_stats['Num_Evaluations'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Evaluations</div>
                <div class="metric-value">{total_eval:,}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if indicators is not None and 'Department' in indicators.columns:
            num_depts = indicators['Department'].nunique()
        elif doctor_stats is not None and 'Source' in doctor_stats.columns:
            num_depts = doctor_stats['Source'].nunique()
        else:
            num_depts = 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Departments</div>
            <div class="metric-value">{num_depts}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    if doctor_stats is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Score Distribution")
            fig = px.histogram(
                doctor_stats, 
                x='Avg_Score',
                nbins=30,
                title="Distribution of Physician Behavior Scores",
                color_discrete_sequence=['#667eea']
            )
            fig.add_vline(x=avg_score, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💬 Comment Sentiment")
            if 'Sentiment_Negative' in doctor_stats.columns:
                sentiment_data = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Count': [
                        doctor_stats.get('Sentiment_Positive', pd.Series([0])).sum(),
                        doctor_stats.get('Sentiment_Neutral', pd.Series([0])).sum(),
                        doctor_stats.get('Sentiment_Negative', pd.Series([0])).sum()
                    ]
                })
                fig = px.pie(
                    sentiment_data, 
                    values='Count', 
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Top and bottom performers
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌟 Top 10 Performers")
            top_10 = doctor_stats.nlargest(10, 'Avg_Score')[['Subject ID', 'Avg_Score', 'Num_Evaluations']]
            top_10['Rank'] = range(1, 11)
            st.dataframe(top_10[['Rank', 'Subject ID', 'Avg_Score', 'Num_Evaluations']], hide_index=True)
        
        with col2:
            st.subheader("⚠️ Bottom 10 Performers")
            bottom_10 = doctor_stats.nsmallest(10, 'Avg_Score')[['Subject ID', 'Avg_Score', 'Num_Evaluations']]
            bottom_10['Rank'] = range(1, 11)
            st.dataframe(bottom_10[['Rank', 'Subject ID', 'Avg_Score', 'Num_Evaluations']], hide_index=True)

# ===========================================================================
# INDIVIDUAL PHYSICIAN MODE
# ===========================================================================
elif mode == "👤 Individual Physician":
    st.header("👤 Individual Physician Report")
    
    # Get physician list
    if behavior_by_year is not None:
        physicians_list = sorted(behavior_by_year['Subject ID'].unique())
    elif doctor_stats is not None:
        physicians_list = sorted(doctor_stats['Subject ID'].unique())
    else:
        st.error("No physician data available")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    selected_physician = st.sidebar.selectbox("Select Physician", physicians_list)
    
    # Get data for selected physician
    if behavior_by_year is not None:
        phys_behavior_all = behavior_by_year[behavior_by_year['Subject ID'] == selected_physician]
        latest_year = phys_behavior_all['Year'].max()
        phys_behavior_latest = phys_behavior_all[phys_behavior_all['Year'] == latest_year].iloc[0]
    else:
        phys_behavior_all = None
        phys_behavior_latest = None
    
    if doctor_stats is not None:
        phys_stats = doctor_stats[doctor_stats['Subject ID'] == selected_physician]
        if len(phys_stats) > 0:
            phys_stats = phys_stats.iloc[0]
        else:
            phys_stats = None
    else:
        phys_stats = None
    
    if indicators is not None:
        phys_indicators = indicators[indicators['Aubnetid'] == selected_physician]
    else:
        phys_indicators = pd.DataFrame()
    
    # Calculate YoY change
    yoy_change = 0
    yoy_pct_change = 0
    has_trend = False
    if phys_behavior_all is not None and len(phys_behavior_all) > 1:
        years_sorted = phys_behavior_all.sort_values('Year')
        previous_year = years_sorted.iloc[-2]
        current_year = years_sorted.iloc[-1]
        yoy_change = current_year['Avg_Score'] - previous_year['Avg_Score']
        yoy_pct_change = (yoy_change / previous_year['Avg_Score']) * 100
        has_trend = True
    
    # Overview cards
    st.subheader("📋 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if phys_behavior_latest is not None:
            score = phys_behavior_latest['Avg_Score']
        elif phys_stats is not None:
            score = phys_stats['Avg_Score']
        else:
            score = 0
        
        color = "#28a745" if score >= 4.5 else "#ffc107" if score >= 4.0 else "#dc3545"
        
        trend_html = ""
        if has_trend:
            if yoy_change > 0:
                trend_html = f'<div class="trend-up">↑ {yoy_change:.2f} (+{yoy_pct_change:.1f}%)</div>'
            elif yoy_change < 0:
                trend_html = f'<div class="trend-down">↓ {yoy_change:.2f} ({yoy_pct_change:.1f}%)</div>'
        
        st.markdown(f"""
        <div class="metric-card" style="background: {color};">
            <div class="metric-label">Behavior Score</div>
            <div class="metric-value">{score:.2f}</div>
            {trend_html}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if len(phys_indicators) > 0:
            avg_visits = phys_indicators['ClinicVisits'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Clinic Visits</div>
                <div class="metric-value">{avg_visits:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: #6c757d;">
                <div class="metric-label">Avg Clinic Visits</div>
                <div class="metric-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if len(phys_indicators) > 0:
            avg_wait = phys_indicators['ClinicWaitingTime'].mean()
            wait_color = "#28a745" if avg_wait < 30 else "#ffc107" if avg_wait < 45 else "#dc3545"
            st.markdown(f"""
            <div class="metric-card" style="background: {wait_color};">
                <div class="metric-label">Avg Wait Time</div>
                <div class="metric-value">{avg_wait:.1f}</div>
                <div class="metric-label">minutes</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: #6c757d;">
                <div class="metric-label">Avg Wait Time</div>
                <div class="metric-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if phys_stats is not None and 'Negative_Comment_Count' in phys_stats:
            neg_count = phys_stats['Negative_Comment_Count']
            neg_color = "#28a745" if neg_count == 0 else "#ffc107" if neg_count < 3 else "#dc3545"
            st.markdown(f"""
            <div class="metric-card" style="background: {neg_color};">
                <div class="metric-label">Negative Comments</div>
                <div class="metric-value">{neg_count:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: #6c757d;">
                <div class="metric-label">Negative Comments</div>
                <div class="metric-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Score Trends", 
        "📊 Performance Metrics", 
        "💬 Comments", 
        "🏥 Clinical Metrics",
        "🎯 Summary"
    ])
    
    # TAB 1: Score Trends
    with tab1:
        st.subheader("📈 Behavior Score Trend Over Time")
        
        if phys_behavior_all is not None and len(phys_behavior_all) > 1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=phys_behavior_all['Year'],
                y=phys_behavior_all['Avg_Score'],
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                marker=dict(size=12),
                text=[f"{score:.2f}" for score in phys_behavior_all['Avg_Score']],
                textposition='top center'
            ))
            
            fig.add_hline(y=4.0, line_dash="dash", line_color="orange", 
                         annotation_text="Target: 4.0")
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Average Score",
                yaxis_range=[0, 5.5],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Only one year of data available")
    
    # TAB 2: Performance Metrics
    with tab2:
        if phys_stats is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Behavior Score Analysis")
                st.metric("Average Score", f"{phys_stats['Avg_Score']:.3f}")
                st.metric("Standard Deviation", f"{phys_stats.get('Std_Dev', 0):.3f}")
                st.metric("Min Score", f"{phys_stats.get('Min_Score', 0):.3f}")
                st.metric("Max Score", f"{phys_stats.get('Max_Score', 0):.3f}")
                
                # Comparison
                if doctor_stats is not None:
                    overall_avg = doctor_stats['Avg_Score'].mean()
                    diff = phys_stats['Avg_Score'] - overall_avg
                    if diff > 0:
                        st.success(f"✅ {diff:.3f} points above average")
                    elif diff < 0:
                        st.warning(f"⚠️ {abs(diff):.3f} points below average")
            
            with col2:
                st.markdown("### Percentile Ranking")
                if doctor_stats is not None:
                    percentile = (doctor_stats['Avg_Score'] < phys_stats['Avg_Score']).sum() / len(doctor_stats) * 100
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=percentile,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Percentile Rank"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 75], 'color': "gray"},
                                {'range': [75, 100], 'color': "lightgreen"}
                            ],
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"Better than {percentile:.1f}% of physicians")
    
    # TAB 3: Comments
    with tab3:
        if phys_stats is not None and 'Sentiment_Positive' in phys_stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pos_count = phys_stats.get('Sentiment_Positive', 0)
                if pd.isna(pos_count):
                    pos_count = 0
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Positive</h3>
                    <h1>{pos_count:.0f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                neu_count = phys_stats.get('Sentiment_Neutral', 0)
                if pd.isna(neu_count):
                    neu_count = 0
                st.markdown(f"""
                <div class="info-box">
                    <h3>➖ Neutral</h3>
                    <h1>{neu_count:.0f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                neg_count = phys_stats.get('Sentiment_Negative', 0)
                if pd.isna(neg_count):
                    neg_count = 0
                st.markdown(f"""
                <div class="warning-box">
                    <h3>❌ Negative</h3>
                    <h1>{neg_count:.0f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Pie chart
            total = pos_count + neu_count + neg_count
            if total > 0:
                sentiment_data = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Count': [pos_count, neu_count, neg_count]
                })
                fig = px.pie(
                    sentiment_data,
                    values='Count',
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Sentiment data not available")
    
    # TAB 4: Clinical Metrics
    with tab4:
        if len(phys_indicators) > 1:
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Clinic Visits', 'Waiting Time (min)', 'Complaints')
            )
            
            fig.add_trace(
                go.Scatter(x=phys_indicators['Year'], y=phys_indicators['ClinicVisits'], 
                          mode='lines+markers', line=dict(color='#28a745', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=phys_indicators['Year'], y=phys_indicators['ClinicWaitingTime'],
                          mode='lines+markers', line=dict(color='#ffc107', width=2)),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=phys_indicators['Year'], y=phys_indicators['PatientComplaints'],
                          mode='lines+markers', line=dict(color='#dc3545', width=2)),
                row=1, col=3
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.markdown("### Historical Data")
            display_df = phys_indicators[['FiscalCycle', 'ClinicVisits', 'ClinicWaitingTime', 'PatientComplaints']]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        elif len(phys_indicators) == 1:
            st.info("ℹ️ Only one period available")
            latest = phys_indicators.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Clinic Visits", f"{latest['ClinicVisits']:,.0f}")
            with col2:
                st.metric("Waiting Time", f"{latest['ClinicWaitingTime']:.1f} min")
            with col3:
                st.metric("Complaints", f"{latest['PatientComplaints']:.0f}")
        else:
            st.info("ℹ️ No clinical data available")
    
    # TAB 5: Summary
    with tab5:
        st.subheader("🎯 Performance Summary")
        
        # Overall assessment
        if phys_stats is not None:
            score = phys_stats['Avg_Score']
            
            if score >= 4.5:
                st.success("✅ **Excellent Performance** - Exceeds expectations")
            elif score >= 4.0:
                st.success("✅ **Good Performance** - Meets expectations")
            elif score >= 3.5:
                st.warning("⚠️ **Fair Performance** - Room for improvement")
            else:
                st.error("❌ **Needs Attention** - Below expectations")
        
        # Key metrics summary
        st.markdown("### Key Metrics")
        summary_data = []
        
        if phys_stats is not None:
            summary_data.append({"Metric": "Behavior Score", "Value": f"{phys_stats['Avg_Score']:.2f}/5.0"})
            summary_data.append({"Metric": "Evaluations", "Value": f"{phys_stats.get('Num_Evaluations', 0):,.0f}"})
            summary_data.append({"Metric": "Negative Comments", "Value": f"{phys_stats.get('Negative_Comment_Count', 0):.0f}"})
        
        if len(phys_indicators) > 0:
            summary_data.append({"Metric": "Avg Clinic Visits", "Value": f"{phys_indicators['ClinicVisits'].mean():.0f}"})
            summary_data.append({"Metric": "Avg Wait Time", "Value": f"{phys_indicators['ClinicWaitingTime'].mean():.1f} min"})
            summary_data.append({"Metric": "Avg Complaints", "Value": f"{phys_indicators['PatientComplaints'].mean():.1f}"})
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ===========================================================================
# DEPARTMENTAL ANALYSIS MODE
# ===========================================================================
elif mode == "🏥 Departmental Analysis":
    st.header("🏥 Departmental Performance Analysis")
    
    # Determine department column
    if indicators is not None and 'Department' in indicators.columns:
        dept_col = 'Department'
        departments = sorted(indicators[dept_col].dropna().unique())
        use_source = 'indicators'
    elif doctor_stats is not None and 'Source' in doctor_stats.columns:
        dept_col = 'Source'
        departments = sorted(doctor_stats[dept_col].dropna().unique())
        use_source = 'stats'
    else:
        st.error("No department information available")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    selected_departments = st.sidebar.multiselect(
        "Select Department(s)",
        departments,
        default=departments
    )
    
    if len(selected_departments) == 0:
        st.warning("⚠️ Please select at least one department")
        st.stop()
    
    # Filter data
    if use_source == 'stats' and doctor_stats is not None:
        dept_data_stats = doctor_stats[doctor_stats[dept_col].isin(selected_departments)]
    else:
        dept_data_stats = None
    
    if indicators is not None:
        dept_data_indicators = indicators[indicators[dept_col].isin(selected_departments)]
    else:
        dept_data_indicators = None
    
    # Overview
    st.subheader("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Departments", len(selected_departments))
    
    with col2:
        if dept_data_stats is not None:
            st.metric("Physicians", f"{len(dept_data_stats):,}")
        elif dept_data_indicators is not None:
            st.metric("Physicians", f"{dept_data_indicators['Aubnetid'].nunique():,}")
    
    with col3:
        if dept_data_indicators is not None:
            st.metric("Avg Visits", f"{dept_data_indicators['ClinicVisits'].mean():.0f}")
    
    with col4:
        if dept_data_indicators is not None:
            st.metric("Avg Wait Time", f"{dept_data_indicators['ClinicWaitingTime'].mean():.1f} min")
    
    st.markdown("---")
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Clinic Metrics",
        "⭐ Behavior Scores", 
        "📈 Rankings",
        "🎯 Insights"
    ])
    
    # TAB 1: Clinic Metrics
    with tab1:
        if dept_data_indicators is not None:
            # Aggregate
            dept_summary = dept_data_indicators.groupby(dept_col).agg({
                'ClinicVisits': ['mean', 'sum'],
                'ClinicWaitingTime': 'mean',
                'PatientComplaints': ['mean', 'sum'],
                'Aubnetid': 'nunique'
            }).round(2)
            
            dept_summary.columns = ['_'.join(col) for col in dept_summary.columns]
            dept_summary = dept_summary.rename(columns={
                'ClinicVisits_mean': 'Avg_Visits',
                'ClinicVisits_sum': 'Total_Visits',
                'ClinicWaitingTime_mean': 'Avg_Wait',
                'PatientComplaints_mean': 'Avg_Complaints',
                'PatientComplaints_sum': 'Total_Complaints',
                'Aubnetid_nunique': 'Num_Physicians'
            })
            dept_summary = dept_summary.reset_index()
            
            # Charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = px.bar(
                    dept_summary.sort_values('Avg_Visits', ascending=False),
                    x=dept_col, y='Avg_Visits',
                    title="Avg Clinic Visits",
                    color='Avg_Visits',
                    color_continuous_scale='Blues'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    dept_summary.sort_values('Avg_Wait', ascending=True),
                    x=dept_col, y='Avg_Wait',
                    title="Avg Waiting Time",
                    color='Avg_Wait',
                    color_continuous_scale='Reds_r'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = px.bar(
                    dept_summary.sort_values('Avg_Complaints', ascending=False),
                    x=dept_col, y='Avg_Complaints',
                    title="Avg Complaints",
                    color='Avg_Complaints',
                    color_continuous_scale='Oranges'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.markdown("### Detailed Statistics")
            st.dataframe(dept_summary, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No clinic metrics available")
    
    # TAB 2: Behavior Scores
    with tab2:
        if dept_data_stats is not None:
            # Aggregate
            behavior_summary = dept_data_stats.groupby(dept_col).agg({
                'Avg_Score': ['mean', 'std', 'count']
            }).round(3)
            
            behavior_summary.columns = ['_'.join(col) for col in behavior_summary.columns]
            behavior_summary = behavior_summary.rename(columns={
                'Avg_Score_mean': 'Avg_Score',
                'Avg_Score_std': 'StdDev',
                'Avg_Score_count': 'Num_Physicians'
            })
            behavior_summary = behavior_summary.reset_index()
            
            # Bar chart
            fig = px.bar(
                behavior_summary.sort_values('Avg_Score', ascending=False),
                x=dept_col, y='Avg_Score',
                title="Average Behavior Score by Department",
                color='Avg_Score',
                color_continuous_scale='RdYlGn',
                text='Avg_Score'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_xaxis(tickangle=45)
            fig.update_layout(yaxis_range=[0, 5.5])
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot
            fig = px.box(
                dept_data_stats,
                x=dept_col, y='Avg_Score',
                title="Score Distribution by Department",
                color=dept_col
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.dataframe(behavior_summary, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No behavior score data available")
    
    # TAB 3: Rankings
    with tab3:
        if dept_data_stats is not None:
            behavior_summary = dept_data_stats.groupby(dept_col).agg({
                'Avg_Score': 'mean'
            }).round(3).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Top Performers")
                top = behavior_summary.nlargest(10, 'Avg_Score')
                top['Rank'] = range(1, len(top) + 1)
                st.dataframe(top[['Rank', dept_col, 'Avg_Score']], hide_index=True)
            
            with col2:
                st.markdown("### ⚠️ Needs Attention")
                bottom = behavior_summary.nsmallest(10, 'Avg_Score')
                bottom['Rank'] = range(1, len(bottom) + 1)
                st.dataframe(bottom[['Rank', dept_col, 'Avg_Score']], hide_index=True)
    
    # TAB 4: Insights
    with tab4:
        st.subheader("🎯 Key Insights")
        
        # Best and worst
        if dept_data_stats is not None:
            behavior_summary = dept_data_stats.groupby(dept_col).agg({
                'Avg_Score': 'mean'
            }).round(3).reset_index()
            behavior_summary = behavior_summary.sort_values('Avg_Score', ascending=False)
            
            if len(behavior_summary) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    best = behavior_summary.iloc[0]
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>🌟 Best: {best[dept_col]}</h3>
                        <p><strong>Score:</strong> {best['Avg_Score']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(behavior_summary) > 1:
                    with col2:
                        worst = behavior_summary.iloc[-1]
                        st.markdown(f"""
                        <div class="warning-box">
                            <h3>⚠️ Needs Attention: {worst[dept_col]}</h3>
                            <p><strong>Score:</strong> {worst['Avg_Score']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Correlation
        if dept_data_indicators is not None and len(dept_data_indicators) > 2:
            st.markdown("### Correlation: Wait Time vs Complaints")
            
            dept_summary = dept_data_indicators.groupby(dept_col).agg({
                'ClinicWaitingTime': 'mean',
                'PatientComplaints': 'mean',
                'Aubnetid': 'nunique'
            }).reset_index()
            
            fig = px.scatter(
                dept_summary,
                x='ClinicWaitingTime', y='PatientComplaints',
                size='Aubnetid',
                color=dept_col,
                hover_name=dept_col,
                title="Waiting Time vs Complaints",
                labels={
                    'ClinicWaitingTime': 'Avg Wait Time (min)',
                    'PatientComplaints': 'Avg Complaints',
                    'Aubnetid': 'Physicians'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 2rem;">
    <p>AUBMC Physician Performance Dashboard - Complete Version</p>
    <p><em>For internal use only</em></p>
</div>
""", unsafe_allow_html=True)
