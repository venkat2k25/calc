import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="NHS Energy Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for white UI with black text
st.markdown("""
<style>
    .main {
        background-color: white;
        color: black;
    }
    
    .stApp {
        background-color: white;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .insight-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        color: #856404;
    }
    
    .explanation-box {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        color: #0c5460;
    }
    
    .formula-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        color: #495057;
    }
    
    .highlight-number {
        font-size: 24px;
        font-weight: bold;
        color: #dc3545;
    }
    
    .section-header {
        background-color: #343a40;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    
    .comparison-text {
        font-size: 16px;
        line-height: 1.6;
        color: #212529;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process the NHS energy data"""
    try:
        # Load the data
        df = pd.read_csv('ERIC_Site.csv')
        trust_df = pd.read_csv('ERIC_TRUST.csv')
        
        # Clean and process data
        df = df.dropna(subset=['Total energy (kWh)', 'Gross floor area (m2)'])
        df = df[df['Total energy (kWh)'] > 0]
        df = df[df['Gross floor area (m2)'] > 0]
        
        # Calculate energy intensity
        df['Energy_Intensity'] = df['Total energy (kWh)'] / df['Gross floor area (m2)']
        
        # Merge with trust data
        df_merged = df.merge(trust_df, on='Trust code', how='left')
        
        return df_merged
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_explanation_box(title, content):
    """Create a styled explanation box"""
    st.markdown(f"""
    <div class="explanation-box">
        <h4 style="color: #0c5460; margin-bottom: 10px;">{title}</h4>
        <p style="margin: 0; line-height: 1.6;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_insight_box(content):
    """Create a styled insight box"""
    st.markdown(f"""
    <div class="insight-box">
        <p style="margin: 0; font-weight: bold;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_formula_box(formula, explanation):
    """Create a styled formula box"""
    st.markdown(f"""
    <div class="formula-box">
        <strong>Formula:</strong> {formula}<br>
        <strong>Explanation:</strong> {explanation}
    </div>
    """, unsafe_allow_html=True)

def energy_analysis_page(df):
    """Energy consumption analysis with detailed explanations"""
    st.markdown('<div class="section-header">🔋 ENERGY CONSUMPTION ANALYSIS</div>', unsafe_allow_html=True)
    
    # Calculate key metrics
    total_energy = df['Total energy (kWh)'].sum()
    total_floor_area = df['Gross floor area (m2)'].sum()
    avg_energy_intensity = total_energy / total_floor_area
    
    create_formula_box(
        "Average Energy Intensity = Total Energy (kWh) ÷ Total Gross Floor Area (m²)",
        f"We sum all energy consumption across {len(df)} NHS sites ({total_energy:,.0f} kWh) and divide by total floor area ({total_floor_area:,.0f} m²) to get the NHS average energy intensity of {avg_energy_intensity:.2f} kWh/m²"
    )
    
    # Top energy consumers
    top_consumers = df.nlargest(15, 'Total energy (kWh)')
    highest_consumer = top_consumers.iloc[0]
    
    create_insight_box(f"🏆 HIGHEST ENERGY CONSUMER: {highest_consumer['Site name']} ({highest_consumer['Trust name']}) uses {highest_consumer['Total energy (kWh)']:,.0f} kWh annually - that's {(highest_consumer['Total energy (kWh)']/df['Total energy (kWh)'].mean()*100-100):.1f}% above the average NHS site consumption of {df['Total energy (kWh)'].mean():,.0f} kWh")
    
    fig1 = px.bar(
        top_consumers, 
        x='Total energy (kWh)', 
        y='Site name',
        title="Top 15 Energy Consuming NHS Sites",
        orientation='h',
        color='Total energy (kWh)',
        color_continuous_scale='Reds',
        text='Total energy (kWh)'
    )
    fig1.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        height=600
    )
    fig1.update_traces(texttemplate='%{text:,.0f} kWh', textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)
    
    create_explanation_box(
        "Why This Chart Matters",
        f"This chart identifies the NHS sites with the highest absolute energy consumption. The top consumer uses {highest_consumer['Total energy (kWh)']:,.0f} kWh annually, which is equivalent to powering approximately {int(highest_consumer['Total energy (kWh)']/3500)} average UK homes for a year (assuming 3,500 kWh per home). These high-consumption sites represent the biggest opportunities for energy savings through efficiency improvements."
    )
    
    # Energy intensity analysis
    st.markdown('<div class="section-header">📊 ENERGY INTENSITY ANALYSIS</div>', unsafe_allow_html=True)
    
    # Most efficient trusts
    trust_efficiency = df.groupby('Trust name').agg({
        'Total energy (kWh)': 'sum',
        'Gross floor area (m2)': 'sum'
    }).reset_index()
    trust_efficiency['Energy_Intensity'] = trust_efficiency['Total energy (kWh)'] / trust_efficiency['Gross floor area (m2)']
    trust_efficiency = trust_efficiency.sort_values('Energy_Intensity')
    
    most_efficient = trust_efficiency.iloc[0]
    least_efficient = trust_efficiency.iloc[-1]
    
    create_insight_box(f"🌟 MOST EFFICIENT TRUST: {most_efficient['Trust name']} achieves {most_efficient['Energy_Intensity']:.2f} kWh/m² - that's {(1-most_efficient['Energy_Intensity']/avg_energy_intensity)*100:.1f}% more efficient than the NHS average. Meanwhile, {least_efficient['Trust name']} uses {least_efficient['Energy_Intensity']:.2f} kWh/m² - {(least_efficient['Energy_Intensity']/avg_energy_intensity-1)*100:.1f}% above average.")
    
    fig2 = px.bar(
        trust_efficiency.head(15),
        x='Energy_Intensity',
        y='Trust name',
        title="15 Most Energy Efficient NHS Trusts (kWh per m²)",
        orientation='h',
        color='Energy_Intensity',
        color_continuous_scale='Greens_r',
        text='Energy_Intensity'
    )
    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        height=600
    )
    fig2.update_traces(texttemplate='%{text:.1f} kWh/m²', textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)
    
    create_explanation_box(
        "Energy Intensity Calculation Explained",
        f"Energy Intensity = Total Energy Consumption ÷ Gross Floor Area. This metric allows fair comparison between trusts of different sizes. A lower number indicates better efficiency. The NHS average is {avg_energy_intensity:.2f} kWh/m². The most efficient trust ({most_efficient['Trust name']}) demonstrates that significant energy savings are possible across the NHS estate."
    )
    
    # Distribution analysis
    st.markdown('<div class="section-header">📈 ENERGY DISTRIBUTION PATTERNS</div>', unsafe_allow_html=True)
    
    fig3 = px.histogram(
        df, 
        x='Energy_Intensity',
        nbins=50,
        title="Distribution of Energy Intensity Across NHS Sites",
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add mean and median lines
    mean_intensity = df['Energy_Intensity'].mean()
    median_intensity = df['Energy_Intensity'].median()
    
    fig3.add_vline(x=mean_intensity, line_dash="dash", line_color="red", 
                   annotation_text=f"Mean: {mean_intensity:.2f} kWh/m²")
    fig3.add_vline(x=median_intensity, line_dash="dash", line_color="green", 
                   annotation_text=f"Median: {median_intensity:.2f} kWh/m²")
    
    fig3.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Statistical analysis
    q1 = df['Energy_Intensity'].quantile(0.25)
    q3 = df['Energy_Intensity'].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df['Energy_Intensity'] < q1 - 1.5*iqr) | (df['Energy_Intensity'] > q3 + 1.5*iqr)]
    
    create_explanation_box(
        "Statistical Distribution Analysis",
        f"The histogram shows energy intensity distribution across {len(df)} NHS sites. Mean intensity is {mean_intensity:.2f} kWh/m² while median is {median_intensity:.2f} kWh/m². The difference indicates a right-skewed distribution - most sites cluster around lower values with some high-consumption outliers. We identified {len(outliers)} statistical outliers that consume significantly more energy than typical sites, representing priority targets for energy efficiency interventions."
    )
    
    # Quartile analysis
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #28a745;">Most Efficient Quarter</h4>
            <p class="highlight-number">{q1:.2f} kWh/m²</p>
            <p>Bottom 25% of sites use less than this amount per m²</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #ffc107;">Below Average</h4>
            <p class="highlight-number">{median_intensity:.2f} kWh/m²</p>
            <p>Median consumption - half of all sites are below this level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #fd7e14;">Above Average</h4>
            <p class="highlight-number">{q3:.2f} kWh/m²</p>
            <p>Top 25% of sites exceed this consumption level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #dc3545;">Outliers</h4>
            <p class="highlight-number">{len(outliers)}</p>
            <p>Sites with exceptionally high energy use requiring urgent attention</p>
        </div>
        """, unsafe_allow_html=True)

def financial_analysis_page(df):
    """Financial analysis with detailed cost explanations"""
    st.markdown('<div class="section-header">💰 FINANCIAL IMPACT ANALYSIS</div>', unsafe_allow_html=True)
    
    # Assume average electricity cost
    avg_cost_per_kwh = 0.15  # £0.15 per kWh (typical commercial rate)
    
    create_formula_box(
        "Annual Energy Cost = Total Energy (kWh) × Cost per kWh (£0.15)",
        f"We calculate energy costs using the average commercial electricity rate of £0.15 per kWh. This rate is based on typical NHS energy procurement contracts and includes both unit costs and standing charges averaged across the estate."
    )
    
    df['Annual_Cost'] = df['Total energy (kWh)'] * avg_cost_per_kwh
    df['Cost_per_m2'] = df['Annual_Cost'] / df['Gross floor area (m2)']
    
    total_annual_cost = df['Annual_Cost'].sum()
    
    create_insight_box(f"💸 TOTAL NHS ENERGY SPEND: The NHS spends approximately £{total_annual_cost:,.0f} annually on electricity across these {len(df)} sites. This represents {total_annual_cost/1000000:.1f} million pounds that could be reduced through energy efficiency measures.")
    
    # Highest cost sites
    highest_cost_sites = df.nlargest(15, 'Annual_Cost')
    highest_cost_site = highest_cost_sites.iloc[0]
    
    fig1 = px.bar(
        highest_cost_sites,
        x='Annual_Cost',
        y='Site name',
        title="Top 15 Highest Energy Cost NHS Sites (Annual £)",
        orientation='h',
        color='Annual_Cost',
        color_continuous_scale='Reds',
        text='Annual_Cost'
    )
    fig1.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        height=600
    )
    fig1.update_traces(texttemplate='£%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)
    
    create_insight_box(f"🏆 HIGHEST ENERGY COST: {highest_cost_site['Site name']} spends £{highest_cost_site['Annual_Cost']:,.0f} annually on electricity - equivalent to the average annual salary of {int(highest_cost_site['Annual_Cost']/35000)} NHS nurses (assuming £35,000 average salary).")
    
    # Savings potential analysis
    st.markdown('<div class="section-header">💡 ENERGY SAVINGS POTENTIAL</div>', unsafe_allow_html=True)
    
    # Calculate potential savings if all sites achieved top quartile efficiency
    target_efficiency = df['Energy_Intensity'].quantile(0.25)  # Top 25% efficiency
    
    df['Potential_Energy_Savings'] = np.maximum(0, 
        (df['Energy_Intensity'] - target_efficiency) * df['Gross floor area (m2)'])
    df['Potential_Cost_Savings'] = df['Potential_Energy_Savings'] * avg_cost_per_kwh
    
    total_potential_savings = df['Potential_Cost_Savings'].sum()
    sites_with_savings = len(df[df['Potential_Cost_Savings'] > 0])
    
    create_formula_box(
        "Savings Potential = (Current Intensity - Target Intensity) × Floor Area × Cost per kWh",
        f"We calculate potential savings by comparing each site's current energy intensity ({df['Energy_Intensity'].mean():.2f} kWh/m² average) to the top quartile performance ({target_efficiency:.2f} kWh/m²). If all sites achieved this efficiency level, the NHS could save £{total_potential_savings:,.0f} annually."
    )
    
    savings_opportunities = df[df['Potential_Cost_Savings'] > 0].nlargest(15, 'Potential_Cost_Savings')
    
    fig2 = px.bar(
        savings_opportunities,
        x='Potential_Cost_Savings',
        y='Site name',
        title="Top 15 Energy Cost Savings Opportunities (Annual £)",
        orientation='h',
        color='Potential_Cost_Savings',
        color_continuous_scale='Greens',
        text='Potential_Cost_Savings'
    )
    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        height=600
    )
    fig2.update_traces(texttemplate='£%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)
    
    biggest_opportunity = savings_opportunities.iloc[0]
    create_insight_box(f"🎯 BIGGEST SAVINGS OPPORTUNITY: {biggest_opportunity['Site name']} could save £{biggest_opportunity['Potential_Cost_Savings']:,.0f} annually by improving to top-quartile efficiency. This would reduce their energy intensity from {biggest_opportunity['Energy_Intensity']:.2f} to {target_efficiency:.2f} kWh/m² - a {((biggest_opportunity['Energy_Intensity']-target_efficiency)/biggest_opportunity['Energy_Intensity']*100):.1f}% reduction.")
    
    # Cost per m² analysis
    st.markdown('<div class="section-header">📊 COST EFFICIENCY ANALYSIS</div>', unsafe_allow_html=True)
    
    fig3 = px.box(
        df,
        y='Cost_per_m2',
        title="Distribution of Energy Costs per Square Meter Across NHS Sites",
        color_discrete_sequence=['#1f77b4']
    )
    fig3.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    avg_cost_per_m2 = df['Cost_per_m2'].mean()
    median_cost_per_m2 = df['Cost_per_m2'].median()
    
    create_explanation_box(
        "Cost per Square Meter Analysis",
        f"The box plot shows the distribution of energy costs per m² across NHS sites. The average cost is £{avg_cost_per_m2:.2f}/m² while the median is £{median_cost_per_m2:.2f}/m². The box shows the interquartile range (middle 50% of sites), while outliers represent sites with exceptionally high or low costs per unit area. This metric helps identify sites that may benefit from targeted energy efficiency investments."
    )
    
    # Investment analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #28a745;">Total Savings Potential</h4>
            <p class="highlight-number">£{total_potential_savings:,.0f}</p>
            <p>Annual savings if all sites achieved top-quartile efficiency</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #17a2b8;">Sites with Savings Potential</h4>
            <p class="highlight-number">{sites_with_savings}</p>
            <p>Number of sites that could benefit from efficiency improvements</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #ffc107;">Average Payback Period</h4>
            <p class="highlight-number">3-5 years</p>
            <p>Typical ROI for NHS energy efficiency investments</p>
        </div>
        """, unsafe_allow_html=True)

def carbon_analysis_page(df):
    """Carbon footprint analysis with environmental context"""
    st.markdown('<div class="section-header">🌱 CARBON FOOTPRINT ANALYSIS</div>', unsafe_allow_html=True)
    
    # Carbon conversion factor (UK grid average)
    carbon_factor = 0.233  # kg CO2 per kWh (2023 UK grid average)
    
    create_formula_box(
        "Carbon Emissions = Total Energy (kWh) × Carbon Factor (0.233 kg CO₂/kWh)",
        f"We calculate carbon emissions using the UK electricity grid carbon intensity factor of 0.233 kg CO₂ per kWh. This factor represents the average carbon emissions from UK electricity generation in 2023, including the mix of renewable, nuclear, gas, and other sources. The factor has decreased significantly over recent years due to increased renewable energy adoption."
    )
    
    df['Carbon_Emissions'] = df['Total energy (kWh)'] * carbon_factor / 1000  # Convert to tonnes
    df['Carbon_per_m2'] = df['Carbon_Emissions'] / df['Gross floor area (m2)']
    
    total_carbon = df['Carbon_Emissions'].sum()
    
    create_insight_box(f"🌍 TOTAL NHS CARBON FOOTPRINT: These NHS sites generate {total_carbon:,.0f} tonnes of CO₂ annually from electricity consumption. This is equivalent to the annual emissions from {int(total_carbon/4.6):,} average UK cars (assuming 4.6 tonnes CO₂ per car per year).")
    
    # Highest carbon emitters
    highest_carbon = df.nlargest(15, 'Carbon_Emissions')
    top_emitter = highest_carbon.iloc[0]
    
    fig1 = px.bar(
        highest_carbon,
        x='Carbon_Emissions',
        y='Site name',
        title="Top 15 Carbon Emitting NHS Sites (Tonnes CO₂ per year)",
        orientation='h',
        color='Carbon_Emissions',
        color_continuous_scale='Reds',
        text='Carbon_Emissions'
    )
    fig1.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        height=600
    )
    fig1.update_traces(texttemplate='%{text:.0f}t CO₂', textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)
    
    create_insight_box(f"🏭 HIGHEST CARBON EMITTER: {top_emitter['Site name']} generates {top_emitter['Carbon_Emissions']:.0f} tonnes of CO₂ annually - equivalent to the emissions from {int(top_emitter['Carbon_Emissions']/4.6):,} cars. To offset this, approximately {int(top_emitter['Carbon_Emissions']/0.025):,} mature trees would need to be planted (assuming 25kg CO₂ absorption per tree per year).")
    
    # Carbon intensity by trust type
    if 'Trust type' in df.columns:
        st.markdown('<div class="section-header">🏥 CARBON INTENSITY BY TRUST TYPE</div>', unsafe_allow_html=True)
        
        trust_type_carbon = df.groupby('Trust type').agg({
            'Carbon_Emissions': 'sum',
            'Gross floor area (m2)': 'sum',
            'Site name': 'count'
        }).reset_index()
        trust_type_carbon['Carbon_per_m2'] = trust_type_carbon['Carbon_Emissions'] / trust_type_carbon['Gross floor area (m2)']
        trust_type_carbon = trust_type_carbon.sort_values('Carbon_per_m2', ascending=False)
        
        fig2 = px.bar(
            trust_type_carbon,
            x='Trust type',
            y='Carbon_per_m2',
            title="Carbon Emissions per m² by NHS Trust Type",
            color='Carbon_per_m2',
            color_continuous_scale='Oranges',
            text='Carbon_per_m2'
        )
        fig2.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black',
            xaxis_tickangle=-45
        )
        fig2.update_traces(texttemplate='%{text:.3f}t/m²', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
        
        highest_carbon_type = trust_type_carbon.iloc[0]
        create_explanation_box(
            "Trust Type Carbon Analysis",
            f"{highest_carbon_type['Trust type']} trusts have the highest carbon intensity at {highest_carbon_type['Carbon_per_m2']:.3f} tonnes CO₂/m². This could be due to the specialized equipment and 24/7 operations typical of these facilities. The variation between trust types highlights the need for tailored decarbonization strategies based on operational requirements."
        )
    
    # NHS Net Zero pathway
    st.markdown('<div class="section-header">🎯 NHS NET ZERO PATHWAY</div>', unsafe_allow_html=True)
    
    # NHS Net Zero targets
    current_year = 2024
    net_zero_year = 2040
    interim_target_year = 2032
    interim_reduction = 0.8  # 80% reduction by 2032
    
    years = list(range(current_year, net_zero_year + 1))
    current_emissions = total_carbon
    
    # Linear reduction pathway
    emissions_pathway = []
    for year in years:
        if year <= interim_target_year:
            # 80% reduction by 2032
            reduction_factor = (year - current_year) / (interim_target_year - current_year) * interim_reduction
        else:
            # Remaining 20% reduction from 2032 to 2040
            remaining_years = net_zero_year - interim_target_year
            years_past_interim = year - interim_target_year
            additional_reduction = (years_past_interim / remaining_years) * 0.2
            reduction_factor = interim_reduction + additional_reduction
        
        emissions_pathway.append(current_emissions * (1 - reduction_factor))
    
    pathway_df = pd.DataFrame({
        'Year': years,
        'Emissions': emissions_pathway
    })
    
    fig3 = px.line(
        pathway_df,
        x='Year',
        y='Emissions',
        title="NHS Net Zero Carbon Reduction Pathway",
        markers=True
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="green", 
                   annotation_text="Net Zero Target")
    fig3.add_vline(x=interim_target_year, line_dash="dash", line_color="orange", 
                   annotation_text="80% Reduction Target")
    fig3.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    required_annual_reduction = current_emissions * interim_reduction / (interim_target_year - current_year)
    
    create_explanation_box(
        "NHS Net Zero Commitment",
        f"The NHS has committed to achieving net zero carbon emissions by 2040, with an interim target of 80% reduction by 2032. Based on current emissions of {current_emissions:,.0f} tonnes CO₂, the NHS needs to reduce emissions by {required_annual_reduction:,.0f} tonnes CO₂ per year to meet the 2032 target. This requires a combination of energy efficiency improvements, renewable energy adoption, and operational changes across the entire NHS estate."
    )
    
    # Environmental equivalencies
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cars_equivalent = int(total_carbon / 4.6)
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #dc3545;">Car Equivalent</h4>
            <p class="highlight-number">{cars_equivalent:,}</p>
            <p>Annual emissions equivalent to this many cars</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        trees_needed = int(total_carbon / 0.025)
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #28a745;">Trees to Offset</h4>
            <p class="highlight-number">{trees_needed:,}</p>
            <p>Mature trees needed to absorb this CO₂</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        homes_equivalent = int(total_carbon / 2.3)
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #17a2b8;">Home Equivalent</h4>
            <p class="highlight-number">{homes_equivalent:,}</p>
            <p>Annual emissions from this many UK homes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        flights_equivalent = int(total_carbon / 0.255)
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #ffc107;">Flight Equivalent</h4>
            <p class="highlight-number">{flights_equivalent:,}</p>
            <p>London to Paris flights worth of emissions</p>
        </div>
        """, unsafe_allow_html=True)

def trust_analysis_page(df):
    """Individual trust analysis with peer comparisons"""
    st.markdown('<div class="section-header">🏥 INDIVIDUAL TRUST ANALYSIS</div>', unsafe_allow_html=True)
    
    # Trust selector
    trust_names = sorted(df['Trust name'].dropna().unique())
    selected_trust = st.selectbox("Select a Trust for Detailed Analysis:", trust_names)
    
    if selected_trust:
        trust_data = df[df['Trust name'] == selected_trust]
        
        if len(trust_data) > 0:
            # Trust overview
            total_sites = len(trust_data)
            total_energy = trust_data['Total energy (kWh)'].sum()
            total_area = trust_data['Gross floor area (m2)'].sum()
            trust_intensity = total_energy / total_area
            
            create_insight_box(f"📊 TRUST OVERVIEW: {selected_trust} operates {total_sites} sites with total energy consumption of {total_energy:,.0f} kWh across {total_area:,.0f} m² of floor space, achieving an overall energy intensity of {trust_intensity:.2f} kWh/m².")
            
            # Calculate percentile ranking
            all_trust_intensities = df.groupby('Trust name').apply(
                lambda x: x['Total energy (kWh)'].sum() / x['Gross floor area (m2)'].sum()
            ).sort_values()
            
            trust_rank = (all_trust_intensities < trust_intensity).sum() + 1
            total_trusts = len(all_trust_intensities)
            percentile = (total_trusts - trust_rank) / total_trusts * 100
            
            create_formula_box(
                "Percentile Ranking = (Number of trusts with higher intensity ÷ Total trusts) × 100",
                f"{selected_trust} ranks {trust_rank} out of {total_trusts} trusts for energy efficiency. This places them in the {percentile:.1f}th percentile - meaning they perform better than {(total_trusts-trust_rank)/total_trusts*100:.1f}% of NHS trusts."
            )
            
            # Site-level analysis
            if len(trust_data) > 1:
                st.markdown('<div class="section-header">🏢 SITE-LEVEL PERFORMANCE</div>', unsafe_allow_html=True)
                
                fig1 = px.bar(
                    trust_data.sort_values('Energy_Intensity'),
                    x='Energy_Intensity',
                    y='Site name',
                    title=f"Energy Intensity by Site - {selected_trust}",
                    orientation='h',
                    color='Energy_Intensity',
                    color_continuous_scale='RdYlGn_r',
                    text='Energy_Intensity'
                )
                fig1.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='black',
                    height=max(400, len(trust_data) * 30)
                )
                fig1.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig1, use_container_width=True)
                
                best_site = trust_data.loc[trust_data['Energy_Intensity'].idxmin()]
                worst_site = trust_data.loc[trust_data['Energy_Intensity'].idxmax()]
                
                create_insight_box(f"🏆 BEST PERFORMING SITE: {best_site['Site name']} achieves {best_site['Energy_Intensity']:.2f} kWh/m². WORST PERFORMING SITE: {worst_site['Site name']} uses {worst_site['Energy_Intensity']:.2f} kWh/m² - that's {(worst_site['Energy_Intensity']/best_site['Energy_Intensity']-1)*100:.1f}% higher than the trust's best site.")
            
            # Peer comparison
            st.markdown('<div class="section-header">👥 PEER GROUP COMPARISON</div>', unsafe_allow_html=True)
            
            if 'Trust type' in df.columns and pd.notna(trust_data['Trust type'].iloc[0]):
                trust_type = trust_data['Trust type'].iloc[0]
                peer_trusts = df[df['Trust type'] == trust_type]
                
                peer_comparison = peer_trusts.groupby('Trust name').agg({
                    'Total energy (kWh)': 'sum',
                    'Gross floor area (m2)': 'sum'
                }).reset_index()
                peer_comparison['Energy_Intensity'] = peer_comparison['Total energy (kWh)'] / peer_comparison['Gross floor area (m2)']
                peer_comparison = peer_comparison.sort_values('Energy_Intensity')
                
                # Highlight selected trust
                peer_comparison['Highlight'] = peer_comparison['Trust name'] == selected_trust
                
                fig2 = px.bar(
                    peer_comparison,
                    x='Energy_Intensity',
                    y='Trust name',
                    title=f"Energy Intensity Comparison - {trust_type} Trusts",
                    orientation='h',
                    color='Highlight',
                    color_discrete_map={True: '#ff7f0e', False: '#1f77b4'},
                    text='Energy_Intensity'
                )
                fig2.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='black',
                    height=max(400, len(peer_comparison) * 25),
                    showlegend=False
                )
                fig2.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)
                
                peer_rank = (peer_comparison['Energy_Intensity'] < trust_intensity).sum() + 1
                total_peers = len(peer_comparison)
                peer_percentile = (total_peers - peer_rank) / total_peers * 100
                
                create_explanation_box(
                    "Peer Group Analysis",
                    f"Among {trust_type} trusts, {selected_trust} ranks {peer_rank} out of {total_peers} for energy efficiency ({peer_percentile:.1f}th percentile). The best performing {trust_type} trust achieves {peer_comparison.iloc[0]['Energy_Intensity']:.2f} kWh/m², while the average for this trust type is {peer_comparison['Energy_Intensity'].mean():.2f} kWh/m²."
                )
            
            # Financial impact
            st.markdown('<div class="section-header">💰 FINANCIAL ANALYSIS</div>', unsafe_allow_html=True)
            
            avg_cost_per_kwh = 0.15
            annual_cost = total_energy * avg_cost_per_kwh
            
            # Calculate potential savings
            target_efficiency = df['Energy_Intensity'].quantile(0.25)
            potential_energy_savings = max(0, (trust_intensity - target_efficiency) * total_area)
            potential_cost_savings = potential_energy_savings * avg_cost_per_kwh
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #dc3545;">Annual Energy Cost</h4>
                    <p class="highlight-number">£{annual_cost:,.0f}</p>
                    <p>Total electricity spend across all sites</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #28a745;">Potential Savings</h4>
                    <p class="highlight-number">£{potential_cost_savings:,.0f}</p>
                    <p>Annual savings if achieving top-quartile efficiency</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                savings_percentage = (potential_cost_savings / annual_cost * 100) if annual_cost > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #17a2b8;">Savings Potential</h4>
                    <p class="highlight-number">{savings_percentage:.1f}%</p>
                    <p>Percentage reduction in energy costs possible</p>
                </div>
                """, unsafe_allow_html=True)
            
            if potential_cost_savings > 0:
                create_insight_box(f"💡 SAVINGS OPPORTUNITY: {selected_trust} could save £{potential_cost_savings:,.0f} annually ({savings_percentage:.1f}% reduction) by improving energy efficiency to match the top 25% of NHS trusts. This would require reducing energy intensity from {trust_intensity:.2f} to {target_efficiency:.2f} kWh/m².")

def overview_page(df):
    """Overview dashboard with key insights"""
    st.markdown('<div class="section-header">📊 NHS ENERGY PERFORMANCE OVERVIEW</div>', unsafe_allow_html=True)
    
    # Key metrics
    total_sites = len(df)
    total_energy = df['Total energy (kWh)'].sum()
    total_area = df['Gross floor area (m2)'].sum()
    avg_intensity = total_energy / total_area
    total_trusts = df['Trust name'].nunique()
    
    create_insight_box(f"🏥 NHS ESTATE OVERVIEW: Analysis covers {total_sites:,} NHS sites across {total_trusts} trusts, representing {total_area:,.0f} m² of healthcare facilities consuming {total_energy:,.0f} kWh annually - equivalent to powering {int(total_energy/3500):,} average UK homes.")
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #1f77b4;">Total Sites</h4>
            <p class="highlight-number">{total_sites:,}</p>
            <p>NHS healthcare facilities analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #ff7f0e;">Total Energy</h4>
            <p class="highlight-number">{total_energy/1000000:.1f}M</p>
            <p>kWh consumed annually</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #2ca02c;">Average Intensity</h4>
            <p class="highlight-number">{avg_intensity:.2f}</p>
            <p>kWh per m² across NHS estate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        annual_cost = total_energy * 0.15
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #d62728;">Annual Cost</h4>
            <p class="highlight-number">£{annual_cost/1000000:.1f}M</p>
            <p>Estimated electricity spend</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Efficiency distribution
    st.markdown('<div class="section-header">⚡ ENERGY EFFICIENCY DISTRIBUTION</div>', unsafe_allow_html=True)
    
    # Categorize sites by efficiency
    q1 = df['Energy_Intensity'].quantile(0.25)
    q2 = df['Energy_Intensity'].quantile(0.50)
    q3 = df['Energy_Intensity'].quantile(0.75)
    
    def categorize_efficiency(intensity):
        if intensity <= q1:
            return 'Highly Efficient (Top 25%)'
        elif intensity <= q2:
            return 'Above Average (25-50%)'
        elif intensity <= q3:
            return 'Below Average (50-75%)'
        else:
            return 'Poor Efficiency (Bottom 25%)'
    
    df['Efficiency_Category'] = df['Energy_Intensity'].apply(categorize_efficiency)
    
    efficiency_counts = df['Efficiency_Category'].value_counts()
    
    fig1 = px.pie(
        values=efficiency_counts.values,
        names=efficiency_counts.index,
        title="Distribution of NHS Sites by Energy Efficiency",
        color_discrete_sequence=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    )
    fig1.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    create_explanation_box(
        "Efficiency Distribution Analysis",
        f"The pie chart shows that {efficiency_counts.get('Highly Efficient (Top 25%)', 0)} sites ({efficiency_counts.get('Highly Efficient (Top 25%)', 0)/total_sites*100:.1f}%) achieve top-quartile efficiency (≤{q1:.2f} kWh/m²), while {efficiency_counts.get('Poor Efficiency (Bottom 25%)', 0)} sites ({efficiency_counts.get('Poor Efficiency (Bottom 25%)', 0)/total_sites*100:.1f}%) have poor efficiency (>{q3:.2f} kWh/m²). This distribution highlights significant variation in energy performance across the NHS estate."
    )
    
    # Trust type comparison
    if 'Trust type' in df.columns:
        st.markdown('<div class="section-header">🏥 PERFORMANCE BY TRUST TYPE</div>', unsafe_allow_html=True)
        
        trust_type_data = []
        for trust_type in df['Trust type'].dropna().unique():
            type_data = df[df['Trust type'] == trust_type]
            trust_type_data.append({
                'Trust Type': trust_type,
                'Energy Intensity': type_data['Energy_Intensity'].values
            })
        
        # Create box plot data
        fig2 = go.Figure()
        
        for i, data in enumerate(trust_type_data):
            fig2.add_trace(go.Box(
                y=data['Energy Intensity'],
                name=data['Trust Type'],
                boxpoints='outliers'
            ))
        
        fig2.update_layout(
            title="Energy Intensity Distribution by Trust Type",
            yaxis_title="Energy Intensity (kWh/m²)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Calculate trust type averages
        trust_type_avg = df.groupby('Trust type')['Energy_Intensity'].agg(['mean', 'count']).reset_index()
        trust_type_avg = trust_type_avg.sort_values('mean')
        
        best_type = trust_type_avg.iloc[0]
        worst_type = trust_type_avg.iloc[-1]
        
        create_insight_box(f"🏆 BEST PERFORMING TRUST TYPE: {best_type['Trust type']} trusts average {best_type['mean']:.2f} kWh/m² across {best_type['count']} sites. HIGHEST CONSUMPTION: {worst_type['Trust type']} trusts average {worst_type['mean']:.2f} kWh/m² - {(worst_type['mean']/best_type['mean']-1)*100:.1f}% higher than the most efficient trust type.")
    
    # Top and bottom performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">🌟 TOP PERFORMERS</div>', unsafe_allow_html=True)
        top_performers = df.nsmallest(10, 'Energy_Intensity')[['Site name', 'Trust name', 'Energy_Intensity']]
        
        for idx, row in top_performers.iterrows():
            st.markdown(f"""
            <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 10px; margin: 5px 0;">
                <strong>{row['Site name']}</strong><br>
                <small>{row['Trust name']}</small><br>
                <span style="color: #155724; font-weight: bold;">{row['Energy_Intensity']:.2f} kWh/m²</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">⚠️ IMPROVEMENT OPPORTUNITIES</div>', unsafe_allow_html=True)
        bottom_performers = df.nlargest(10, 'Energy_Intensity')[['Site name', 'Trust name', 'Energy_Intensity']]
        
        for idx, row in bottom_performers.iterrows():
            st.markdown(f"""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 10px; margin: 5px 0;">
                <strong>{row['Site name']}</strong><br>
                <small>{row['Trust name']}</small><br>
                <span style="color: #721c24; font-weight: bold;">{row['Energy_Intensity']:.2f} kWh/m²</span>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application"""
    st.title("🏥 NHS Energy Analysis Dashboard")
    st.markdown("**Comprehensive analysis of NHS energy consumption, costs, and carbon emissions with detailed mathematical explanations**")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Unable to load data. Please ensure the CSV files are available.")
        return
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Page:",
        ["Overview", "Energy Analysis", "Financial Analysis", "Carbon Analysis", "Trust Analysis"]
    )
    
    # Display selected page
    if page == "Overview":
        overview_page(df)
    elif page == "Energy Analysis":
        energy_analysis_page(df)
    elif page == "Financial Analysis":
        financial_analysis_page(df)
    elif page == "Carbon Analysis":
        carbon_analysis_page(df)
    elif page == "Trust Analysis":
        trust_analysis_page(df)

if __name__ == "__main__":
    main()