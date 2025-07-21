import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from io import BytesIO
import google.generativeai as genai


GEMINI_API_KEY = "AIzaSyDUeI-vKkIO_HfMD1jVo_ccKhoiXDjf5V8"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("🚨 Gemini API key missing.")
    st.stop()

@st.cache_data
def load_data(uploaded_file, encoding='utf-8-sig'):
    """Loads data from CSV or Parquet, handling potential errors."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, encoding=encoding)
        elif uploaded_file.name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Parquet file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def convert_df_to_csv(df):
    """Converts DataFrame to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def calculate_kpis(df):
    """Calculates baseline, target, and savings KPIs."""
    df_processed = df.copy()

    # Data Cleaning and Pre-calculation
    df_processed['Total CO₂ (tCO₂e)'] = df_processed['Total CO₂ (kgCO₂e)'] / 1000
    df_processed['CO₂ per m² (tCO₂/m²)'] = df_processed['CO2_per_m2'] / 1000
    df_processed.rename(columns={'kWh_per_m2': 'Energy per m² (kWh/m²)'}, inplace=True)

    numeric_cols = [
        'Energy per m² (kWh/m²)', 'CO₂ per m² (tCO₂/m²)', 'cost_per_kWh',
        'Gross internal floor area (m²)', 'Bed Count', 'Total Energy (kWh)',
        'Total Costs (£)'
    ]
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

    # Per-bed KPIs
    df_processed['Energy per Bed (kWh)'] = (df_processed['Total Energy (kWh)'] / df_processed['Bed Count']).replace([np.inf, -np.inf], 0).fillna(0)
    df_processed['CO₂ per Bed (tCO₂e)'] = (df_processed['Total CO₂ (tCO₂e)'] / df_processed['Bed Count']).replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate Benchmarks (Target KPIs)
    if 'Trust Type' in df_processed.columns:
        df_processed['Target Energy per m²'] = df_processed.groupby('Trust Type')['Energy per m² (kWh/m²)'].transform('median')
        df_processed['Target CO₂ per m²'] = df_processed.groupby('Trust Type')['CO₂ per m² (tCO₂/m²)'].transform('median')
        df_processed['Target cost_per_kWh'] = df_processed.groupby('Trust Type')['cost_per_kWh'].transform('median')
        df_processed['Target Energy per Bed'] = df_processed.groupby('Trust Type')['Energy per Bed (kWh)'].transform('median')
        df_processed['Target CO₂ per Bed'] = df_processed.groupby('Trust Type')['CO₂ per Bed (tCO₂e)'].transform('median')

    # Calculate Deviations and Savings Potential
    df_processed['Energy Deviation (kWh/m²)'] = df_processed['Energy per m² (kWh/m²)'] - df_processed['Target Energy per m²']
    df_processed['Energy Deviation (%)'] = (df_processed['Energy Deviation (kWh/m²)'] / df_processed['Target Energy per m²']).replace([np.inf, -np.inf], 0).fillna(0) * 100

    df_processed['Potential Energy Saved (kWh)'] = (df_processed['Energy Deviation (kWh/m²)'] * df_processed['Gross internal floor area (m²)']).apply(lambda x: max(x, 0))
    df_processed['Potential Cost Saved (£)'] = df_processed['Potential Energy Saved (kWh)'] * df_processed['cost_per_kWh']
    df_processed['Potential CO₂ Saved (tCO₂)'] = ((df_processed['CO₂ per m² (tCO₂/m²)'] - df_processed['Target CO₂ per m²']) * df_processed['Gross internal floor area (m²)']).apply(lambda x: max(x, 0))

    # Efficiency Labeling
    def get_efficiency_label(row):
        if row['Target Energy per m²'] == 0:
            return 'No Benchmark'
        ratio = row['Energy per m² (kWh/m²)'] / row['Target Energy per m²']
        if ratio > 1.2:
            return 'High-Risk'
        elif ratio < 0.9:
            return 'Efficient'
        else:
            return 'Moderate'
    df_processed['Clustering Efficiency Label'] = df_processed.apply(get_efficiency_label, axis=1)

    df_processed['Cluster Distance Metric'] = df_processed.groupby('Trust Type')['Energy per m² (kWh/m²)'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0).fillna(0)

    return df_processed

def explain_calculation(title, formula, variables, result, reasoning):
    """Creates a standardized explanation box for calculations."""
    st.info(f"""
    **📊 {title}**
    
    **Formula:** {formula}
    
    **Variables:**
    {variables}
    
    **Result:** {result}
    
    **Why this matters:** {reasoning}
    """)

def create_top_performers_chart(df, metric, title, unit, ascending=True):
    """Creates a horizontal bar chart for top/bottom performers"""
    top_10 = df.nlargest(10, metric) if not ascending else df.nsmallest(10, metric)
    
    fig = px.bar(
        top_10, 
        x=metric, 
        y='Trust Name',
        orientation='h',
        title=title,
        color=metric,
        color_continuous_scale='RdYlGn_r' if not ascending else 'RdYlGn'
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending' if ascending else 'total descending'},
        xaxis_title=unit,
        showlegend=False
    )
    
    return fig, top_10

def create_efficiency_distribution_chart(df):
    """Creates efficiency distribution pie chart"""
    efficiency_counts = df['Clustering Efficiency Label'].value_counts()
    
    colors = {
        'Efficient': '#2E8B57',
        'Moderate': '#FFD700', 
        'High-Risk': '#DC143C',
        'No Benchmark': '#808080'
    }
    
    fig = px.pie(
        values=efficiency_counts.values,
        names=efficiency_counts.index,
        title='Trust Efficiency Distribution',
        color=efficiency_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_trust_type_comparison(df):
    """Creates trust type comparison box plot"""
    if 'Trust Type' not in df.columns:
        return None
        
    fig = px.box(
        df, 
        x='Trust Type', 
        y='Energy per m² (kWh/m²)',
        title='Energy Consumption Distribution by Trust Type',
        color='Trust Type'
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig

def create_correlation_heatmap(df):
    """Creates correlation heatmap for key metrics"""
    numeric_cols = [
        'Energy per m² (kWh/m²)', 
        'CO₂ per m² (tCO₂/m²)', 
        'cost_per_kWh',
        'Bed Count',
        'Gross internal floor area (m²)'
    ]
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[available_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title='Correlation Matrix of Key Metrics',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    fig.update_layout(height=500)
    return fig

def create_savings_potential_chart(df):
    """Creates savings potential visualization"""
    top_savings = df.nlargest(15, 'Potential Cost Saved (£)')
    
    fig = px.bar(
        top_savings,
        x='Trust Name',
        y='Potential Cost Saved (£)',
        title='Top 15 Trusts by Cost Savings Potential',
        color='Potential Cost Saved (£)',
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig, top_savings

def page_data_preprocessing():
    st.title("📊 Data Preprocessing")
    st.markdown("Upload ERIC Site and ERIC Trust CSV files to begin analysis.")

    st.info("""
    **Why Data Preprocessing?**
    
    Raw NHS data comes in separate files (site-level and trust-level) that need to be:
    1. **Merged** - Combining site data with trust metadata
    2. **Aggregated** - Summing energy consumption across all sites per trust
    3. **Normalized** - Converting to per-unit metrics (per m², per bed)
    4. **Validated** - Handling missing values and data quality issues
    
    This preprocessing ensures consistent, comparable metrics across all trusts.
    """)

    col1, col2 = st.columns(2)
    site_file = col1.file_uploader("Upload ERIC_Site.csv", type=["csv"])
    trust_file = col2.file_uploader("Upload ERIC_TRUST.csv", type=["csv"])

    if site_file and trust_file:
        try:
            site_data = load_data(site_file)
            trust_data = load_data(trust_file)
            if site_data is None or trust_data is None:
                return

            with st.spinner("Processing data..."):
                # Energy columns aggregation
                energy_columns = [
                    'Thermal energy consumption (KWh)', 'Electrical energy consumption (KWh)',
                    'Electricity - green electricity consumed (kWh)', 'Electricity - trust owned solar consumed (kWh)',
                    'Electricity - third party owned solar consumed (kWh)', 'Electricity - other renewables consumed (kWh)',
                    'Gas consumed (kWh)', 'Oil consumed (kWh)', 'Non-fossil fuel - renewable consumed (kWh)',
                    'Steam consumed (kWh)', 'Hot water consumed (kWh)', 'Solar electricity generated (kWh)'
                ]
                
                st.subheader("Preprocessing Steps")
                
                explain_calculation(
                    "Energy Aggregation",
                    "Total Energy = Σ(All Energy Sources)",
                    f"• Number of energy sources: {len(energy_columns)}\n• Sources include: Thermal, Electrical, Gas, Oil, Renewables, etc.",
                    f"Sum of all energy consumption types per site",
                    "We sum all energy sources to get total consumption because hospitals use multiple energy types."
                )
                
                cost_columns = [
                    'Electricity - green electricity tariff costs (£)', 'Electricity - trust owned solar costs (£)',
                    'Electricity - third party owned solar costs (£)', 'Electricity - other renewables costs (£)',
                    'Electricity - other costs (£)', 'Gas costs (£)', 'Oil costs (£)',
                    'Non-fossil fuel - renewable costs (£)', 'Other energy costs (£)'
                ]
                
                explain_calculation(
                    "Cost Aggregation",
                    "Total Costs = Σ(All Energy Costs)",
                    f"• Number of cost categories: {len(cost_columns)}\n• Includes: Electricity tariffs, gas costs, renewable costs, etc.",
                    f"Sum of all energy-related costs per site",
                    "Total costs capture the complete financial impact of energy consumption."
                )
                
                trust_co2_columns = [
                    'Waste re-use scheme - Carbon savings (CO2e (tonnes))',
                    'Carbon savings from investment in energy efficient schemes (CO2e (tonnes))'
                ]

                available_energy_columns = [col for col in energy_columns if col in site_data.columns]
                available_cost_columns = [col for col in cost_columns if col in site_data.columns]
                available_trust_co2_columns = [col for col in trust_co2_columns if col in trust_data.columns]

                site_data['Total Energy (kWh)'] = site_data[available_energy_columns].sum(axis=1, skipna=True)
                site_data['Total Costs (£)'] = site_data[available_cost_columns].sum(axis=1, skipna=True)

                # Aggregate by Trust Code
                agg_dict = {
                    'Total Energy (kWh)': 'sum', 'Total Costs (£)': 'sum',
                    'Gross internal floor area (m²)': 'sum',
                    'Single bedrooms for patients with en-suite facilities (No.)': 'sum',
                    'Single bedrooms for patients without en-suite facilities (No.)': 'sum',
                    'Isolation rooms (No.)': 'sum', 'Trust Name': 'first'
                }
                
                explain_calculation(
                    "Trust-Level Aggregation",
                    "Trust Total = Σ(All Sites per Trust)",
                    f"• Energy: Sum across all sites\n• Costs: Sum across all sites\n• Floor Area: Sum across all sites\n• Beds: Sum across all sites",
                    f"Combined metrics for each trust",
                    "We aggregate to get trust-level performance, the management unit for energy decisions."
                )
                
                available_agg_dict = {k: v for k, v in agg_dict.items() if k in site_data.columns}
                site_data_agg = site_data.groupby('Trust Code').agg(available_agg_dict).reset_index()

                # Merge with trust data
                trust_merge_cols = ['Trust Code', 'Trust Name']
                if 'Trust Type' in trust_data.columns:
                    trust_merge_cols.append('Trust Type')
                trust_merge_cols.extend(available_trust_co2_columns)

                merged_data = pd.merge(site_data_agg, trust_data[trust_merge_cols], on='Trust Code', how='left', suffixes=('_site', '_trust'))
                merged_data['Trust Name'] = merged_data['Trust Name_trust'].combine_first(merged_data['Trust Name_site'])

                # Calculate bed count and KPIs
                bed_cols = [
                    'Single bedrooms for patients with en-suite facilities (No.)',
                    'Single bedrooms for patients without en-suite facilities (No.)',
                    'Isolation rooms (No.)'
                ]
                available_bed_columns = [col for col in bed_cols if col in merged_data.columns]
                merged_data['Bed Count'] = merged_data[available_bed_columns].sum(axis=1, skipna=True)
                merged_data['Total CO₂ (kgCO₂e)'] = merged_data[available_trust_co2_columns].sum(axis=1, skipna=True) * 1000

                # Calculate per-unit metrics
                with np.errstate(divide='ignore', invalid='ignore'):
                    merged_data['kWh_per_m2'] = merged_data['Total Energy (kWh)'] / merged_data['Gross internal floor area (m²)']
                    merged_data['CO2_per_m2'] = merged_data['Total CO₂ (kgCO₂e)'] / merged_data['Gross internal floor area (m²)']
                    merged_data['cost_per_kWh'] = merged_data['Total Costs (£)'] / merged_data['Total Energy (kWh)']

                explain_calculation(
                    "Intensity Metrics",
                    "Energy per m² = Total Energy ÷ Floor Area",
                    f"• Energy per m² = kWh ÷ m²\n• CO₂ per m² = kgCO₂e ÷ m²\n• Cost per kWh = £ ÷ kWh",
                    f"Normalized performance metrics",
                    "Per-unit metrics allow fair comparison between trusts of different sizes."
                )

                merged_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                merged_data = calculate_kpis(merged_data)

                st.session_state.processed_data = merged_data
                st.success("✅ Data processed successfully!")
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(merged_data.head(), use_container_width=True)

                # Download processed data
                csv = convert_df_to_csv(merged_data)
                st.download_button(
                    label="📥 Download Processed Data",
                    data=csv,
                    file_name="processed_nhs_data.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")


def page_overview(df):
    st.title("📈 Mathematical Overview")
    st.markdown("**Problem Statement:** Calculate key performance indicators for NHS Trust efficiency analysis")
    
    # Key Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_trusts = df['Trust Name'].nunique()
    total_energy_kwh = df['Total Energy (kWh)'].sum()
    total_cost_pounds = df['Total Costs (£)'].sum()
    total_co2_tonnes = df['Total CO₂ (tCO₂e)'].sum()
    
    with col1:
        st.metric("Total Trusts", f"{total_trusts:,}")
    with col2:
        st.metric("Total Energy", f"{total_energy_kwh/1e6:,.1f} GWh")
    with col3:
        st.metric("Total Costs", f"£{total_cost_pounds/1e6:,.1f}M")
    with col4:
        st.metric("Total CO₂", f"{total_co2_tonnes/1e3:,.1f} ktCO₂e")
    
    # Problem 1: Basic Counts and Totals
    st.subheader("Problem 1: Basic Calculations")
    st.markdown("**Given:** Dataset of NHS Trust performance metrics")
    st.markdown("**Find:** Total counts and sums")
    
    st.markdown("**Solution:**")
    st.markdown(f"• Total Trusts = {total_trusts:,}")
    st.markdown(f"• Total Energy = Σ(Total Energy) = {total_energy_kwh:,.0f} kWh")
    st.markdown(f"• Total Costs = Σ(Total Costs) = £{total_cost_pounds:,.0f}")
    st.markdown(f"• Total CO₂ = Σ(Total CO₂) = {total_co2_tonnes:,.0f} tCO₂e")
    
    # Efficiency Distribution Visualization
    st.subheader("📊 Trust Efficiency Distribution")
    efficiency_fig = create_efficiency_distribution_chart(df)
    st.plotly_chart(efficiency_fig, use_container_width=True)
    
    efficiency_counts = df['Clustering Efficiency Label'].value_counts()
    
    st.markdown("**Efficiency Analysis:**")
    for label, count in efficiency_counts.items():
        percentage = (count / total_trusts) * 100
        emoji = "🟢" if label == "Efficient" else "🟡" if label == "Moderate" else "🔴"
        st.markdown(f"• {emoji} **{label}**: {count} trusts ({percentage:.1f}%)")
    
    explain_calculation(
        "Efficiency Classification Method",
        "Efficiency Ratio = Actual Energy per m² ÷ Target Energy per m²",
        f"• Efficient: Ratio < 0.9\n• Moderate: 0.9 ≤ Ratio ≤ 1.2\n• High-Risk: Ratio > 1.2",
        f"Performance-based trust categorization",
        "This classification identifies trusts needing urgent attention or performing well."
    )
    
    # Trust Type Comparison
    if 'Trust Type' in df.columns:
        st.subheader("🏥 Trust Type Performance Comparison")
        trust_type_fig = create_trust_type_comparison(df)
        if trust_type_fig:
            st.plotly_chart(trust_type_fig, use_container_width=True)
        
        # Trust type statistics
        trust_type_stats = df.groupby('Trust Type').agg({
            'Energy per m² (kWh/m²)': ['mean', 'count'],
            'Potential Cost Saved (£)': 'sum'
        }).round(1)
        
        st.markdown("**Trust Type Analysis:**")
        for trust_type in trust_type_stats.index:
            avg_energy = trust_type_stats.loc[trust_type, ('Energy per m² (kWh/m²)', 'mean')]
            count = trust_type_stats.loc[trust_type, ('Energy per m² (kWh/m²)', 'count')]
            savings = trust_type_stats.loc[trust_type, ('Potential Cost Saved (£)', 'sum')]
            st.markdown(f"• **{trust_type}** ({count} trusts): {avg_energy:.1f} kWh/m² avg, £{savings:,.0f} savings potential")

def page_energy(df):
    st.title("⚡ Energy Mathematics")
    st.markdown("**Problem Set:** Energy consumption analysis and statistical calculations")
    
    # Key Energy Metrics
    energy_values = df['Energy per m² (kWh/m²)'].dropna()
    avg_e_per_m2 = energy_values.mean()
    median_e_per_m2 = energy_values.median()
    total_potential_e_saved = df['Potential Energy Saved (kWh)'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Energy", f"{avg_e_per_m2:,.1f} kWh/m²")
    with col2:
        st.metric("Median Energy", f"{median_e_per_m2:,.1f} kWh/m²")
    with col3:
        st.metric("Potential Savings", f"{total_potential_e_saved/1e6:,.1f} GWh")
    with col4:
        min_energy = energy_values.min()
        max_energy = energy_values.max()
        st.metric("Energy Range", f"{max_energy - min_energy:,.1f} kWh/m²")
    
    # Top Energy Consumers
    st.subheader("🔥 Highest Energy Consumers")
    energy_fig, top_energy = create_top_performers_chart(
        df, 'Energy per m² (kWh/m²)', 
        'Top 10 Trusts by Energy Consumption per m²', 
        'kWh/m²', 
        ascending=False
    )
    st.plotly_chart(energy_fig, use_container_width=True)
    
    # Highlight top consumer
    if len(top_energy) > 0:
        top_consumer = top_energy.iloc[0]
        st.warning(f"🚨 **Highest energy usage**: {top_consumer['Trust Name']} consumes {top_consumer['Energy per m² (kWh/m²)']:,.1f} kWh/m², which is {((top_consumer['Energy per m² (kWh/m²)'] / avg_e_per_m2) - 1) * 100:.1f}% above average!")
    
    # Most Efficient Trusts
    st.subheader("🌟 Most Energy Efficient Trusts")
    efficient_fig, top_efficient = create_top_performers_chart(
        df, 'Energy per m² (kWh/m²)', 
        'Top 10 Most Energy Efficient Trusts', 
        'kWh/m²', 
        ascending=True
    )
    st.plotly_chart(efficient_fig, use_container_width=True)
    
    # Highlight most efficient
    if len(top_efficient) > 0:
        most_efficient = top_efficient.iloc[0]
        st.success(f"⭐ **Most efficient**: {most_efficient['Trust Name']} uses only {most_efficient['Energy per m² (kWh/m²)']:,.1f} kWh/m², which is {((avg_e_per_m2 / most_efficient['Energy per m² (kWh/m²)']) - 1) * 100:.1f}% better than average!")
    
    # Energy Distribution
    st.subheader("📊 Energy Consumption Distribution")
    fig = px.histogram(
        df, 
        x='Energy per m² (kWh/m²)', 
        nbins=30, 
        title='Distribution of Energy Consumption per m²',
        color_discrete_sequence=['#1f77b4']
    )
    fig.add_vline(x=avg_e_per_m2, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {avg_e_per_m2:.1f}")
    fig.add_vline(x=median_e_per_m2, line_dash="dash", line_color="green", 
                  annotation_text=f"Median: {median_e_per_m2:.1f}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Analysis
    st.subheader("📈 Statistical Analysis")
    min_energy = energy_values.min()
    max_energy = energy_values.max()
    range_energy = max_energy - min_energy
    std_energy = energy_values.std()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Central Tendency:**")
        st.markdown(f"• Mean = {avg_e_per_m2:,.1f} kWh/m²")
        st.markdown(f"• Median = {median_e_per_m2:,.1f} kWh/m²")
        st.markdown(f"• Mode = {energy_values.mode().iloc[0] if len(energy_values.mode()) > 0 else 'N/A'}")
    
    with col2:
        st.markdown("**Variability:**")
        st.markdown(f"• Range = {range_energy:,.1f} kWh/m²")
        st.markdown(f"• Standard Deviation = {std_energy:,.1f} kWh/m²")
        st.markdown(f"• Coefficient of Variation = {(std_energy/avg_e_per_m2)*100:.1f}%")
    
    explain_calculation(
        "Energy Performance Insights",
        "Performance Gap = (Individual - Benchmark) / Benchmark × 100%",
        f"• Average consumption: {avg_e_per_m2:,.1f} kWh/m²\n• Best performer: {min_energy:,.1f} kWh/m²\n• Worst performer: {max_energy:,.1f} kWh/m²",
        f"Wide variation indicates significant improvement opportunities",
        f"The {range_energy:,.1f} kWh/m² range shows some trusts use {(max_energy/min_energy):.1f}x more energy per m² than others."
    )

def page_financial(df):
    st.title("💰 Financial Mathematics")
    st.markdown("**Problem Set:** Cost analysis and financial calculations")
    
    # Key Financial Metrics
    total_cost_pounds = df['Total Costs (£)'].sum()
    total_energy_kwh = df['Total Energy (kWh)'].sum()
    avg_cost_kwh = df['cost_per_kWh'].mean()
    weighted_avg_cost_kwh = total_cost_pounds / total_energy_kwh if total_energy_kwh > 0 else 0
    total_potential_savings = df['Potential Cost Saved (£)'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Annual Costs", f"£{total_cost_pounds/1e6:,.1f}M")
    with col2:
        st.metric("Avg Cost per kWh", f"£{avg_cost_kwh:.3f}")
    with col3:
        st.metric("Potential Savings", f"£{total_potential_savings/1e6:,.1f}M")
    with col4:
        st.metric("Savings %", f"{(total_potential_savings/total_cost_pounds)*100:.1f}%")
    
    # Cost Savings Potential
    st.subheader("💡 Highest Cost Savings Potential")
    savings_fig, top_savings = create_savings_potential_chart(df)
    st.plotly_chart(savings_fig, use_container_width=True)
    
    # Highlight top savings opportunity
    if len(top_savings) > 0:
        top_saver = top_savings.iloc[0]
        st.info(f"💰 **Biggest savings opportunity**: {top_saver['Trust Name']} could save £{top_saver['Potential Cost Saved (£)']:,.0f} annually through energy efficiency improvements!")
    
    # Highest Cost per kWh
    st.subheader("💸 Highest Energy Costs per kWh")
    cost_fig, top_costs = create_top_performers_chart(
        df, 'cost_per_kWh', 
        'Top 10 Trusts by Cost per kWh', 
        '£/kWh', 
        ascending=False
    )
    st.plotly_chart(cost_fig, use_container_width=True)
    
    # Highlight highest cost
    if len(top_costs) > 0:
        highest_cost = top_costs.iloc[0]
        st.warning(f"🚨 **Highest energy costs**: {highest_cost['Trust Name']} pays £{highest_cost['cost_per_kWh']:.3f} per kWh, which is {((highest_cost['cost_per_kWh'] / avg_cost_kwh) - 1) * 100:.1f}% above average!")
    
    # Cost Distribution Analysis
    st.subheader("📊 Cost Distribution Analysis")
    cost_values = df['cost_per_kWh'].dropna()
    
    fig = px.box(
        df, 
        y='cost_per_kWh', 
        title='Distribution of Energy Costs per kWh Across Trusts',
        color_discrete_sequence=['#ff7f0e']
    )
    fig.add_hline(y=avg_cost_kwh, line_dash="dash", line_color="red", 
                  annotation_text=f"Average: £{avg_cost_kwh:.3f}")
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Analysis
    st.subheader("📈 Return on Investment Analysis")
    typical_investment_per_kwh_saved = 2.5
    investment_cost = df['Potential Energy Saved (kWh)'] * typical_investment_per_kwh_saved
    annual_savings = df['Potential Cost Saved (£)']
    
    total_investment = investment_cost.sum()
    total_annual_savings = annual_savings.sum()
    simple_payback_years = total_investment / total_annual_savings if total_annual_savings > 0 else float('inf')
    roi_percent = (total_annual_savings / total_investment) * 100 if total_investment > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Investment Required", f"£{total_investment/1e6:,.1f}M")
    with col2:
        st.metric("Payback Period", f"{simple_payback_years:.1f} years")
    with col3:
        st.metric("Annual ROI", f"{roi_percent:.1f}%")
    
    explain_calculation(
        "ROI Analysis Results",
        "Payback Period = Investment Cost ÷ Annual Savings; ROI = (Annual Savings ÷ Investment) × 100",
        f"• Total investment needed: £{total_investment/1e6:,.1f}M\n• Annual savings potential: £{total_annual_savings/1e6:,.1f}M",
        f"Financial viability assessment",
        f"A payback period of {simple_payback_years:.1f} years is {'very attractive' if simple_payback_years < 3 else 'reasonable' if simple_payback_years < 5 else 'challenging'} for energy efficiency investments."
    )

def page_carbon(df):
    st.title("🌍 Carbon Mathematics")
    st.markdown("**Problem Set:** Carbon emissions analysis and environmental calculations")
    
    # Key Carbon Metrics
    total_co2_tonnes = df['Total CO₂ (tCO₂e)'].sum()
    total_energy_kwh = df['Total Energy (kWh)'].sum()
    avg_co2_per_m2 = df['CO₂ per m² (tCO₂/m²)'].mean()
    carbon_intensity = total_co2_tonnes / (total_energy_kwh / 1000) if total_energy_kwh > 0 else 0
    total_potential_co2_saved = df['Potential CO₂ Saved (tCO₂)'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total CO₂ Emissions", f"{total_co2_tonnes/1e3:,.1f} ktCO₂e")
    with col2:
        st.metric("Carbon Intensity", f"{carbon_intensity:.3f} tCO₂e/MWh")
    with col3:
        st.metric("Potential CO₂ Savings", f"{total_potential_co2_saved/1e3:,.1f} ktCO₂e")
    with col4:
        st.metric("Avg CO₂ per m²", f"{avg_co2_per_m2:.3f} tCO₂e/m²")
    
    # Highest Carbon Emitters
    st.subheader("🏭 Highest Carbon Emitters per m²")
    carbon_fig, top_carbon = create_top_performers_chart(
        df, 'CO₂ per m² (tCO₂/m²)', 
        'Top 10 Trusts by CO₂ Emissions per m²', 
        'tCO₂e/m²', 
        ascending=False
    )
    st.plotly_chart(carbon_fig, use_container_width=True)
    
    # Highlight top emitter
    if len(top_carbon) > 0:
        top_emitter = top_carbon.iloc[0]
        st.warning(f"🚨 **Highest carbon emissions**: {top_emitter['Trust Name']} emits {top_emitter['CO₂ per m² (tCO₂/m²)']:,.3f} tCO₂e/m², which is {((top_emitter['CO₂ per m² (tCO₂/m²)'] / avg_co2_per_m2) - 1) * 100:.1f}% above average!")
    
    # Energy vs Carbon Relationship
    st.subheader("⚡ Energy vs Carbon Intensity Relationship")
    scatter_fig = px.scatter(
        df, 
        x='Energy per m² (kWh/m²)', 
        y='CO₂ per m² (tCO₂/m²)', 
        color='Trust Type' if 'Trust Type' in df.columns else None,
        size='Gross internal floor area (m²)',
        title='Energy vs Carbon Intensity by Trust',
        hover_data=['Trust Name']
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Carbon Reduction Targets
    st.subheader("🎯 NHS Net Zero Carbon Targets")
    current_year = 2025
    target_year_80 = 2030
    target_year_100 = 2040
    
    years_to_80_target = target_year_80 - current_year
    target_emissions_80 = total_co2_tonnes * 0.2
    annual_reduction_80 = (total_co2_tonnes - target_emissions_80) / years_to_80_target
    
    # Create target visualization
    years = list(range(current_year, target_year_100 + 1))
    current_emissions = [total_co2_tonnes] + [0] * (len(years) - 1)
    target_80_line = []
    target_100_line = []
    
    for year in years:
        if year <= target_year_80:
            reduction_so_far = (year - current_year) * annual_reduction_80
            target_80_line.append(max(0, total_co2_tonnes - reduction_so_far))
        else:
            target_80_line.append(target_emissions_80)
        
        if year <= target_year_100:
            target_100_line.append(max(0, total_co2_tonnes * (1 - (year - current_year) / (target_year_100 - current_year))))
        else:
            target_100_line.append(0)
    
    target_fig = go.Figure()
    target_fig.add_trace(go.Scatter(x=years, y=current_emissions, mode='markers', name='Current Emissions', marker=dict(color='red', size=10)))
    target_fig.add_trace(go.Scatter(x=years, y=target_80_line, mode='lines', name='80% Reduction Target', line=dict(color='orange', dash='dash')))
    target_fig.add_trace(go.Scatter(x=years, y=target_100_line, mode='lines', name='Net Zero Target', line=dict(color='green', dash='dot')))
    
    target_fig.update_layout(
        title='NHS Carbon Reduction Pathway',
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (tCO₂e)',
        height=500
    )
    st.plotly_chart(target_fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("2030 Target", f"{target_emissions_80/1e3:,.1f} ktCO₂e", f"-{((total_co2_tonnes - target_emissions_80)/total_co2_tonnes)*100:.0f}%")
    with col2:
        st.metric("Annual Reduction Needed", f"{annual_reduction_80/1e3:,.1f} ktCO₂e/year")
    with col3:
        st.metric("Reduction Rate", f"{(annual_reduction_80/total_co2_tonnes)*100:.1f}%/year")
    
    # Environmental Impact Equivalencies
    st.subheader("🌱 Environmental Impact Context")
    cars_per_tonne_co2 = 0.22
    trees_per_tonne_co2 = 40
    homes_per_tonne_co2 = 0.2
    
    equivalent_cars = total_co2_tonnes * cars_per_tonne_co2
    equivalent_trees = total_co2_tonnes * trees_per_tonne_co2
    equivalent_homes = total_co2_tonnes * homes_per_tonne_co2
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🚗 Equivalent Cars", f"{equivalent_cars:,.0f}", "driven for 1 year")
    with col2:
        st.metric("🌳 Trees Needed", f"{equivalent_trees:,.0f}", "to offset emissions")
    with col3:
        st.metric("🏠 Equivalent Homes", f"{equivalent_homes:,.0f}", "annual emissions")
    
    explain_calculation(
        "Carbon Reduction Challenge",
        "Annual Reduction = (Current - Target) ÷ Years Available",
        f"• Current emissions: {total_co2_tonnes/1e3:,.1f} ktCO₂e\n• 2030 target: {target_emissions_80/1e3:,.1f} ktCO₂e\n• Time available: {years_to_80_target} years",
        f"NHS must reduce emissions by {(annual_reduction_80/total_co2_tonnes)*100:.1f}% annually",
        f"This requires unprecedented efficiency improvements and renewable energy adoption across all NHS trusts."
    )

def page_trust_analysis(df):
    st.title("🏥 Trust Performance Analysis")
    st.markdown("**Problem Set:** Individual trust performance and peer comparison")
    
    # Trust Selection
    st.subheader("🔍 Select Trust for Detailed Analysis")
    selected_trust = st.selectbox("Choose a Trust:", df['Trust Name'].unique())
    
    if selected_trust:
        trust_data = df[df['Trust Name'] == selected_trust].iloc[0]
        
        # Trust Overview
        st.subheader(f"📊 {selected_trust} - Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Energy per m²", f"{trust_data['Energy per m² (kWh/m²)']:,.1f} kWh/m²")
        with col2:
            st.metric("Cost per kWh", f"£{trust_data['cost_per_kWh']:.3f}")
        with col3:
            st.metric("CO₂ per m²", f"{trust_data['CO₂ per m² (tCO₂/m²)']:,.3f} tCO₂e/m²")
        with col4:
            efficiency_label = trust_data['Clustering Efficiency Label']
            emoji = "🟢" if efficiency_label == "Efficient" else "🟡" if efficiency_label == "Moderate" else "🔴"
            st.metric("Efficiency Rating", f"{emoji} {efficiency_label}")
        
        # Peer Comparison
        if 'Trust Type' in df.columns:
            trust_type = trust_data['Trust Type']
            peer_group = df[df['Trust Type'] == trust_type]
            
            st.subheader(f"👥 Peer Comparison - {trust_type}")
            
            # Create peer comparison chart
            peer_metrics = ['Energy per m² (kWh/m²)', 'cost_per_kWh', 'CO₂ per m² (tCO₂/m²)']
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Energy per m²', 'Cost per kWh', 'CO₂ per m²'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            for i, metric in enumerate(peer_metrics, 1):
                # Box plot for peer group
                fig.add_trace(
                    go.Box(y=peer_group[metric], name=f'{trust_type} Peers', showlegend=False),
                    row=1, col=i
                )
                # Highlight selected trust
                fig.add_trace(
                    go.Scatter(
                        x=[0], y=[trust_data[metric]], 
                        mode='markers', 
                        marker=dict(color='red', size=15, symbol='star'),
                        name=selected_trust if i == 1 else None,
                        showlegend=True if i == 1 else False
                    ),
                    row=1, col=i
                )
            
            fig.update_layout(height=400, title_text=f"{selected_trust} vs {trust_type} Peers")
            st.plotly_chart(fig, use_container_width=True)
            
            # Peer statistics
            peer_stats = peer_group[peer_metrics].describe()
            
            st.markdown("**Peer Group Statistics:**")
            for metric in peer_metrics:
                trust_value = trust_data[metric]
                peer_median = peer_stats.loc['50%', metric]
                peer_mean = peer_stats.loc['mean', metric]
                percentile = (peer_group[metric] <= trust_value).mean() * 100
                
                if metric == 'Energy per m² (kWh/m²)':
                    unit = "kWh/m²"
                elif metric == 'cost_per_kWh':
                    unit = "£/kWh"
                else:
                    unit = "tCO₂e/m²"
                
                performance = "better" if (metric == 'Energy per m² (kWh/m²)' or metric == 'cost_per_kWh' or metric == 'CO₂ per m² (tCO₂/m²)') and trust_value < peer_median else "worse"
                
                st.markdown(f"• **{metric}**: {trust_value:.3f} {unit} (Peer median: {peer_median:.3f}, {percentile:.0f}th percentile - {performance} than median)")
        
        # Improvement Opportunities
        st.subheader("💡 Improvement Opportunities")
        
        potential_energy_saved = trust_data['Potential Energy Saved (kWh)']
        potential_cost_saved = trust_data['Potential Cost Saved (£)']
        potential_co2_saved = trust_data['Potential CO₂ Saved (tCO₂)']
        
        if potential_energy_saved > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Energy Savings Potential", f"{potential_energy_saved:,.0f} kWh")
            with col2:
                st.metric("Cost Savings Potential", f"£{potential_cost_saved:,.0f}")
            with col3:
                st.metric("CO₂ Savings Potential", f"{potential_co2_saved:,.1f} tCO₂e")
            
            st.success(f"💰 **{selected_trust}** could save £{potential_cost_saved:,.0f} annually and reduce CO₂ emissions by {potential_co2_saved:,.1f} tonnes through energy efficiency improvements!")
        else:
            st.info(f"✅ **{selected_trust}** is already performing at or above its peer benchmark!")
    
    # Trust Rankings
    st.subheader("🏆 Trust Performance Rankings")
    
    # Create ranking dataframe
    ranking_df = df.copy()
    ranking_df['Energy Rank'] = ranking_df['Energy per m² (kWh/m²)'].rank(method='min')
    ranking_df['Cost Rank'] = ranking_df['cost_per_kWh'].rank(method='min')
    ranking_df['Carbon Rank'] = ranking_df['CO₂ per m² (tCO₂/m²)'].rank(method='min')
    ranking_df['Overall Rank'] = (ranking_df['Energy Rank'] + ranking_df['Cost Rank'] + ranking_df['Carbon Rank']) / 3
    
    # Top performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌟 Top 5 Overall Performers:**")
        top_performers = ranking_df.nsmallest(5, 'Overall Rank')
        for i, (_, row) in enumerate(top_performers.iterrows(), 1):
            st.markdown(f"{i}. **{row['Trust Name']}** (Rank: {row['Overall Rank']:.1f})")
    
    with col2:
        st.markdown("**⚠️ Bottom 5 Performers:**")
        bottom_performers = ranking_df.nlargest(5, 'Overall Rank')
        for i, (_, row) in enumerate(bottom_performers.iterrows(), 1):
            st.markdown(f"{i}. **{row['Trust Name']}** (Rank: {row['Overall Rank']:.1f})")
    
    # Correlation Analysis
    st.subheader("🔗 Performance Correlation Analysis")
    corr_fig = create_correlation_heatmap(df)
    st.plotly_chart(corr_fig, use_container_width=True)

def page_ai_analysis():
    st.title("🤖 AI-Justified Analysis")
    st.markdown("**Analyze individual NHS site energy efficiency using AI to infer service types and justify energy scores.**")

    uploaded_file = st.file_uploader("Upload NHS site data (.csv or .parquet)", type=["csv", "parquet"], key="ai_file_uploader")
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None:
            return

        df.fillna(0, inplace=True)

        # Compute total energy
        def compute_total_energy(row):
            return (
                row.get("Thermal energy consumption (KWh)", 0)
                + row.get("Electrical energy consumption (KWh)", 0)
                + row.get("Gas consumed (kWh)", 0)
                + row.get("Oil consumed (kWh)", 0)
                + row.get("Steam consumed (kWh)", 0)
                + row.get("Hot water consumed (kWh)", 0)
            )

        df["total_energy_kwh"] = df.apply(compute_total_energy, axis=1)
        df["energy_per_m2"] = df["total_energy_kwh"] / (df["Gross internal floor area (m²)"] + 1e-6)

        # Trust Type Based Scoring
        def get_percentile_score(row, df):
            trust_type = row.get("Trust Type", "Unknown")
            peer_group = df[df["Trust Type"] == trust_type]["energy_per_m2"]
            energy_val = row["energy_per_m2"]
            if len(peer_group) > 1:
                percentile_rank = np.sum(peer_group <= energy_val) / len(peer_group) * 100
                banded_score = int(np.ceil(percentile_rank / 5.0)) * 5
                return min(banded_score, 100)
            return 50

        df["energy_score"] = df.apply(lambda row: get_percentile_score(row, df), axis=1)

        # Site overview metrics
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sites", len(df))
        with col2:
            st.metric("Avg Energy Score", f"{df['energy_score'].mean():.1f}/100")
        with col3:
            st.metric("Total Energy", f"{df['total_energy_kwh'].sum()/1e6:,.1f} GWh")
        with col4:
            st.metric("Avg Energy/m²", f"{df['energy_per_m2'].mean():,.1f} kWh/m²")

        # Gemini AI Prompt
        def infer_services_gemini(row):
            prompt = f"""
You're analyzing NHS site infrastructure data to classify services and justify energy efficiency.

**Site Name:** {row.get('Site Name', 'N/A')}

---

## Site Energy Efficiency
- Energy usage per square metre: {row.get('energy_per_m2', 0):.2f} kWh/m²
- Energy score (0–100, based on percentile vs similar Trust Types): {row.get('energy_score', 0)} / 100

**Interpretation:** Higher scores mean worse energy efficiency (e.g., score 100 = very inefficient, 0 = very efficient).

---

## Infrastructure Details
- Pathology area: {row.get('Pathology (m²)', 0)} m²
- Clinical Sterile Services Dept (CSSD): {row.get('Clinical Sterile Services Dept. (CSSD) (m²)', 0)} m²
- Isolation rooms: {row.get('Isolation rooms (No.)', 0)}
- Single ensuite beds: {row.get('Single bedrooms for patients with en-suite facilities (No.)', 0)}
- 999 Contact Centre: {row.get('999 Contact Centre (m²)', 0)} m²
- Hub (make ready station): {row.get('Hub (make ready station) (m²)', 0)} m²
- Ambulance Station: {row.get('Ambulance Station (m²)', 0)} m²
- Staff Accommodation: {row.get('Staff Accommodation (m²)', 0)} m²
- Medical Records area: {row.get('Medical records (m²)', 0)} m²
- Restaurants/Cafés: {row.get("Restaurants and cafés (m²)", 0)} m²

Other Info:
- Reported service type (if available): {row.get('Service types', 'N/A')}

---

### Your Task:
1. Based on the site name and infrastructure, **infer the most likely service type** (e.g., Acute Hospital, Mental Health Unit, Ambulance Station, Community Care Hub).
2. Justify whether the energy score is reasonable or surprising, given the infrastructure and service mix.

### Format your output exactly like this:
**Inferred Service Type:** [Inferred type]
**Justification:** [Explain what infrastructure/services drive energy usage — e.g., 24/7 operation, labs, imaging, ambulance fleet, etc.]
"""
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"Error in AI analysis: {str(e)}"

        # Site Selection with Search
        st.subheader("🔍 Site Analysis")
        
        # Search functionality
        search_term = st.text_input("🔎 Search for a site:", placeholder="Type site name...")
        
        if search_term:
            filtered_sites = df[df['Site Name'].str.contains(search_term, case=False, na=False)]['Site Name'].unique()
            if len(filtered_sites) > 0:
                site_name = st.selectbox("Select from search results:", filtered_sites)
            else:
                st.warning("No sites found matching your search.")
                site_name = st.selectbox("Select NHS Site:", df["Site Name"].unique())
        else:
            site_name = st.selectbox("Select NHS Site:", df["Site Name"].unique())
        
        if site_name:
            selected_row = df[df["Site Name"] == site_name].iloc[0]
            site_energy = selected_row["total_energy_kwh"]
            energy_per_m2 = selected_row["energy_per_m2"]
            score = selected_row["energy_score"]
            
            # Site metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Site", site_name)
            with col2:
                st.metric("Total Energy", f"{int(site_energy):,} kWh")
            with col3:
                st.metric("Energy per m²", f"{energy_per_m2:,.1f} kWh/m²")
            with col4:
                score_color = "🟢" if score <= 30 else "🟡" if score <= 70 else "🔴"
                st.metric("Energy Score", f"{score_color} {score}/100")
            
            # Generate AI analysis
            with st.spinner("🤖 Generating AI analysis..."):
                justification = infer_services_gemini(selected_row)

            # Display Results
            st.subheader("🤖 AI Analysis Results")
            st.info(justification)
            
            # Additional site details
            with st.expander("📋 Detailed Site Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Infrastructure:**")
                    st.markdown(f"• Floor Area: {selected_row.get('Gross internal floor area (m²)', 0):,.0f} m²")
                    st.markdown(f"• Pathology: {selected_row.get('Pathology (m²)', 0):,.0f} m²")
                    st.markdown(f"• CSSD: {selected_row.get('Clinical Sterile Services Dept. (CSSD) (m²)', 0):,.0f} m²")
                    st.markdown(f"• Isolation Rooms: {selected_row.get('Isolation rooms (No.)', 0):,.0f}")
                    st.markdown(f"• En-suite Beds: {selected_row.get('Single bedrooms for patients with en-suite facilities (No.)', 0):,.0f}")
                
                with col2:
                    st.markdown("**Support Services:**")
                    st.markdown(f"• 999 Contact Centre: {selected_row.get('999 Contact Centre (m²)', 0):,.0f} m²")
                    st.markdown(f"• Ambulance Station: {selected_row.get('Ambulance Station (m²)', 0):,.0f} m²")
                    st.markdown(f"• Staff Accommodation: {selected_row.get('Staff Accommodation (m²)', 0):,.0f} m²")
                    st.markdown(f"• Medical Records: {selected_row.get('Medical records (m²)', 0):,.0f} m²")
                    st.markdown(f"• Restaurants/Cafés: {selected_row.get('Restaurants and cafés (m²)', 0):,.0f} m²")
        
        # Energy Score Distribution
        st.subheader("📊 Energy Score Distribution Across All Sites")
        score_fig = px.histogram(
            df, 
            x='energy_score', 
            nbins=20,
            title='Distribution of Energy Scores (0=Most Efficient, 100=Least Efficient)',
            color_discrete_sequence=['#1f77b4']
        )
        score_fig.add_vline(x=df['energy_score'].mean(), line_dash="dash", line_color="red", 
                           annotation_text=f"Average: {df['energy_score'].mean():.1f}")
        st.plotly_chart(score_fig, use_container_width=True)
        
    else:
        st.warning("⚠️ Please upload a valid NHS site data file to begin analysis.")

# -------------------------------------
# Main Application
# -------------------------------------
def main():
    st.set_page_config(
        page_title="NHS Energy Analysis", 
        layout="wide",
        page_icon="🏥",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 0.5rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: #2c3e50;
        margin-top: 2rem;
    }
    
    h3 {
        color: #34495e;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 0.5rem;
        border: none;
        background: linear-gradient(90deg, #1f77b4, #2c3e50);
        color: white;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2c3e50, #1f77b4);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 0.5rem;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border-radius: 0.5rem;
        border: 2px dashed #1f77b4;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1f77b4, #2c3e50);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #1f77b4, #2c3e50); border-radius: 0.5rem; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; border: none;">🏥 NHS Energy Analysis Platform</h1>
        <p style="color: white; margin: 0; font-size: 1.2rem;">Mathematical Analysis & AI-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("## 🧭 Navigation")
    pages = {
        "📊 Data Preprocessing": page_data_preprocessing,
        "📈 Mathematical Overview": page_overview,
        "⚡ Energy Analysis": page_energy,
        "💰 Financial Analysis": page_financial,
        "🌍 Carbon Analysis": page_carbon,
        "🏥 Trust Analysis": page_trust_analysis,
        "🤖 AI-Justified Analysis": page_ai_analysis
    }
    
    selected_page = st.sidebar.selectbox("Select Analysis Page", list(pages.keys()))
    
    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    This platform provides comprehensive mathematical analysis of NHS Trust energy performance with:
    
    • 📊 Statistical calculations
    • 📈 Performance benchmarking  
    • 💰 Financial impact analysis
    • 🌍 Carbon footprint assessment
    • 🤖 AI-powered insights
    """)
    
    # Check if data is processed for non-preprocessing pages
    if selected_page not in ["📊 Data Preprocessing", "🤖 AI-Justified Analysis"] and 'processed_data' not in st.session_state:
        st.warning("⚠️ Please process data first using the **Data Preprocessing** page.")
        st.info("👈 Use the sidebar to navigate to Data Preprocessing and upload your ERIC data files.")
        return
    
    # Run selected page
    if selected_page in ["📊 Data Preprocessing", "🤖 AI-Justified Analysis"]:
        pages[selected_page]()
    else:
        pages[selected_page](st.session_state.processed_data)

if __name__ == "__main__":
    main()