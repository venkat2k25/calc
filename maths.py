import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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


def page_data_preprocessing():
    st.title("Data Preprocessing")
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
                st.success("Data processed successfully!")
                st.dataframe(merged_data.head())

                # Download processed data
                csv = convert_df_to_csv(merged_data)
                st.download_button(
                    label="Download Processed Data",
                    data=csv,
                    file_name="processed_nhs_data.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error during processing: {e}")


def page_overview(df):
    st.title("Mathematical Overview")
    st.markdown("**Problem Statement:** Calculate key performance indicators for NHS Trust efficiency analysis")
    
    # Problem 1: Basic Counts and Totals
    st.subheader("Problem 1: Basic Calculations")
    st.markdown("**Given:** Dataset of NHS Trust performance metrics")
    st.markdown("**Find:** Total counts and sums")
    
    total_trusts = df['Trust Name'].nunique()
    total_energy_kwh = df['Total Energy (kWh)'].sum()
    total_cost_pounds = df['Total Costs (£)'].sum()
    total_co2_tonnes = df['Total CO₂ (tCO₂e)'].sum()
    
    st.markdown("**Solution:**")
    st.markdown(f"• Total Trusts = {total_trusts:,}")
    st.markdown(f"• Total Energy = Σ(Total Energy) = {total_energy_kwh:,.0f} kWh")
    st.markdown(f"• Total Costs = Σ(Total Costs) = £{total_cost_pounds:,.0f}")
    st.markdown(f"• Total CO₂ = Σ(Total CO₂) = {total_co2_tonnes:,.0f} tCO₂e")
    
    explain_calculation(
        "Why These Totals Matter",
        "System Total = Σ(Individual Trust Values)",
        f"• {total_trusts} trusts represent the NHS hospital system\n• Each trust contributes to national energy consumption",
        f"Complete picture of NHS energy footprint",
        "These totals establish the baseline for the NHS hospital system's energy consumption, costs, and carbon emissions."
    )
    
    # Problem 2: Unit Conversions
    st.subheader("Problem 2: Unit Conversions")
    st.markdown("**Given:** Totals in base units")
    st.markdown("**Find:** Convert to larger units for readability")
    
    total_energy_gwh = total_energy_kwh / 1e6
    total_cost_m = total_cost_pounds / 1e6
    total_co2_kt = total_co2_tonnes / 1e3
    
    st.markdown("**Solution:**")
    st.markdown(f"• Energy in GWh = {total_energy_kwh:,.0f} kWh ÷ 1,000,000 = {total_energy_gwh:,.1f} GWh")
    st.markdown(f"• Cost in Millions = £{total_cost_pounds:,.0f} ÷ 1,000,000 = £{total_cost_m:,.1f}M")
    st.markdown(f"• CO₂ in Kilotonnes = {total_co2_tonnes:,.0f} tCO₂e ÷ 1,000 = {total_co2_kt:,.1f} ktCO₂e")
    
    explain_calculation(
        "Unit Conversion Logic",
        "Large Unit = Small Unit ÷ Conversion Factor",
        f"• GWh = Gigawatt-hours (10⁶ kWh)\n• £M = Millions of pounds (10⁶ £)\n• ktCO₂e = Kilotonnes CO₂ equivalent (10³ tonnes)",
        f"More readable large-scale figures",
        "Large numbers are easier to comprehend when converted to appropriate units."
    )
    
    # Problem 3: Efficiency Distribution
    st.subheader("Problem 3: Efficiency Classification")
    st.markdown("**Given:** Efficiency labels based on energy performance")
    st.markdown("**Find:** Count of trusts in each category")
    
    efficiency_counts = df['Clustering Efficiency Label'].value_counts()
    
    st.markdown("**Solution:**")
    for label, count in efficiency_counts.items():
        percentage = (count / total_trusts) * 100
        st.markdown(f"• {label}: {count} trusts ({percentage:.1f}%)")
    
    explain_calculation(
        "Efficiency Classification Method",
        "Efficiency Ratio = Actual Energy per m² ÷ Target Energy per m²",
        f"• Efficient: Ratio < 0.9\n• Moderate: 0.9 ≤ Ratio ≤ 1.2\n• High-Risk: Ratio > 1.2",
        f"Performance-based trust categorization",
        "This classification identifies trusts needing urgent attention or performing well."
    )
    
    # Problem 4: Savings Potential by Trust Type
    st.subheader("Problem 4: Savings Analysis by Trust Type")
    st.markdown("**Given:** Potential cost savings per trust")
    st.markdown("**Find:** Total savings potential grouped by trust type")
    
    if 'Trust Type' in df.columns:
        savings_by_type = df.groupby('Trust Type')['Potential Cost Saved (£)'].sum().sort_values(ascending=False)
        
        st.markdown("**Solution:**")
        for trust_type, savings in savings_by_type.items():
            st.markdown(f"• {trust_type}: £{savings:,.0f}")
        
        explain_calculation(
            "Savings Potential Calculation",
            "Potential Savings = (Actual - Target) × Floor Area × Cost per kWh",
            f"• Only positive deviations count\n• Multiply by floor area to get total kWh savings",
            f"Financial impact of efficiency improvements",
            "This shows the financial benefit if trusts performed at their peer group median."
        )

def page_energy(df):
    st.title("Energy Mathematics")
    st.markdown("**Problem Set:** Energy consumption analysis and statistical calculations")
    
    # Problem 1: Central Tendency
    st.subheader("Problem 1: Central Tendency Measures")
    st.markdown("**Given:** Energy per m² values for all trusts")
    st.markdown("**Find:** Mean, median, and total potential savings")
    
    energy_values = df['Energy per m² (kWh/m²)'].dropna()
    avg_e_per_m2 = energy_values.mean()
    median_e_per_m2 = energy_values.median()
    total_potential_e_saved = df['Potential Energy Saved (kWh)'].sum()
    
    st.markdown("**Solution:**")
    st.markdown(f"• Mean = {avg_e_per_m2:,.1f} kWh/m²")
    st.markdown(f"• Median = {median_e_per_m2:,.1f} kWh/m²")
    st.markdown(f"• Total Potential Savings = {total_potential_e_saved:,.0f} kWh = {total_potential_e_saved/1e6:,.2f} GWh")
    
    explain_calculation(
        "Why Mean vs Median Matter",
        "Mean = Average; Median = Middle Value",
        f"• Mean = {avg_e_per_m2:,.1f} kWh/m²\n• Median = {median_e_per_m2:,.1f} kWh/m²",
        f"Central tendency comparison",
        f"The {'mean > median' if avg_e_per_m2 > median_e_per_m2 else 'median > mean'} suggests the data is {'right-skewed' if avg_e_per_m2 > median_e_per_m2 else 'left-skewed'}."
    )
    
    # Problem 2: Range and Distribution
    st.subheader("Problem 2: Range Analysis")
    st.markdown("**Given:** Energy consumption data")
    st.markdown("**Find:** Minimum, maximum, and range")
    
    min_energy = energy_values.min()
    max_energy = energy_values.max()
    range_energy = max_energy - min_energy
    std_energy = energy_values.std()
    
    st.markdown("**Solution:**")
    st.markdown(f"• Minimum = {min_energy:,.1f} kWh/m²")
    st.markdown(f"• Maximum = {max_energy:,.1f} kWh/m²")
    st.markdown(f"• Range = {range_energy:,.1f} kWh/m²")
    st.markdown(f"• Standard Deviation = {std_energy:,.1f} kWh/m²")
    
    explain_calculation(
        "Variability Analysis",
        "Range = Max - Min; Std Dev = √(Σ(x - μ)²/n)",
        f"• Range = {range_energy:,.1f} kWh/m²\n• Std Dev = {std_energy:,.1f} kWh/m²",
        f"Measure of data spread",
        f"The large range ({range_energy:,.1f} kWh/m²) shows significant variation in energy performance."
    )
    
    # Problem 3: Trust Type Analysis
    st.subheader("Problem 3: Group Statistics")
    st.markdown("**Given:** Energy data grouped by Trust Type")
    st.markdown("**Find:** Average energy consumption per trust type")
    
    if 'Trust Type' in df.columns:
        energy_by_type = df.groupby('Trust Type')['Energy per m² (kWh/m²)'].agg(['mean', 'count', 'std']).round(1)
        
        st.markdown("**Solution:**")
        for trust_type, stats in energy_by_type.iterrows():
            st.markdown(f"• {trust_type}: {stats['mean']:,.1f} kWh/m² (n={stats['count']}, σ={stats['std']:,.1f})")
        
        explain_calculation(
            "Trust Type Differences",
            "Group Mean = Σ(Group Values) ÷ Group Count",
            f"• Different trust types have different energy needs",
            f"Trust type benchmarking basis",
            "Different trust types have different energy profiles due to varying equipment."
        )
    
    # Create energy distribution chart
    st.subheader("Energy Distribution Visualization")
    fig = px.histogram(df, x='Energy per m² (kWh/m²)', nbins=30, 
                      title='Distribution of Energy Consumption per m²')
    fig.add_vline(x=avg_e_per_m2, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {avg_e_per_m2:.1f}")
    fig.add_vline(x=median_e_per_m2, line_dash="dash", line_color="green", 
                  annotation_text=f"Median: {median_e_per_m2:.1f}")
    st.plotly_chart(fig, use_container_width=True)


def page_financial(df):
    st.title("Financial Mathematics")
    st.markdown("**Problem Set:** Cost analysis and financial calculations")
    
    # Problem 1: Cost Metrics
    st.subheader("Problem 1: Cost per Unit Analysis")
    st.markdown("**Given:** Total costs and energy consumption")
    st.markdown("**Find:** Average cost per kWh and total financial metrics")
    
    total_cost_pounds = df['Total Costs (£)'].sum()
    total_energy_kwh = df['Total Energy (kWh)'].sum()
    avg_cost_kwh = df['cost_per_kWh'].mean()
    weighted_avg_cost_kwh = total_cost_pounds / total_energy_kwh if total_energy_kwh > 0 else 0
    total_potential_savings = df['Potential Cost Saved (£)'].sum()
    
    st.markdown("**Solution:**")
    st.markdown(f"• Total Annual Costs = £{total_cost_pounds:,.0f}")
    st.markdown(f"• Simple Average Cost per kWh = £{avg_cost_kwh:.3f}")
    st.markdown(f"• Weighted Average Cost per kWh = £{weighted_avg_cost_kwh:.3f}")
    st.markdown(f"• Total Potential Savings = £{total_potential_savings:,.0f}")
    
    explain_calculation(
        "Cost Analysis Methods",
        "Simple Avg = Σ(Individual Costs/kWh) ÷ n; Weighted Avg = Total £ ÷ Total kWh",
        f"• Simple average = £{avg_cost_kwh:.3f}\n• Weighted average = £{weighted_avg_cost_kwh:.3f}",
        f"Two different cost perspectives",
        "Weighted average accounts for consumption size."
    )
    
    # Problem 2: Cost Distribution
    st.subheader("Problem 2: Cost Distribution Analysis")
    st.markdown("**Given:** Cost per kWh for all trusts")
    st.markdown("**Find:** Statistical measures of cost distribution")
    
    cost_values = df['cost_per_kWh'].dropna()
    min_cost = cost_values.min()
    max_cost = cost_values.max()
    median_cost = cost_values.median()
    std_cost = cost_values.std()
    q1_cost = cost_values.quantile(0.25)
    q3_cost = cost_values.quantile(0.75)
    iqr_cost = q3_cost - q1_cost
    
    st.markdown("**Solution:**")
    st.markdown(f"• Minimum Cost = £{min_cost:.3f} per kWh")
    st.markdown(f"• Maximum Cost = £{max_cost:.3f} per kWh")
    st.markdown(f"• Median Cost = £{median_cost:.3f} per kWh")
    st.markdown(f"• Standard Deviation = £{std_cost:.3f}")
    st.markdown(f"• Interquartile Range = £{iqr_cost:.3f}")
    
    explain_calculation(
        "Cost Variability Insights",
        "IQR = Q3 - Q1",
        f"• Range = £{max_cost - min_cost:.3f}\n• IQR = £{iqr_cost:.3f}",
        f"Cost variation analysis",
        f"The wide range suggests differences in energy procurement."
    )
    
    # Problem 3: ROI Analysis
    st.subheader("Problem 3: Return on Investment Analysis")
    st.markdown("**Given:** Potential savings and estimated implementation costs")
    st.markdown("**Find:** ROI metrics for energy efficiency investments")
    
    typical_investment_per_kwh_saved = 2.5
    investment_cost = df['Potential Energy Saved (kWh)'] * typical_investment_per_kwh_saved
    annual_savings = df['Potential Cost Saved (£)']
    
    total_investment = investment_cost.sum()
    total_annual_savings = annual_savings.sum()
    simple_payback_years = total_investment / total_annual_savings if total_annual_savings > 0 else float('inf')
    roi_percent = (total_annual_savings / total_investment) * 100 if total_investment > 0 else 0
    
    st.markdown("**Solution:**")
    st.markdown(f"• Total Investment Required = £{total_investment:,.0f}")
    st.markdown(f"• Annual Savings Potential = £{total_annual_savings:,.0f}")
    st.markdown(f"• Simple Payback Period = {simple_payback_years:.1f} years")
    st.markdown(f"• Annual ROI = {roi_percent:.1f}%")
    
    explain_calculation(
        "ROI Calculation Method",
        "Payback Period = Investment Cost ÷ Annual Savings; ROI = (Annual Savings ÷ Investment) × 100",
        f"• Investment = {df['Potential Energy Saved (kWh)'].sum():,.0f} kWh × £{typical_investment_per_kwh_saved}",
        f"Financial viability assessment",
        f"A payback period of {simple_payback_years:.1f} years is {'very attractive' if simple_payback_years < 3 else 'reasonable'}."
    )
    
    # Create cost analysis chart
    st.subheader("Cost Analysis Visualization")
    fig = px.box(df, y='cost_per_kWh', title='Distribution of Energy Costs per kWh Across Trusts')
    fig.add_hline(y=avg_cost_kwh, line_dash="dash", line_color="red", 
                  annotation_text=f"Average: £{avg_cost_kwh:.3f}")
    fig.add_hline(y=median_cost, line_dash="dash", line_color="green", 
                  annotation_text=f"Median: £{median_cost:.3f}")
    st.plotly_chart(fig, use_container_width=True)


def page_carbon(df):
    st.title("🌍 Carbon Mathematics")
    st.markdown("**Problem Set:** Carbon emissions analysis and environmental calculations")
    
    # Problem 1: Carbon Intensity
    st.subheader("Problem 1: Carbon Intensity Analysis")
    st.markdown("**Given:** CO₂ emissions and energy consumption data")
    st.markdown("**Find:** Carbon intensity metrics and totals")
    
    total_co2_tonnes = df['Total CO₂ (tCO₂e)'].sum()
    total_energy_kwh = df['Total Energy (kWh)'].sum()
    avg_co2_per_m2 = df['CO₂ per m² (tCO₂/m²)'].mean()
    carbon_intensity = total_co2_tonnes / (total_energy_kwh / 1000) if total_energy_kwh > 0 else 0
    total_potential_co2_saved = df['Potential CO₂ Saved (tCO₂)'].sum()
    
    st.markdown("**Solution:**")
    st.markdown(f"• Total CO₂ Emissions = {total_co2_tonnes:,.0f} tCO₂e")
    st.markdown(f"• Average CO₂ per m² = {avg_co2_per_m2:.3f} tCO₂e/m²")
    st.markdown(f"• Carbon Intensity = {carbon_intensity:.3f} tCO₂e/MWh")
    st.markdown(f"• Potential CO₂ Savings = {total_potential_co2_saved:,.0f} tCO₂e")
    
    explain_calculation(
        "Carbon Intensity Calculation",
        "Carbon Intensity = Total CO₂ ÷ Total Energy (MWh)",
        f"• Total CO₂ = {total_co2_tonnes:,.0f} tCO₂e\n• Total Energy = {total_energy_kwh/1000:,.0f} MWh",
        f"System-wide carbon performance",
        f"Carbon intensity shows emissions per unit of energy consumed."
    )
    
    # Problem 2: Carbon Reduction Targets
    st.subheader("Problem 2: Carbon Reduction Target Analysis")
    st.markdown("**Given:** Current emissions and NHS Net Zero targets")
    st.markdown("**Find:** Required reduction rates and target emissions")
    
    current_year = 2025
    target_year_80 = 2030
    target_year_100 = 2040
    
    years_to_80_target = target_year_80 - current_year
    years_to_100_target = target_year_100 - current_year
    
    target_emissions_80 = total_co2_tonnes * 0.2
    target_emissions_100 = 0
    
    annual_reduction_80 = (total_co2_tonnes - target_emissions_80) / years_to_80_target
    
    st.markdown("**Solution:**")
    st.markdown(f"• Current Emissions = {total_co2_tonnes:,.0f} tCO₂e")
    st.markdown(f"• 2030 Target (80% reduction) = {target_emissions_80:,.0f} tCO₂e")
    st.markdown(f"• 2040 Target (100% reduction) = {target_emissions_100:,.0f} tCO₂e")
    st.markdown(f"• Required Annual Reduction (to 2030) = {annual_reduction_80:,.0f} tCO₂e/year")
    
    explain_calculation(
        "Carbon Reduction Mathematics",
        "Annual Reduction = (Current - Target) ÷ Years",
        f"• Reduction needed = {total_co2_tonnes - target_emissions_80:,.0f} tCO₂e\n• Time available = {years_to_80_target} years",
        f"NHS Net Zero pathway requirements",
        f"The NHS needs to reduce emissions by {(annual_reduction_80/total_co2_tonnes)*100:.1f}% annually."
    )
    
    # Problem 3: Environmental Impact
    st.subheader("Problem 3: Environmental Impact Equivalencies")
    st.markdown("**Given:** CO₂ emissions totals")
    st.markdown("**Find:** Real-world equivalencies for context")
    
    cars_per_tonne_co2 = 0.22
    trees_per_tonne_co2 = 40
    homes_per_tonne_co2 = 0.2
    
    equivalent_cars = total_co2_tonnes * cars_per_tonne_co2
    equivalent_trees = total_co2_tonnes * trees_per_tonne_co2
    equivalent_homes = total_co2_tonnes * homes_per_tonne_co2
    
    st.markdown("**Solution:**")
    st.markdown(f"• Equivalent to {equivalent_cars:,.0f} cars driven for one year")
    st.markdown(f"• Would require {equivalent_trees:,.0f} mature trees to offset")
    st.markdown(f"• Equivalent to {equivalent_homes:,.0f} average homes' annual emissions")
    
    explain_calculation(
        "Environmental Context Calculations",
        "Equivalent Units = Total CO₂ × Conversion Factor",
        f"• Cars: {total_co2_tonnes:,.0f} tCO₂e × {cars_per_tonne_co2} = {equivalent_cars:,.0f} cars",
        f"Relatable environmental impact scale",
        "These equivalencies communicate the scale of NHS emissions."
    )
    
    # Create carbon distribution chart
    st.subheader("Carbon Distribution Visualization")
    fig = px.scatter(df, x='Energy per m² (kWh/m²)', y='CO₂ per m² (tCO₂/m²)', 
                    color='Trust Type', size='Gross internal floor area (m²)',
                    title='Energy vs Carbon Intensity by Trust Type')
    st.plotly_chart(fig, use_container_width=True)

def page_trust_analysis(df):
    st.title("🏥 Trust Performance Analysis")
    st.markdown("**Problem Set:** Individual trust performance and peer comparison")
    
    # Problem 1: Trust Ranking
    st.subheader("Problem 1: Trust Performance Ranking")
    st.markdown("**Given:** Energy efficiency metrics for all trusts")
    st.markdown("**Find:** Ranking and percentile analysis")
    
    df_ranked = df.copy()
    df_ranked['Energy Efficiency Rank'] = df_ranked['Energy per m² (kWh/m²)'].rank(method='min')
    df_ranked['Energy Efficiency Percentile'] = df_ranked['Energy per m² (kWh/m²)'].rank(pct=True) * 100
    
    top_5_efficient = df_ranked.nsmallest(5, 'Energy per m² (kWh/m²)')
    bottom_5_efficient = df_ranked.nlargest(5, 'Energy per m² (kWh/m²)')
    
    st.markdown("**Solution - Top 5 Most Efficient:**")
    for i, (_, row) in enumerate(top_5_efficient.iterrows(), 1):
        st.markdown(f"• {i}. {row['Trust Name']}: {row['Energy per m² (kWh/m²)']:.1f} kWh/m²")
    
    st.markdown("**Bottom 5 Performers:**")
    for i, (_, row) in enumerate(bottom_5_efficient.iterrows(), 1):
        st.markdown(f"• {i}. {row['Trust Name']}: {row['Energy per m² (kWh/m²)']:.1f} kWh/m²")
    
    explain_calculation(
        "Ranking Methodology",
        "Rank = Position when sorted by Energy per m² (ascending)",
        f"• Lower energy per m² = better efficiency",
        f"Peer comparison framework",
        "Rankings identify best and worst performers for targeted interventions."
    )
    
    # Problem 2: Trust Type Benchmarking
    st.subheader("Problem 2: Trust Type Benchmarking Analysis")
    st.markdown("**Given:** Trust performance within peer groups")
    st.markdown("**Find:** Deviation from peer benchmarks")
    
    if 'Trust Type' in df.columns:
        peer_stats = df.groupby('Trust Type').agg({
            'Energy per m² (kWh/m²)': ['count', 'mean', 'median', 'std'],
            'Potential Energy Saved (kWh)': 'sum',
            'Potential Cost Saved (£)': 'sum'
        }).round(2)
        
        st.markdown("**Solution - Peer Group Analysis:**")
        for trust_type in peer_stats.index:
            stats = peer_stats.loc[trust_type]
            count = stats[('Energy per m² (kWh/m²)', 'count')]
            mean_val = stats[('Energy per m² (kWh/m²)', 'mean')]
            median_val = stats[('Energy per m² (kWh/m²)', 'median')]
            std_val = stats[('Energy per m² (kWh/m²)', 'std')]
            
            st.markdown(f"• **{trust_type}** (n={count}): Mean={mean_val:.1f}, Median={median_val:.1f}, Std={std_val:.1f} kWh/m²")
        
        explain_calculation(
            "Peer Group Benchmarking",
            "Deviation = Individual Value - Peer Median",
            f"• Each trust compared to same trust type peers",
            f"Fair comparison methodology",
            "Comparing trusts to similar peer types is more fair."
        )
    
    # Problem 3: Improvement Potential
    st.subheader("Problem 3: Individual Trust Improvement Potential")
    st.markdown("**Given:** Current performance vs benchmarks")
    st.markdown("**Find:** Specific improvement opportunities")
    
    if len(df) > 0:
        sample_trust = df.iloc[0]
        
        trust_name = sample_trust['Trust Name']
        trust_type = sample_trust.get('Trust Type', 'N/A')
        current_energy = sample_trust['Energy per m² (kWh/m²)']
        target_energy = sample_trust['Target Energy per m²']
        potential_savings_kwh = sample_trust['Potential Energy Saved (kWh)']
        potential_savings_cost = sample_trust['Potential Cost Saved (£)']
        
        improvement_percent = ((current_energy - target_energy) / target_energy) * 100 if target_energy > 0 else 0
        
        st.markdown(f"**Sample Analysis - {trust_name}:**")
        st.markdown(f"• Trust Type: {trust_type}")
        st.markdown(f"• Current Performance: {current_energy:.1f} kWh/m²")
        st.markdown(f"• Peer Benchmark: {target_energy:.1f} kWh/m²")
        st.markdown(f"• Performance Gap: {improvement_percent:.1f}% above benchmark")
        st.markdown(f"• Potential Savings: {potential_savings_kwh:,.0f} kWh = £{potential_savings_cost:,.0f}")
        
        explain_calculation(
            "Individual Trust Analysis",
            "Improvement % = ((Current - Target) ÷ Target) × 100",
            f"• Current = {current_energy:.1f} kWh/m²\n• Target = {target_energy:.1f} kWh/m²",
            f"Specific improvement opportunity",
            f"This trust {'performs above' if improvement_percent > 0 else 'performs at or below'} its peer benchmark."
        )

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

        # Streamlit UI: Site Selection
        site_name = st.selectbox("Select NHS Site", df["Site Name"].unique())
        selected_row = df[df["Site Name"] == site_name].iloc[0]
        site_energy = selected_row["total_energy_kwh"]
        energy_per_m2 = selected_row["energy_per_m2"]
        score = selected_row["energy_score"]
        
        with st.spinner("Generating AI analysis..."):
            justification = infer_services_gemini(selected_row)

        # Display Results
        col1, col2, col3 = st.columns(3)
        col1.metric("Site", site_name)
        col2.metric("Total Energy (kWh)", f"{int(site_energy):,}")
        col3.metric("Energy Score (per m² in Trust Type)", f"{score} / 100")

        st.markdown("### Gemini-Inferred Service Summary")
        st.info(justification)
    else:
        st.warning("Please upload a valid NHS site data file to begin.")

# -------------------------------------
# Main Application
# -------------------------------------
def main():
    st.set_page_config(page_title="NHS Energy Analysis", layout="wide")
    

    st.markdown("""
<style>
/* MAIN BACKGROUND AND TEXT */
body, .main, .block-container {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* HEADINGS AND PARAGRAPH TEXT */
h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: #000000 !important;
}

/* INPUT FIELDS */
input, select, textarea {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
}

/* SELECTBOX / DROPDOWN (combobox) */
div[role="combobox"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 6px;
    padding: 0.25rem;
}

/* DROPDOWN OPTIONS */
ul[role="listbox"] {
    background-color: #FFFFFF !important;
    color: #FFFFFF !important;
    border: 1px solid #CCC !important;
}

/* DROPDOWN OPTION HOVER */
ul[role="listbox"] > li:hover {
    background-color: #f0f0f0 !important;
    color: #FFFFFF !important;
}

/* BUTTONS */
.stButton > button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
}
.stButton > button:hover {
    background-color: #f0f0f0 !important;
}

/* FILE UPLOADER DROPZONES */
section[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 2px dashed #000000 !important;
}

/* ALERT BOXES */
.stAlert {
    background-color: #f0f0f0 !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
}
</style>
""", unsafe_allow_html=True)


    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = {
        "Data Preprocessing": page_data_preprocessing,
        "Mathematical Overview": page_overview,
        "Energy Analysis": page_energy,
        "Financial Analysis": page_financial,
        "Carbon Analysis": page_carbon,
        "Trust Analysis": page_trust_analysis,
        "AI-Justified Analysis": page_ai_analysis
    }
    
    selected_page = st.sidebar.selectbox("Select Analysis Page", list(pages.keys()))
    
    # Check if data is processed for non-preprocessing pages
    if selected_page != "Data Preprocessing" and selected_page != "AI-Justified Analysis" and 'processed_data' not in st.session_state:
        st.warning("Please process data first using the Data Preprocessing page.")
        return
    
    # Run selected page
    if selected_page == "Data Preprocessing" or selected_page == "AI-Justified Analysis":
        pages[selected_page]()
    else:
        pages[selected_page](st.session_state.processed_data)

if __name__ == "__main__":
    main()