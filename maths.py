import streamlit as st  
import pandas as pd
import numpy as np
import plotly.express as px
import os
from io import BytesIO
import google.generativeai as genai
import json
import uuid
from fpdf import FPDF
import base64
import requests
from io import StringIO



GEMINI_API_KEY = "AIzaSyDUeI-vKkIO_HfMD1jVo_ccKhoiXDjf5V8"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error(" Gemini API key is missing. Please provide a valid API key.")
    st.stop()

@st.cache_data
def load_data(uploaded_file, encoding='utf-8-sig'):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.write(" Successfully loaded CSV file.")
            return df
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
            st.write(" Successfully loaded Parquet file.")
            return df
        else:
            st.error(" Unsupported file format. Please upload a CSV or Parquet file.")
            return None
    except Exception as e:
        st.error(f" Error loading file: {str(e)}")
        return None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def calculate_kpis(df):
    df_processed = df.copy()
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
    
    df_processed['Energy per Bed (kWh)'] = (df_processed['Total Energy (kWh)'] / df_processed['Bed Count']).replace([np.inf, -np.inf], 0).fillna(0)
    df_processed['CO₂ per Bed (tCO₂e)'] = (df_processed['Total CO₂ (tCO₂e)'] / df_processed['Bed Count']).replace([np.inf, -np.inf], 0).fillna(0)
    
    if 'Trust Type' in df_processed.columns:
        df_processed['Target Energy per m²'] = df_processed.groupby('Trust Type')['Energy per m² (kWh/m²)'].transform('median')
        df_processed['Target CO₂ per m²'] = df_processed.groupby('Trust Type')['CO₂ per m² (tCO₂/m²)'].transform('median')
        df_processed['Target cost_per_kWh'] = df_processed.groupby('Trust Type')['cost_per_kWh'].transform('median')
        df_processed['Target Energy per Bed'] = df_processed.groupby('Trust Type')['Energy per Bed (kWh)'].transform('median')
        df_processed['Target CO₂ per Bed'] = df_processed.groupby('Trust Type')['CO₂ per Bed (tCO₂e)'].transform('median')
    
    df_processed['Energy Deviation (kWh/m²)'] = df_processed['Energy per m² (kWh/m²)'] - df_processed['Target Energy per m²']
    df_processed['Energy Deviation (%)'] = (df_processed['Energy Deviation (kWh/m²)'] / df_processed['Target Energy per m²']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    
    df_processed['Potential Energy Saved (kWh)'] = (df_processed['Energy Deviation (kWh/m²)'] * df_processed['Gross internal floor area (m²)']).apply(lambda x: max(x, 0))
    df_processed['Potential Cost Saved (£)'] = df_processed['Potential Energy Saved (kWh)'] * df_processed['cost_per_kWh']
    df_processed['Potential CO₂ Saved (tCO₂)'] = ((df_processed['CO₂ per m² (tCO₂/m²)'] - df_processed['Target CO₂ per m²']) * df_processed['Gross internal floor area (m²)']).apply(lambda x: max(x, 0))
    
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
    st.info(f"""
    ** {title}**
    
    **Formula:** {formula}
    
    **Variables:**
    {variables}
    
    **Result:** {result}
    
    **Why this matters:** {reasoning}
    """)


@st.cache_data
def load_data_from_github(url):
    """Load CSV data from GitHub raw URL"""
    try:
        # Convert GitHub blob URL to raw URL
        if "github.com" in url and "/blob/" in url:
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        else:
            raw_url = url
            
        st.write(f"📡 Attempting to load data from: {raw_url}")
        
        # Make request to GitHub
        response = requests.get(raw_url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Read CSV from string content
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content, encoding='utf-8-sig')
        
        st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error loading data: {str(e)}")
        return None
    except pd.errors.EmptyDataError:
        st.error(" The CSV file appears to be empty")
        return None
    except pd.errors.ParserError as e:
        st.error(f" Error parsing CSV file: {str(e)}")
        return None
    except Exception as e:
        st.error(f" Unexpected error loading data: {str(e)}")
        return None

def page_data_preprocessing():
    st.title(" Data Preprocessing")
    st.markdown("""
    **Purpose:**  
    Load ERIC Site and Trust data from GitHub, merge them, aggregate site-level data to trust-level, and compute KPIs for energy analysis.

    **What You'll Do:**
    - Load `ERIC_Site.csv` and `ERIC_TRUST.csv` from GitHub.
    - Review preprocessing steps.
    - Download the processed dataset.
    """)

    st.info("""
    **Preprocessing Steps**
    1. Load remote CSVs via GitHub.
    2. Merge site and trust data using `Trust Code`.
    3. Aggregate energy, costs, and floor area.
    4. Compute per-unit metrics and KPIs.
    5. Handle missing values.
    """)

    # GitHub URLs (corrected to raw URLs)
    site_csv_url = "https://github.com/venkat2k25/calc/blob/main/ERIC_Site.csv"
    trust_csv_url = "https://github.com/venkat2k25/calc/blob/main/ERIC_TRUST.csv"

    # Load data from GitHub
    with st.spinner("Loading data from GitHub..."):
        site_data = load_data_from_github(site_csv_url)
        trust_data = load_data_from_github(trust_csv_url)

    if site_data is None or trust_data is None:
        st.error(" Failed to load one or both datasets from GitHub.")
        st.markdown("""
        **Troubleshooting Tips:**
        - Check if the GitHub repository is public
        - Verify the file paths are correct
        - Ensure you have internet connectivity
        - Try refreshing the page
        """)
        return

    # Display basic info about loaded data
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Site Data Rows", len(site_data))
        st.metric("Site Data Columns", len(site_data.columns))
    with col2:
        st.metric("Trust Data Rows", len(trust_data))
        st.metric("Trust Data Columns", len(trust_data.columns))

    try:
        with st.spinner("Processing data..."):
            energy_columns = [
                'Thermal energy consumption (KWh)', 'Electrical energy consumption (KWh)',
                'Electricity - green electricity consumed (kWh)', 'Electricity - trust owned solar consumed (kWh)',
                'Electricity - third party owned solar consumed (kWh)', 'Electricity - other renewables consumed (kWh)',
                'Gas consumed (kWh)', 'Oil consumed (kWh)', 'Non-fossil fuel - renewable consumed (kWh)',
                'Steam consumed (kWh)', 'Hot water consumed (kWh)', 'Solar electricity generated (kWh)'
            ]
            
            st.subheader("Step 1: Energy Aggregation")
            st.markdown("Aggregating all energy sources.")
            explain_calculation(
                "Energy Aggregation",
                "Total Energy = Σ(All Energy Sources)",
                f"• Sources: {len(energy_columns)} types",
                f"Sum of all energy consumption",
                "Captures comprehensive energy use."
            )
            
            cost_columns = [
                'Electricity - green electricity tariff costs (£)', 'Electricity - trust owned solar costs (£)',
                'Electricity - third party owned solar costs (£)', 'Electricity - other renewables costs (£)',
                'Electricity - other costs (£)', 'Gas costs (£)', 'Oil costs (£)',
                'Non-fossil fuel - renewable costs (£)', 'Other energy costs (£)'
            ]
            
            st.subheader("Step 2: Cost Aggregation")
            st.markdown("Aggregating all energy-related costs.")
            explain_calculation(
                "Cost Aggregation",
                "Total Costs = Σ(All Energy Costs)",
                f"• Cost categories: {len(cost_columns)}",
                f"Sum of all costs",
                "Ensures complete financial overview."
            )
            
            trust_co2_columns = [
                'Waste re-use scheme - Carbon savings (CO2e (tonnes))',
                'Carbon savings from investment in energy efficient schemes (CO2e (tonnes))'
            ]

            # Check available columns
            available_energy_columns = [col for col in energy_columns if col in site_data.columns]
            available_cost_columns = [col for col in cost_columns if col in site_data.columns]
            available_trust_co2_columns = [col for col in trust_co2_columns if col in trust_data.columns]

            # Display column availability
            st.write(f"Available energy columns: {len(available_energy_columns)}/{len(energy_columns)}")
            st.write(f"Available cost columns: {len(available_cost_columns)}/{len(cost_columns)}")
            st.write(f"Available CO2 columns: {len(available_trust_co2_columns)}/{len(trust_co2_columns)}")

            if not available_energy_columns or not available_cost_columns:
                st.error("Required energy or cost columns missing.")
                st.write("**Missing columns analysis:**")
                if not available_energy_columns:
                    st.write("- No energy columns found")
                if not available_cost_columns:
                    st.write("- No cost columns found")
                return
                
            if 'Trust Code' not in site_data.columns or 'Trust Code' not in trust_data.columns:
                st.error("Trust Code column missing.")
                st.write("**Available columns in site_data:**", list(site_data.columns)[:10])
                st.write("**Available columns in trust_data:**", list(trust_data.columns)[:10])
                return

            # Calculate totals
            site_data['Total Energy (kWh)'] = site_data[available_energy_columns].sum(axis=1, skipna=True)
            site_data['Total Costs (£)'] = site_data[available_cost_columns].sum(axis=1, skipna=True)

            agg_dict = {
                'Total Energy (kWh)': 'sum', 
                'Total Costs (£)': 'sum',
                'Gross internal floor area (m²)': 'sum',
                'Single bedrooms for patients with en-suite facilities (No.)': 'sum',
                'Single bedrooms for patients without en-suite facilities (No.)': 'sum',
                'Isolation rooms (No.)': 'sum', 
                'Trust Name': 'first'
            }
            
            st.subheader("Step 3: Trust-Level Aggregation")
            st.markdown("Summing metrics across all sites per trust.")
            explain_calculation(
                "Trust-Level Aggregation",
                "Trust Total = Σ(All Sites per Trust)",
                f"• Metrics: Energy, Costs, Floor Area, Beds",
                f"Combined trust metrics",
                "Reflects management unit for NHS energy decisions."
            )
            
            # Filter aggregation dictionary for available columns
            available_agg_dict = {k: v for k, v in agg_dict.items() if k in site_data.columns}
            site_data_agg = site_data.groupby('Trust Code').agg(available_agg_dict).reset_index()

            # Prepare trust merge columns
            trust_merge_cols = ['Trust Code', 'Trust Name']
            if 'Trust Type' in trust_data.columns:
                trust_merge_cols.append('Trust Type')
            trust_merge_cols.extend(available_trust_co2_columns)

            # Merge datasets
            merged_data = pd.merge(
                site_data_agg, 
                trust_data[trust_merge_cols], 
                on='Trust Code', 
                how='left', 
                suffixes=('_site', '_trust')
            )
            
            # Handle duplicate Trust Name columns
            if 'Trust Name_trust' in merged_data.columns and 'Trust Name_site' in merged_data.columns:
                merged_data['Trust Name'] = merged_data['Trust Name_trust'].combine_first(merged_data['Trust Name_site'])
                merged_data.drop(['Trust Name_site', 'Trust Name_trust'], axis=1, inplace=True, errors='ignore')

            # Calculate bed count
            bed_cols = [
                'Single bedrooms for patients with en-suite facilities (No.)',
                'Single bedrooms for patients without en-suite facilities (No.)',
                'Isolation rooms (No.)'
            ]
            available_bed_columns = [col for col in bed_cols if col in merged_data.columns]
            merged_data['Bed Count'] = merged_data[available_bed_columns].sum(axis=1, skipna=True)
            
            # Calculate CO2 if available
            if available_trust_co2_columns:
                merged_data['Total CO₂ (kgCO₂e)'] = merged_data[available_trust_co2_columns].sum(axis=1, skipna=True) * 1000
            else:
                merged_data['Total CO₂ (kgCO₂e)'] = 0

            # Calculate intensity metrics
            with np.errstate(divide='ignore', invalid='ignore'):
                merged_data['kWh_per_m2'] = merged_data['Total Energy (kWh)'] / merged_data['Gross internal floor area (m²)']
                merged_data['CO2_per_m2'] = merged_data['Total CO₂ (kgCO₂e)'] / merged_data['Gross internal floor area (m²)']
                merged_data['cost_per_kWh'] = merged_data['Total Costs (£)'] / merged_data['Total Energy (kWh)']

            st.subheader("Step 4: Intensity Metrics")
            st.markdown("Computing normalized metrics.")
            explain_calculation(
                "Intensity Metrics",
                "Energy per m² = Total Energy ÷ Floor Area",
                f"• Metrics: kWh/m², kgCO₂e/m², £/kWh",
                f"Normalized metrics",
                "Enables comparison across trusts."
            )

            # Clean infinite values
            merged_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Calculate KPIs
            merged_data = calculate_kpis(merged_data)
            st.session_state.processed_data = merged_data
            
            st.success("Data processed successfully!")
            
            # Show processing summary
            st.subheader("Processing Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trusts", len(merged_data))
            with col2:
                st.metric("Total Energy (GWh)", f"{merged_data['Total Energy (kWh)'].sum()/1e6:.1f}")
            with col3:
                st.metric("Total Costs (£M)", f"{merged_data['Total Costs (£)'].sum()/1e6:.1f}")
            
            st.subheader("Processed Data Preview")
            st.dataframe(merged_data.head(), use_container_width=True)

            st.subheader("Download Processed Data")
            csv = convert_df_to_csv(merged_data)
            st.download_button(
                label="Download Processed Data",
                data=csv,
                file_name="processed_nhs_data.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.exception(e)  # This will show the full traceback for debugging

def page_dashboard(df):
    st.title(" NHS Energy Dashboard")
    
    st.markdown("""
    Welcome to the **NHS Trust Energy Dashboard** — a unified view of energy efficiency, cost-saving opportunities, and carbon intensity across NHS Trusts.

    ---
    ###  Purpose:
    - Provide decision-makers with actionable insights.
    - Highlight performance gaps and areas of improvement.
    - Support the **Net Zero** transition through data transparency.

    ---
    ###  What's Inside:
    - Efficiency distribution (Pie)
    - Energy by Trust Type (Bar)
    - Cost savings potential (Bar)
    - Energy vs Carbon (Scatter)
    - Floor area & cost relationship (Bubble)
    - Energy-Carbon correlation (Heatmap)
    """)

    st.info("""
    This dashboard empowers **analysts**, **sustainability officers**, and **NHS strategists** to monitor, compare, and act on trust-level energy data.
    """)

    # Create columns for better layout
    col1, col2 = st.columns(2)

    # Pie chart: Efficiency distribution
    with col1:
        st.subheader(" Efficiency Label Distribution")
        efficiency_counts = df['Clustering Efficiency Label'].value_counts()
        
        fig_pie = px.pie(
            values=efficiency_counts.values,
            names=efficiency_counts.index,
            title="Trust Efficiency Distribution",
            color_discrete_map={
                'High-Risk': '#FF6B6B',
                'Moderate': '#FFD93D', 
                'Efficient': '#6BCB77'
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Summary metrics
    with col2:
        st.subheader("Key Metrics")
        total_trusts = len(df)
        efficient_trusts = len(df[df['Clustering Efficiency Label'] == 'Efficient'])
        high_risk_trusts = len(df[df['Clustering Efficiency Label'] == 'High-Risk'])
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Trusts", total_trusts)
            st.metric("Efficient Trusts", efficient_trusts, 
                     delta=f"{(efficient_trusts/total_trusts*100):.1f}%")
        with col_b:
            st.metric("High-Risk Trusts", high_risk_trusts,
                     delta=f"{(high_risk_trusts/total_trusts*100):.1f}%")
            st.metric("Total Potential Savings", f"£{df['Potential Cost Saved (£)'].sum():,.0f}")

    # Bar chart: Energy by Trust Type
    st.subheader(" Average Energy Usage by Trust Type")
    if 'Trust Type' in df.columns:
        energy_by_type = df.groupby('Trust Type')['Energy per m² (kWh/m²)'].agg(['mean', 'count']).round(1)
        energy_by_type.columns = ['Average Energy per m²', 'Number of Trusts']
        
        fig_energy = px.bar(
            x=energy_by_type.index,
            y=energy_by_type['Average Energy per m²'],
            title="Average Energy Consumption by Trust Type",
            labels={'x': 'Trust Type', 'y': 'kWh/m²'},
            color=energy_by_type['Average Energy per m²'],
            color_continuous_scale='Blues'
        )
        fig_energy.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_energy, use_container_width=True)
        
        # Show the data table
        st.dataframe(energy_by_type, use_container_width=True)

    # Bar chart: Savings by Trust Type
    st.subheader(" Potential Cost Savings by Trust Type")
    if 'Trust Type' in df.columns:
        savings_by_type = df.groupby('Trust Type')['Potential Cost Saved (£)'].sum().round(0)
        
        fig_savings = px.bar(
            x=savings_by_type.index,
            y=savings_by_type.values,
            title="Total Potential Cost Savings by Trust Type",
            labels={'x': 'Trust Type', 'y': 'Potential Savings (£)'},
            color=savings_by_type.values,
            color_continuous_scale='Oranges'
        )
        fig_savings.update_layout(height=400, showlegend=False)
        fig_savings.update_traces(texttemplate='£%{y:,.0f}', textposition='outside')
        st.plotly_chart(fig_savings, use_container_width=True)

    # Additional visualizations
    col3, col4 = st.columns(2)
    
    # Scatter plot: Energy vs CO2
    with col3:
        st.subheader(" Energy vs Carbon Emissions")
        fig_scatter = px.scatter(
            df,
            x='Energy per m² (kWh/m²)',
            y='CO₂ per m² (tCO₂/m²)',
            size='Gross internal floor area (m²)',
            color='Clustering Efficiency Label',
            hover_data=['Trust Name'],
            title="Energy vs Carbon Intensity",
            color_discrete_map={
                'High-Risk': '#FF6B6B',
                'Moderate': '#FFD93D', 
                'Efficient': '#6BCB77'
            }
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Box plot: Energy distribution by efficiency label
    with col4:
        st.subheader(" Energy Distribution by Efficiency")
        fig_box = px.box(
            df,
            x='Clustering Efficiency Label',
            y='Energy per m² (kWh/m²)',
            color='Clustering Efficiency Label',
            title="Energy Consumption Distribution",
            color_discrete_map={
                'High-Risk': '#FF6B6B',
                'Moderate': '#FFD93D', 
                'Efficient': '#6BCB77'
            }
        )
        fig_box.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Matrix")
    numeric_cols = [
        'Energy per m² (kWh/m²)', 
        'CO₂ per m² (tCO₂/m²)', 
        'cost_per_kWh',
        'Gross internal floor area (m²)', 
        'Bed Count'
    ]
    
    # Filter columns that exist in the dataframe
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) > 1:
        corr_matrix = df[available_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Key Metrics",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Top/Bottom performers
    st.subheader(" Performance Rankings")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.write("**Most Efficient Trusts (Lowest Energy per m²)**")
        top_efficient = df.nsmallest(5, 'Energy per m² (kWh/m²)')[['Trust Name', 'Energy per m² (kWh/m²)', 'Clustering Efficiency Label']]
        st.dataframe(top_efficient, use_container_width=True, hide_index=True)
    
    with col6:
        st.write("**Highest Energy Consumers (Highest Energy per m²)**")
        top_consumers = df.nlargest(5, 'Energy per m² (kWh/m²)')[['Trust Name', 'Energy per m² (kWh/m²)', 'Clustering Efficiency Label']]
        st.dataframe(top_consumers, use_container_width=True, hide_index=True)

    def generate_pdf_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)

        # Replace Unicode characters with ASCII equivalents
        content = """NHS Trust Energy Analysis Report
    1. Objective
    The analysis aims to evaluate NHS Trusts on their energy consumption, cost efficiency, and carbon performance. This supports identifying inefficiencies, benchmarking performance across peer groups, and providing actionable insights to support the NHS Net Zero target.

    2. Data Collection and Preprocessing
    Input Files
    ERIC_Site.csv: Contains site-level metrics such as energy consumed, floor area, and cost.

    ERIC_TRUST.csv: Contains trust-level metadata and carbon-saving initiatives.

    Preprocessing Workflow
    The page_data_preprocessing() function defines the preprocessing pipeline. Key steps include:

    Step 1: Energy Aggregation
    Formula:
    Total Energy (kWh) = Sum [energy consumption columns]

    Relevant columns:

    Thermal energy consumption (KWh)

    Electrical energy consumption (KWh)

    Gas, Oil, Steam, Hot water, and Renewable sources

    Step 2: Cost Aggregation
    Formula:
    Total Costs (£) = Sum [energy cost columns]

    Cost columns are aggregated to provide a total operational energy cost per trust.

    Step 3: Trust-Level Aggregation
    Formula:

    Trust Total Metric = Sum over all sites with same 'Trust Code'
    e.g., Total Energy per Trust = Sum Site Energy

    Step 4: Intensity Metrics
    Formulas:

    Energy per m² = Total Energy (kWh) / Gross Internal Floor Area (m²)  
    CO₂ per m² = Total CO₂ (kg) / Gross Internal Floor Area (m²)  
    Cost per kWh = Total Costs (£) / Total Energy (kWh)

    Step 5: CO₂ Conversion
    Formula:
    Total CO₂ (tCO₂e) = Total CO₂ (kgCO₂e) / 1000

    Bed count is estimated as:

    Bed Count = Sum [single ensuite + non-ensuite + isolation rooms]

    3. KPI Calculation and Benchmarking
    Implemented in calculate_kpis(df)

    Metrics Computed
    Energy per Bed (kWh)
    = Total Energy / Bed Count

    CO₂ per Bed (tCO₂e)
    = Total CO₂ / Bed Count

    Efficiency Targets (by Trust Type):
    Median values used as benchmarks:

    Target Energy per m² = Median of group('Trust Type')['Energy per m²']

    Deviation Metrics
    Energy Deviation (kWh/m²) = Energy per m² - Target Energy per m²  
    Energy Deviation (%) = (Deviation / Target) * 100  
    Potential Energy Saved = max(Deviation, 0) * Floor Area  
    Potential Cost Saved = Potential Energy Saved * cost_per_kWh  
    Potential CO₂ Saved = (CO₂ per m² - Target CO₂ per m²) * Floor Area

    Efficiency Labels
    Based on:

    Efficiency Ratio = Energy per m² / Target Energy per m²

    if ratio > 1.2 -> High-Risk  
    if ratio < 0.9 -> Efficient  
    else -> Moderate

    4. Dashboard Visualization
    Implemented in page_dashboard(df)

    Visual Components
    Pie chart: Distribution of efficiency labels.

    Bar charts:
    Avg. Energy per m² by Trust Type
    Total Potential Cost Savings by Trust Type

    Scatter plot:
    x = Energy per m², y = CO₂ per m², size = Floor Area

    Heatmap:
    Correlation matrix of numeric metrics

    5. Mathematical Overview
    Implemented in page_overview(df)

    Totals
    Total Energy = Sum Total Energy (kWh)  
    Total Cost = Sum Total Costs (£)  
    Total CO₂ = Sum Total CO₂ (tCO₂e)

    Unit Conversion
    GWh = kWh / 1,000,000  
    £M = £ / 1,000,000  
    ktCO₂e = tCO₂e / 1,000

    Efficiency Classification Count
    Based on previously calculated efficiency labels

    Trusts grouped by: Efficient, Moderate, High-Risk

    6. Energy Analysis
    Implemented in page_energy(df)

    Descriptive Statistics
    Mean Energy = Sum(Energy per m²) / n  
    Median Energy = Middle value of sorted energy values  
    Range = Max - Min  
    Std Dev = sqrt(Sum(x - mean)² / n)

    Trust Type Aggregation
    Group Mean = Sum(Group Energy per m²) / Group Count

    Visuals
    Histogram: Distribution with mean and median lines

    Bar chart: Top 10 energy-intensive trusts

    7. Financial Analysis
    Implemented in page_financial(df)

    Cost Metrics
    Simple Average Cost per kWh = mean(cost_per_kWh)  
    Weighted Avg = Total Cost / Total Energy  
    Total Potential Savings = Sum Potential Cost Saved (£)

    ROI Analysis
    Assumes:
    Investment Cost = Energy Saved × £2.5/kWh

    Payback Period = Total Investment / Annual Savings  
    ROI (%) = (Annual Savings / Investment) * 100

    Visuals
    Box plot: cost_per_kWh

    Bar chart: Top 10 most expensive trusts per kWh

    8. Carbon Analysis
    Implemented in page_carbon(df)

    Metrics
    Carbon Intensity = Total CO₂ / (Total Energy in MWh)  
    Average CO₂ per m² = Sum(CO₂ per m²) / n  
    Potential CO₂ Saved = max(CO₂ per m² - Target, 0) × Floor Area

    Net Zero Targets
    80% CO₂ reduction by 2030

    100% CO₂ reduction by 2040

    Annual Reduction Needed = (Current CO₂ - Target CO₂) / (Target Year - Current Year)

    Equivalencies
    Cars = Total CO₂ × 0.22  
    Trees = Total CO₂ × 40  
    Homes = Total CO₂ × 0.2

    9. Trust Performance Analysis
    Implemented in page_trust_analysis(df)

    Ranking
    Energy Efficiency Rank = Rank(Energy per m², ascending)  
    Percentile = (Rank / Total) × 100

    Peer Benchmarking
    Target per Trust Type = Median of group('Trust Type')['Energy per m²']
    Improvement Gap (%) = (Actual - Target) / Target × 100

    10. AI Justified Analysis
    Implemented in page_ai_analysis()

    Process
    Upload site-level dataset

    Compute:

    total_energy_kwh = Sum [energy sources]

    energy_per_m2 = total_energy_kwh / floor_area

    energy_score: Percentile score within peer group

    Prompt Gemini AI with infrastructure and energy data:

    Infers likely Service Type

    Justifies Energy Score with reasoning

    Summary Table
    Metric | Formula/Method
    Energy per m² | = Total Energy / Floor Area
    CO₂ per m² | = Total CO₂ / Floor Area
    Cost per kWh | = Total Cost / Total Energy
    Energy Deviation (%) | = (Actual - Target) / Target × 100
    Potential Energy Saved | = max(Actual - Target, 0) × Floor Area
    ROI (%) | = (Annual Savings / Investment Cost) × 100
    Carbon Intensity | = Total CO₂ / (Total Energy in MWh)
    Annual Reduction to Net Zero 2030 | = (Current Emissions - Target Emissions) / Years Remaining
    Efficiency Labeling | Based on ratio of Actual / Target energy use
    AI Energy Score | Percentile of energy_per_m2 in peer group, scaled to 0–100"""

        # Split content into lines and add to PDF
        for line in content.strip().split("\n"):
            # Clean the line of any remaining problematic characters
            clean_line = line.encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 10, clean_line.strip())

        return pdf.output(dest='S').encode('latin1')

    pdf_bytes = generate_pdf_report()

    st.download_button(
        label="Download NHS Energy Analysis Report (PDF)",
        data=pdf_bytes,
        file_name="NHS_Energy_Analysis_Report.pdf",
        mime="application/pdf"
    )

    # Scatter Plot: Energy vs CO₂
    st.subheader("Energy vs. Carbon Intensity (with Floor Area)")
    st.markdown("Bubble chart showing Trusts by their energy and carbon intensity, with bubble size indicating floor area.")
    fig1 = px.scatter(
        df, x='Energy per m² (kWh/m²)', y='CO₂ per m² (tCO₂/m²)',
        size='Gross internal floor area (m²)', color='Trust Type',
        hover_name='Trust Name' if 'Trust Name' in df.columns else None,
        title="Energy vs. CO₂ Intensity by Trust",
        labels={"Energy per m² (kWh/m²)": "Energy (kWh/m²)", "CO₂ per m² (tCO₂/m²)": "CO₂ (tCO₂/m²)"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Heatmap: Correlation Matrix
    st.subheader(" Energy & Carbon Correlation Heatmap")
    st.markdown("Explore how energy usage, cost, carbon emissions, and area correlate across Trusts.")
    numeric_cols = ['Energy per m² (kWh/m²)', 'CO₂ per m² (tCO₂/m²)', 'Potential Cost Saved (£)', 'Gross internal floor area (m²)']
    df_corr = df[numeric_cols].corr()
    fig2 = px.imshow(df_corr, text_auto=True, color_continuous_scale='Viridis', title="Feature Correlation Matrix")
    st.plotly_chart(fig2, use_container_width=True)

    # Line Chart: Energy Over Time (if applicable)
    if 'Reporting Period' in df.columns:
        st.subheader(" Energy Trend Over Time")
        df_time = df.groupby('Reporting Period')['Energy per m² (kWh/m²)'].mean().reset_index()
        fig3 = px.line(df_time, x='Reporting Period', y='Energy per m² (kWh/m²)', title="Average Energy Over Time")
        st.plotly_chart(fig3, use_container_width=True)


def page_overview(df):
    st.title("Mathematical Overview")
    st.markdown("""
    **Purpose:**  
    Summarizes NHS Trust energy performance with totals, unit conversions, efficiency classifications, and savings potential.
    """)
    
    st.subheader("Problem 1: Basic Calculations")
    st.markdown("Calculate total counts and sums.")
    
    total_trusts = df['Trust Name'].nunique()
    total_energy_kwh = df['Total Energy (kWh)'].sum()
    total_cost_pounds = df['Total Costs (£)'].sum()
    total_co2_tonnes = df['Total CO₂ (tCO₂e)'].sum()
    
    st.markdown(f"• Total Trusts: {total_trusts:,}")
    st.markdown(f"• Total Energy: {total_energy_kwh:,.0f} kWh")
    st.markdown(f"• Total Costs: £{total_cost_pounds:,.0f}")
    st.markdown(f"• Total CO₂ Emissions: {total_co2_tonnes:,.0f} tCO₂e")
    
    explain_calculation(
        "System-Wide Totals",
        "System Total = Σ(Individual Trust Values)",
        f"• Trusts: {total_trusts}\n• Energy, Costs, CO₂: Summed",
        f"NHS energy footprint",
        "Establishes baseline for NHS energy use."
    )
    
    st.subheader("Problem 2: Unit Conversions")
    st.markdown("Convert totals to GWh, £M, ktCO₂e.")
    
    total_energy_gwh = total_energy_kwh / 1e6
    total_cost_m = total_cost_pounds / 1e6
    total_co2_kt = total_co2_tonnes / 1e3
    
    st.markdown(f"• Energy: {total_energy_kwh:,.0f} kWh = {total_energy_gwh:,.1f} GWh")
    st.markdown(f"• Costs: £{total_cost_pounds:,.0f} = £{total_cost_m:,.1f}M")
    st.markdown(f"• CO₂ Emissions: {total_co2_tonnes:,.0f} tCO₂e = {total_co2_kt:,.1f} ktCO₂e")
    
    explain_calculation(
        "Unit Conversion Logic",
        "Large Unit = Small Unit ÷ Conversion Factor",
        f"• GWh = kWh ÷ 1,000,000\n• £M = £ ÷ 1,000,000\n• ktCO₂e = tCO₂e ÷ 1,000",
        f"Readable metrics",
        "Improves communication of scale."
    )
    
    st.subheader("Problem 3: Efficiency Classification")
    st.markdown("Count trusts by efficiency category.")
    
    efficiency_counts = df['Clustering Efficiency Label'].value_counts()
    
    for label, count in efficiency_counts.items():
        percentage = (count / total_trusts) * 100
        st.markdown(f"• {label}: {count} trusts ({percentage:.1f}%)")
    
    explain_calculation(
        "Efficiency Classification Method",
        "Efficiency Ratio = Actual Energy per m² ÷ Target Energy per m²",
        f"• Efficient: Ratio < 0.9\n• Moderate: 0.9 ≤ Ratio ≤ 1.2\n• High-Risk: Ratio > 1.2",
        f"Trust performance categories",
        "Identifies trusts needing efficiency improvements."
    )
    
    st.subheader("Problem 4: Savings Analysis by Trust Type")
    st.markdown("Calculate savings by trust type.")
    
    if 'Trust Type' in df.columns:
        savings_by_type = df.groupby('Trust Type')['Potential Cost Saved (£)'].sum().sort_values(ascending=False)
        
        for trust_type, savings in savings_by_type.items():
            st.markdown(f"• {trust_type}: £{savings:,.0f}")
        
        explain_calculation(
            "Savings Potential Calculation",
            "Potential Savings = (Actual - Target) × Floor Area × Cost per kWh",
            f"• Metrics: Energy, Floor Area, Cost",
            f"Financial impact",
            "Quantifies efficiency benefits."
        )



def page_energy(df):
    st.title("Energy Analysis")
    st.markdown("""
    **Purpose:**  
    Analyzes energy consumption patterns with central tendency, variability, and trust type analysis.
    """)

    st.subheader("Problem 1: Central Tendency Measures")
    st.markdown("Calculate mean, median, and potential savings.")

    energy_values = df['Energy per m² (kWh/m²)'].dropna()
    avg_e_per_m2 = energy_values.mean()
    median_e_per_m2 = energy_values.median()
    total_potential_e_saved = df['Potential Energy Saved (kWh)'].sum()

    st.markdown(f"• Mean Energy per m²: {avg_e_per_m2:,.1f} kWh/m²")
    st.markdown(f"• Median Energy per m²: {median_e_per_m2:,.1f} kWh/m²")
    st.markdown(f"• Total Potential Savings: {total_potential_e_saved:,.0f} kWh ({total_potential_e_saved/1e6:,.2f} GWh)")

    explain_calculation(
        "Central Tendency Analysis",
        "Mean = Σ(Values) ÷ n; Median = Middle Value",
        f"• Mean: {avg_e_per_m2:,.1f} kWh/m²\n• Median: {median_e_per_m2:,.1f} kWh/m²",
        f"Energy consumption summary",
        f"{'Right-skewed' if avg_e_per_m2 > median_e_per_m2 else 'Left-skewed'} distribution suggests {'high-energy trusts' if avg_e_per_m2 > median_e_per_m2 else 'uniform consumption'}."
    )

    st.subheader("Problem 2: Range Analysis")
    st.markdown("Determine minimum, maximum, and range.")

    min_energy = energy_values.min()
    max_energy = energy_values.max()
    range_energy = max_energy - min_energy
    std_energy = energy_values.std()

    st.markdown(f"• Minimum Energy per m²: {min_energy:,.1f} kWh/m²")
    st.markdown(f"• Maximum Energy per m²: {max_energy:,.1f} kWh/m²")
    st.markdown(f"• Range: {range_energy:,.1f} kWh/m²")
    st.markdown(f"• Standard Deviation: {std_energy:,.1f} kWh/m²")

    explain_calculation(
        "Variability Analysis",
        "Range = Max - Min; Std Dev = √(Σ(x - μ)²/n)",
        f"• Range: {range_energy:,.1f} kWh/m²\n• Std Dev: {std_energy:,.1f} kWh/m²",
        f"Energy spread measure",
        f"Large range ({range_energy:,.1f} kWh/m²) indicates optimization opportunities."
    )

    st.subheader("Problem 3: Group Statistics")
    st.markdown("Calculate average energy by trust type.")

    if 'Trust Type' in df.columns:
        energy_by_type = df.groupby('Trust Type')['Energy per m² (kWh/m²)'].agg(['mean', 'count', 'std']).round(1)

        for trust_type, stats in energy_by_type.iterrows():
            st.markdown(f"• {trust_type}: {stats['mean']:,.1f} kWh/m² (n={stats['count']}, σ={stats['std']:,.1f})")

        explain_calculation(
            "Trust Type Analysis",
            "Group Mean = Σ(Group Values) ÷ Group Count",
            f"• Trust types have unique needs",
            f"Benchmarking by type",
            "Accounts for operational differences."
        )

    st.subheader("Energy Distribution Visualization")
    st.markdown("""
    **Description:**  
    Histogram of energy per m² with mean and median lines.
    """)
    fig = px.histogram(df, x='Energy per m² (kWh/m²)', nbins=30, 
                       title='Distribution of Energy Consumption per m²')
    fig.add_vline(x=avg_e_per_m2, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {avg_e_per_m2:.1f}")
    fig.add_vline(x=median_e_per_m2, line_dash="dash", line_color="green", 
                  annotation_text=f"Median: {median_e_per_m2:.1f}")
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Top Energy-Consuming Trusts")
    st.markdown("""
    **Description:**  
    This chart displays the top 10 Trusts with the highest energy consumption per square meter.  
    These may indicate either inefficient infrastructure or high operational demand.  
    Reviewing these Trusts helps prioritize intervention or further investigation.
    """)

    if 'Trust Name' in df.columns:
        top_energy_trusts = df[['Trust Name', 'Energy per m² (kWh/m²)']].dropna().sort_values(
            by='Energy per m² (kWh/m²)', ascending=False).head(10)

        fig_top = px.bar(top_energy_trusts,
                         x='Energy per m² (kWh/m²)',
                         y='Trust Name',
                         orientation='h',
                         title="Top 10 Trusts by Energy Use per m²",
                         labels={'Energy per m² (kWh/m²)': 'Energy per m²', 'Trust Name': 'Trust'},
                         height=500)
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)

        # Show the trust with the highest energy consumption
        highest_trust = top_energy_trusts.iloc[0]
        st.markdown(f"**Highest Energy Consuming Trust:** {highest_trust['Trust Name']} — {highest_trust['Energy per m² (kWh/m²)']:.1f} kWh/m²")

    else:
        st.warning("Column 'Trust Name' not found in dataset. Cannot display top consuming Trusts.")


def page_financial(df):
    st.title(" Financial Analysis")
    st.markdown("""
    **Purpose:**  
    Evaluates energy costs and ROI for efficiency improvements.
    """)

    st.subheader("Problem 1: Cost per Unit Analysis")
    st.markdown("Calculate average cost per kWh and totals.")

    total_cost_pounds = df['Total Costs (£)'].sum()
    total_energy_kwh = df['Total Energy (kWh)'].sum()
    avg_cost_kwh = df['cost_per_kWh'].mean()
    weighted_avg_cost_kwh = total_cost_pounds / total_energy_kwh if total_energy_kwh > 0 else 0
    total_potential_savings = df['Potential Cost Saved (£)'].sum()

    st.markdown(f"• Total Annual Costs: £{total_cost_pounds:,.0f}")
    st.markdown(f"• Simple Average Cost per kWh: £{avg_cost_kwh:.3f}")
    st.markdown(f"• Weighted Average Cost per kWh: £{weighted_avg_cost_kwh:.3f}")
    st.markdown(f"• Total Potential Savings: £{total_potential_savings:,.0f}")

    explain_calculation(
        "Cost Analysis Methods",
        "Simple Avg = Σ(Individual Costs/kWh) ÷ n; Weighted Avg = Total £ ÷ Total kWh",
        f"• Simple Average: £{avg_cost_kwh:.3f}\n• Weighted Average: £{weighted_avg_cost_kwh:.3f}",
        f"Cost efficiency perspectives",
        "Weighted average reflects consumption volume."
    )

    st.subheader("Problem 2: Cost Distribution Analysis")
    st.markdown("Analyze cost per kWh distribution.")

    cost_values = df['cost_per_kWh'].dropna()
    min_cost = cost_values.min()
    max_cost = cost_values.max()
    median_cost = cost_values.median()
    std_cost = cost_values.std()
    q1_cost = cost_values.quantile(0.25)
    q3_cost = cost_values.quantile(0.75)
    iqr_cost = q3_cost - q1_cost

    st.markdown(f"• Minimum Cost: £{min_cost:.3f} per kWh")
    st.markdown(f"• Maximum Cost: £{max_cost:.3f} per kWh")
    st.markdown(f"• Median Cost: £{median_cost:.3f} per kWh")
    st.markdown(f"• Standard Deviation: £{std_cost:.3f}")
    st.markdown(f"• Interquartile Range: £{iqr_cost:.3f}")

    explain_calculation(
        "Cost Variability Insights",
        "IQR = Q3 - Q1",
        f"• Range: £{max_cost - min_cost:.3f}\n• IQR: £{iqr_cost:.3f}",
        f"Cost spread measure",
        "Wide range suggests contract variability."
    )

    st.subheader("Problem 3: Return on Investment Analysis")
    st.markdown("Calculate ROI for efficiency investments.")

    typical_investment_per_kwh_saved = 2.5
    investment_cost = df['Potential Energy Saved (kWh)'] * typical_investment_per_kwh_saved
    annual_savings = df['Potential Cost Saved (£)']

    total_investment = investment_cost.sum()
    total_annual_savings = annual_savings.sum()
    simple_payback_years = total_investment / total_annual_savings if total_annual_savings > 0 else float('inf')
    roi_percent = (total_annual_savings / total_investment) * 100 if total_investment > 0 else 0

    st.markdown(f"• Total Investment Required: £{total_investment:,.0f}")
    st.markdown(f"• Annual Savings Potential: £{total_annual_savings:,.0f}")
    st.markdown(f"• Simple Payback Period: {simple_payback_years:.1f} years")
    st.markdown(f"• Annual ROI: {roi_percent:.1f}%")

    explain_calculation(
        "ROI Calculation Method",
        "Payback Period = Investment Cost ÷ Annual Savings; ROI = (Annual Savings ÷ Investment) × 100",
        f"• Investment: {df['Potential Energy Saved (kWh)'].sum():,.0f} kWh × £{typical_investment_per_kwh_saved}",
        f"Financial viability",
        f"Payback period of {simple_payback_years:.1f} years is {'very attractive' if simple_payback_years < 3 else 'reasonable'}."
    )

    st.subheader("Cost Distribution Visualization")
    st.markdown("""
    **Description:**  
    Box plot of cost per kWh with average and median lines.
    """)
    fig = px.box(df, y='cost_per_kWh', title='Distribution of Energy Costs per kWh Across Trusts')
    fig.add_hline(y=avg_cost_kwh, line_dash="dash", line_color="red", 
                  annotation_text=f"Average: £{avg_cost_kwh:.3f}")
    fig.add_hline(y=median_cost, line_dash="dash", line_color="green", 
                  annotation_text=f"Median: £{median_cost:.3f}")
    st.plotly_chart(fig, use_container_width=True)

 
    st.subheader("Top Most Expensive Trusts per kWh")
    st.markdown("""
    **Description:**  
    The following chart shows the top 10 Trusts with the highest average energy cost per unit (kWh).  
    These Trusts may be on expensive contracts or experience supply inefficiencies.  
    Reviewing these outliers is essential for renegotiation or contract analysis.
    """)

    if 'Trust Name' in df.columns:
        top_cost_trusts = df[['Trust Name', 'cost_per_kWh']].dropna().sort_values(
            by='cost_per_kWh', ascending=False).head(10)

        fig_top_cost = px.bar(top_cost_trusts,
                              x='cost_per_kWh',
                              y='Trust Name',
                              orientation='h',
                              title="Top 10 Most Expensive Trusts by Unit Cost",
                              labels={'cost_per_kWh': 'Cost per kWh (£)', 'Trust Name': 'Trust'},
                              height=500)
        fig_top_cost.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_cost, use_container_width=True)

        highest_cost_trust = top_cost_trusts.iloc[0]
        st.markdown(f"**Highest Cost Trust:** {highest_cost_trust['Trust Name']} — £{highest_cost_trust['cost_per_kWh']:.3f} per kWh")
    else:
        st.warning("Column 'Trust Name' not found in dataset. Cannot show top costly Trusts.")


def page_carbon(df):
    st.title(" Carbon Analysis")
    st.markdown("""
    **Purpose:**  
    Assesses carbon emissions and NHS Net Zero targets (80% reduction by 2030, 100% by 2040).
    """)

    st.subheader("Problem 1: Carbon Intensity Analysis")
    st.markdown("Calculate carbon intensity and totals.")

    total_co2_tonnes = df['Total CO₂ (tCO₂e)'].sum()
    total_energy_kwh = df['Total Energy (kWh)'].sum()
    avg_co2_per_m2 = df['CO₂ per m² (tCO₂/m²)'].mean()
    carbon_intensity = total_co2_tonnes / (total_energy_kwh / 1000) if total_energy_kwh > 0 else 0
    total_potential_co2_saved = df['Potential CO₂ Saved (tCO₂)'].sum()

    st.markdown(f"• Total CO₂ Emissions: {total_co2_tonnes:,.0f} tCO₂e")
    st.markdown(f"• Average CO₂ per m²: {avg_co2_per_m2:.3f} tCO₂e/m²")
    st.markdown(f"• Carbon Intensity: {carbon_intensity:.3f} tCO₂e/MWh")
    st.markdown(f"• Potential CO₂ Savings: {total_potential_co2_saved:,.0f} tCO₂e")

    explain_calculation(
        "Carbon Intensity Calculation",
        "Carbon Intensity = Total CO₂ ÷ Total Energy (MWh)",
        f"• Total CO₂: {total_co2_tonnes:,.0f} tCO₂e\n• Total Energy: {total_energy_kwh/1000:,.0f} MWh",
        f"Carbon performance",
        "Highlights environmental efficiency."
    )

    st.subheader("Problem 2: Carbon Reduction Target Analysis")
    st.markdown("Calculate reduction rates for Net Zero targets.")

    current_year = 2025
    target_year_80 = 2030
    target_year_100 = 2040

    years_to_80_target = target_year_80 - current_year
    target_emissions_80 = total_co2_tonnes * 0.2
    target_emissions_100 = 0

    annual_reduction_80 = (total_co2_tonnes - target_emissions_80) / years_to_80_target

    st.markdown(f"• Current Emissions: {total_co2_tonnes:,.0f} tCO₂e")
    st.markdown(f"• 2030 Target (80% reduction): {target_emissions_80:,.0f} tCO₂e")
    st.markdown(f"• 2040 Target (100% reduction): {target_emissions_100:,.0f} tCO₂e")
    st.markdown(f"• Required Annual Reduction (to 2030): {annual_reduction_80:,.0f} tCO₂e/year")

    explain_calculation(
        "Carbon Reduction Mathematics",
        "Annual Reduction = (Current - Target) ÷ Years",
        f"• Reduction Needed: {total_co2_tonnes - target_emissions_80:,.0f} tCO₂e\n• Time: {years_to_80_target} years",
        f"Net Zero requirements",
        f"Requires {(annual_reduction_80/total_co2_tonnes)*100:.1f}% annual reduction."
    )

    st.subheader("Problem 3: Environmental Impact Equivalencies")
    st.markdown("Translate CO₂ emissions into equivalencies.")

    cars_per_tonne_co2 = 0.22
    trees_per_tonne_co2 = 40
    homes_per_tonne_co2 = 0.2

    equivalent_cars = total_co2_tonnes * cars_per_tonne_co2
    equivalent_trees = total_co2_tonnes * trees_per_tonne_co2
    equivalent_homes = total_co2_tonnes * homes_per_tonne_co2

    st.markdown(f"• Equivalent Cars: {equivalent_cars:,.0f} cars driven for one year")
    st.markdown(f"• Equivalent Trees: {equivalent_trees:,.0f} trees to offset")
    st.markdown(f"• Equivalent Homes: {equivalent_homes:,.0f} homes' annual emissions")

    explain_calculation(
        "Environmental Equivalencies",
        "Equivalent Units = Total CO₂ × Conversion Factor",
        f"• Cars: {total_co2_tonnes:,.0f} tCO₂e × {cars_per_tonne_co2}\n• Trees: {total_co2_tonnes:,.0f} tCO₂e × {trees_per_tonne_co2}",
        f"Relatable impact",
        "Communicates emission scale."
    )

    st.subheader("Carbon Distribution Visualization")
    st.markdown("""
    **Description:**  
    Scatter plot of energy vs. carbon intensity.
    """)
    fig = px.scatter(df, x='Energy per m² (kWh/m²)', y='CO₂ per m² (tCO₂/m²)', 
                     color='Trust Type', size='Gross internal floor area (m²)',
                     title='Energy vs Carbon Intensity by Trust Type')
    st.plotly_chart(fig, use_container_width=True)

    #  NEW VISUALIZATION: Top Carbon Emitters
    st.subheader("Top Carbon Emitters per m²")
    st.markdown("""
    **Description:**  
    This chart displays the 10 Trusts with the highest carbon emissions per square meter.  
    These are potential priority areas for sustainability intervention.  
    Helps NHS track aggressive decarbonization needs.
    """)

    if 'Trust Name' in df.columns:
        top_emitters = df[['Trust Name', 'CO₂ per m² (tCO₂/m²)']].dropna().sort_values(
            by='CO₂ per m² (tCO₂/m²)', ascending=False).head(10)

        fig_top = px.bar(top_emitters,
                         x='CO₂ per m² (tCO₂/m²)',
                         y='Trust Name',
                         orientation='h',
                         title="Top 10 Carbon Intensive Trusts (per m²)",
                         labels={'CO₂ per m² (tCO₂/m²)': 'CO₂ per m²', 'Trust Name': 'Trust'},
                         height=500)
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)

        highest_emitter = top_emitters.iloc[0]
        st.markdown(f"**Worst Emitter:** {highest_emitter['Trust Name']} — {highest_emitter['CO₂ per m² (tCO₂/m²)']:.3f} tCO₂e/m²")
    else:
        st.warning("Column 'Trust Name' not found. Cannot show top carbon emitters.")

def page_trust_analysis(df):
    st.title(" Trust Performance Analysis Dashboard")
    st.markdown("""
    This section provides a deep dive into the **energy efficiency performance of NHS Trusts**.  
    It includes **rank-based benchmarking**, **peer-group comparisons**, and a **granular analysis of individual improvement potential**.
    """)

    # -----------------------------------
    st.subheader(" Problem 1: Trust Performance Ranking")
    st.markdown("""
    Objective: **Rank NHS Trusts based on energy efficiency (kWh/m²)**  
    Lower values indicate better performance.
    """)

    # Compute rankings
    df_ranked = df.copy()
    df_ranked['Energy Efficiency Rank'] = df_ranked['Energy per m² (kWh/m²)'].rank(method='min')
    df_ranked['Energy Efficiency Percentile'] = df_ranked['Energy per m² (kWh/m²)'].rank(pct=True) * 100

    # Top and bottom performers
    top_5_efficient = df_ranked.nsmallest(5, 'Energy per m² (kWh/m²)')
    bottom_5_efficient = df_ranked.nlargest(5, 'Energy per m² (kWh/m²)')

    # Display top 5
    st.markdown("####  Top 5 Most Energy Efficient Trusts")
    for i, (_, row) in enumerate(top_5_efficient.iterrows(), 1):
        st.markdown(f"- **{i}. {row['Trust Name']}**: {row['Energy per m² (kWh/m²)']:.1f} kWh/m² (Percentile: {row['Energy Efficiency Percentile']:.1f}%)")

    # Display bottom 5
    st.markdown("####  Bottom 5 Least Efficient Trusts")
    for i, (_, row) in enumerate(bottom_5_efficient.iterrows(), 1):
        st.markdown(f"- **{i}. {row['Trust Name']}**: {row['Energy per m² (kWh/m²)']:.1f} kWh/m² (Percentile: {row['Energy Efficiency Percentile']:.1f}%)")

    explain_calculation(
        "🔍 Ranking Methodology",
        "Trusts are ranked by their 'Energy per m²' metric in ascending order.",
        "• Percentile = (Rank ÷ Total Trusts) × 100",
        "• Higher percentile → lower efficiency",
        "This enables identification of both high performers and outliers needing attention."
    )

    # -----------------------------------
    st.subheader(" Problem 2: Trust Type Benchmarking")
    st.markdown("""
    Objective: **Compare Trusts within their respective peer groups**, based on their `Trust Type`  
    This provides a fair benchmark by accounting for structural or operational differences.
    """)

    if 'Trust Type' in df.columns:
        peer_stats = df.groupby('Trust Type').agg({
            'Energy per m² (kWh/m²)': ['count', 'mean', 'median', 'std'],
            'Potential Energy Saved (kWh)': 'sum',
            'Potential Cost Saved (£)': 'sum'
        }).round(2)

        st.markdown("####  Peer Group Summary")
        for trust_type in peer_stats.index:
            stats = peer_stats.loc[trust_type]
            count = stats[('Energy per m² (kWh/m²)', 'count')]
            mean_val = stats[('Energy per m² (kWh/m²)', 'mean')]
            median_val = stats[('Energy per m² (kWh/m²)', 'median')]
            std_val = stats[('Energy per m² (kWh/m²)', 'std')]
            energy_saved = stats[('Potential Energy Saved (kWh)', 'sum')]
            cost_saved = stats[('Potential Cost Saved (£)', 'sum')]

            st.markdown(f"""
            • **{trust_type}** *(n = {count})*  
              - Mean Energy Usage: **{mean_val:.1f}** kWh/m²  
              - Median: **{median_val:.1f}**, Std Dev: **{std_val:.1f}** kWh/m²  
              - Potential Group Savings: **{energy_saved:,.0f} kWh** (£{cost_saved:,.0f})
            """)

        explain_calculation(
            " Peer Group Benchmarking",
            "Each Trust is benchmarked against the *median performance* of its own type (e.g., Acute, Community).",
            "• Deviation = Trust Energy Value - Peer Group Median",
            "• This allows apples-to-apples comparisons",
            "Ideal for identifying systemic vs operational inefficiencies."
        )

    # -----------------------------------
    st.subheader(" Problem 3: Individual Trust Improvement Potential")
    st.markdown("""
    Objective: **Estimate energy saving potential for a selected Trust**, based on peer benchmarks.
    """)

    if len(df) > 0:
        sample_trust = df.iloc[0]

        trust_name = sample_trust['Trust Name']
        trust_type = sample_trust.get('Trust Type', 'N/A')
        current_energy = sample_trust['Energy per m² (kWh/m²)']
        target_energy = sample_trust['Target Energy per m²']
        potential_savings_kwh = sample_trust['Potential Energy Saved (kWh)']
        potential_savings_cost = sample_trust['Potential Cost Saved (£)']

        improvement_percent = ((current_energy - target_energy) / target_energy) * 100 if target_energy > 0 else 0

        st.markdown(f"####  Sample Trust: **{trust_name}**")
        st.markdown(f"""
        - Trust Type: **{trust_type}**  
        - Current Energy Usage: **{current_energy:.1f} kWh/m²**  
        - Benchmark Target: **{target_energy:.1f} kWh/m²**  
        - Gap to Benchmark: **{improvement_percent:.1f}%** {'above' if improvement_percent > 0 else 'below'} target  
        - Potential Annual Savings: **{potential_savings_kwh:,.0f} kWh** (£{potential_savings_cost:,.0f})
        """)

        explain_calculation(
            " Individual Trust Improvement",
            "Improvement % = ((Current - Target) ÷ Target) × 100",
            f"• Current: {current_energy:.1f} kWh/m²\n• Target: {target_energy:.1f} kWh/m²",
            "Evaluates improvement potential using peer-based benchmarking",
            "Trusts significantly above target should prioritize energy interventions."
        )

def page_ai_analysis():
    st.title(" AI-Justified Site Efficiency Analysis")
    st.markdown("""
    This tool uses **Gemini AI** to:
    1. Infer the likely **service type** for a given NHS site.
    2. Provide a clear **justification** for its energy efficiency score based on infrastructure data.
    """)

    uploaded_file = st.file_uploader("Upload NHS site data (.csv or .parquet)", type=["csv", "parquet"], key="ai_file_uploader")

    if not uploaded_file:
        st.warning("Please upload a valid NHS site data file.")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    df.fillna(0, inplace=True)

    # ---------- ENERGY METRICS ----------
    def compute_total_energy(row):
        return sum([
            row.get("Thermal energy consumption (KWh)", 0),
            row.get("Electrical energy consumption (KWh)", 0),
            row.get("Gas consumed (kWh)", 0),
            row.get("Oil consumed (kWh)", 0),
            row.get("Steam consumed (kWh)", 0),
            row.get("Hot water consumed (kWh)", 0),
        ])

    df["total_energy_kwh"] = df.apply(compute_total_energy, axis=1)
    df["energy_per_m2"] = df["total_energy_kwh"] / (df["Gross internal floor area (m²)"] + 1e-6)

    def compute_energy_score(row, reference_df):
        trust_type = row.get("Trust Type", "Unknown")
        peer_group = reference_df[reference_df["Trust Type"] == trust_type]["energy_per_m2"]
        if len(peer_group) > 1:
            rank = np.sum(peer_group <= row["energy_per_m2"]) / len(peer_group)
            return min(int(np.ceil(rank * 100 / 5)) * 5, 100)
        return 50

    df["energy_score"] = df.apply(lambda row: compute_energy_score(row, df), axis=1)

    # ---------- AI PROMPT ----------
    def generate_ai_prompt(row):
        return f"""
You are an expert in NHS site infrastructure and energy efficiency.

**Site Name**: {row.get('Site Name', 'N/A')}

## Energy Metrics
- Energy per m²: {row.get('energy_per_m2', 0):.2f} kWh/m²
- Energy Score: {row.get('energy_score', 0)} / 100 (0 = efficient, 100 = inefficient)

## Key Infrastructure
- Pathology: {row.get('Pathology (m²)', 0)} m²
- CSSD: {row.get('Clinical Sterile Services Dept. (CSSD) (m²)', 0)} m²
- Isolation Rooms: {row.get('Isolation rooms (No.)', 0)}
- Ensuite Beds: {row.get('Single bedrooms for patients with en-suite facilities (No.)', 0)}
- Contact Centre: {row.get('999 Contact Centre (m²)', 0)} m²
- Hub (Ready Station): {row.get('Hub (make ready station) (m²)', 0)} m²
- Ambulance Station: {row.get('Ambulance Station (m²)', 0)} m²
- Staff Accommodation: {row.get('Staff Accommodation (m²)', 0)} m²
- Medical Records Area: {row.get('Medical records (m²)', 0)} m²
- Restaurants/Cafés: {row.get("Restaurants and cafés (m²)", 0)} m²

Reported Service Type: {row.get('Service types', 'N/A')}

### Your Task:
1. Infer the **primary service type** (e.g., Acute Hospital, Mental Health Unit).
2. Justify the **energy score** based on infrastructure and usage patterns.

**Format:**
Inferred Service Type: [Type]  
Justification: [Explain clearly]
        """

    def infer_services_gemini(row):
        try:
            prompt = generate_ai_prompt(row)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error during AI analysis:\n{str(e)}"

    # ---------- SITE SELECTION UI ----------
    st.subheader(" Select a Site for AI Analysis")
    site_name = st.selectbox("Select NHS Site", df["Site Name"].unique())
    selected_row = df[df["Site Name"] == site_name].iloc[0]

    # ---------- METRICS DISPLAY ----------
    st.subheader("Site Energy Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Site Name", site_name)
    col2.metric("Total Energy (kWh)", f"{int(selected_row['total_energy_kwh']):,}")
    col3.metric("Energy Score", f"{selected_row['energy_score']} / 100")

    # ---------- GEMINI AI OUTPUT ----------
    st.subheader("🔎 Gemini AI Analysis")
    st.markdown("AI-generated inference and justification based on site infrastructure and energy use.")
    with st.spinner("Asking Gemini AI..."):
        analysis = infer_services_gemini(selected_row)

    st.info(analysis)


def main():
    st.set_page_config(
        page_title="NHS Energy Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
<style>

/* Global background and font */
body, .main, .block-container {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    font-family: 'Segoe UI', Arial, sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F8F9FA !important;
    color: #000000 !important;
    border-right: 1px solid #E0E0E0 !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #000000 !important;
}

/* Headings and text */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #000000 !important;
}

/* Input widgets */
input, select, textarea {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 5px !important;
}

/* Selectbox / Dropdown Fix */
div[role="combobox"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 6px !important;
}

/* Main dropdown container */
div[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 6px !important;
}

/* Selected value text */
div[data-testid="stSelectbox"] div[role="combobox"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
}

/* Dropdown arrow and container */
div[data-testid="stSelectbox"] svg {
    color: #000000 !important;
}

/* Dropdown list container */
ul[role="listbox"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

/* Individual dropdown options */
ul[role="listbox"] > li {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    padding: 8px 12px !important;
}

/* Hover state for dropdown options */
ul[role="listbox"] > li:hover {
    background-color: #F0F0F0 !important;
    color: #000000 !important;
}

/* Selected/active option */
ul[role="listbox"] > li[aria-selected="true"] {
    background-color: #E6F3FF !important;
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Focus state for dropdown */
div[data-testid="stSelectbox"] div[role="combobox"]:focus {
    border-color: #4A90E2 !important;
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2) !important;
}

/* Primary Buttons - Blue Theme */
.stButton > button {
    background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #357ABD 0%, #2C5F8A 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4) !important;
    color: #FFFFFF !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 6px rgba(74, 144, 226, 0.3) !important;
}

/* Download Buttons - Green Theme */
.stDownloadButton > button {
    background: linear-gradient(135deg, #28A745 0%, #1E7E34 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1E7E34 0%, #155724 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4) !important;
    color: #FFFFFF !important;
}

.stDownloadButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 6px rgba(40, 167, 69, 0.3) !important;
}

/* Secondary Buttons - Gray Theme */
button[kind="secondary"] {
    background: linear-gradient(135deg, #6C757D 0%, #545B62 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 6px rgba(108, 117, 125, 0.3) !important;
}

button[kind="secondary"]:hover {
    background: linear-gradient(135deg, #545B62 0%, #3A3F44 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(108, 117, 125, 0.4) !important;
}

/* Warning/Alert Buttons - Orange Theme */
button[data-testid*="warning"], 
.stButton[data-testid*="warning"] > button {
    background: linear-gradient(135deg, #FF8C00 0%, #E67E00 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(255, 140, 0, 0.3) !important;
}

button[data-testid*="warning"]:hover,
.stButton[data-testid*="warning"] > button:hover {
    background: linear-gradient(135deg, #E67E00 0%, #CC6F00 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(255, 140, 0, 0.4) !important;
}

/* Error/Danger Buttons - Red Theme */
button[data-testid*="error"], 
.stButton[data-testid*="error"] > button {
    background: linear-gradient(135deg, #DC3545 0%, #B02A37 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3) !important;
}

button[data-testid*="error"]:hover,
.stButton[data-testid*="error"] > button:hover {
    background: linear-gradient(135deg, #B02A37 0%, #8B1E2B 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(220, 53, 69, 0.4) !important;
}

/* Custom Upload Button Styling */
section[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #6F42C1 0%, #5A32A3 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 6px rgba(111, 66, 193, 0.3) !important;
}

section[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #5A32A3 0%, #4A2B87 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(111, 66, 193, 0.4) !important;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 2px dashed #4A90E2 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #357ABD !important;
    background-color: #F8FBFF !important;
}

/* Alert boxes */
.stAlert {
    background-color: #F8F9FA !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1) !important;
}
.stAlert[data-testid="stInfo"] {
    background-color: #E6F3FF !important;
    border-color: #4A90E2 !important;
}
.stAlert[data-testid="stSuccess"] {
    background-color: #E6FFEC !important;
    border-color: #28A745 !important;
}
.stAlert[data-testid="stWarning"] {
    background-color: #FFF3E0 !important;
    border-color: #FF8C00 !important;
}
.stAlert[data-testid="stError"] {
    background-color: #FFE6E6 !important;
    border-color: #DC3545 !important;
}

/* Chart and Plotly elements (fallback) */
.js-plotly-plot .plotly {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Fix dropdown menu z-index */
ul[role="listbox"] {
    z-index: 9999 !important;
}

/* Button focus states */
.stButton > button:focus,
.stDownloadButton > button:focus {
    outline: 3px solid rgba(74, 144, 226, 0.5) !important;
    outline-offset: 2px !important;
}

/* Disabled button states */
.stButton > button:disabled,
.stDownloadButton > button:disabled {
    background: #E0E0E0 !important;
    color: #999999 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Loading spinner overlay for buttons */
.stButton > button[disabled] {
    position: relative !important;
    pointer-events: none !important;
}

</style>
""", unsafe_allow_html=True)




    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    Select a page to explore NHS Trust energy analysis:
    - **Dashboard**: Overview with visualizations.
    - **Data Preprocessing**: Upload and process data.
    - **Mathematical Overview**: Totals and metrics.
    - **Energy Analysis**: Consumption patterns.
    - **Financial Analysis**: Costs and ROI.
    - **Carbon Analysis**: Emissions and Net Zero.
    - **Trust Analysis**: Individual performance.
    - **AI-Justified Analysis**: AI-driven insights.
    """)
    
    pages = {
        
        "Data Preprocessing": page_data_preprocessing,
        "Dashboard": page_dashboard,
        "Mathematical Overview": page_overview,
        "Energy Analysis": page_energy,
        "Financial Analysis": page_financial,
        "Carbon Analysis": page_carbon,
        "Trust Analysis": page_trust_analysis,
        "AI-Justified Analysis": page_ai_analysis
    }
    
    selected_page = st.sidebar.selectbox("Select Analysis Page", list(pages.keys()))
    
    if selected_page not in ["Data Preprocessing", "AI-Justified Analysis"] and 'processed_data' not in st.session_state:
        st.warning("Please process data first using the Data Preprocessing page.")
        return
    
    if selected_page in ["Data Preprocessing", "AI-Justified Analysis"]:
        pages[selected_page]()
    else:
        pages[selected_page](st.session_state.processed_data)

if __name__ == "__main__":
    main()
