import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display

st.title("Benchmarking Indian Ports: \nA Data-Driven Analysis of Operational Efficiency\n\n By:Dipti Kothari-23070126040 and Eric Siqueira-23070126041")

# Define the cleaning function
def clean_dataframe(df):
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
    return df

# Checkbox to apply cleaning function
clean_data = st.checkbox("Clean DataFrames (Remove spaces from column names and string values)")

# File uploaders for each CSV file
trt_file = st.file_uploader("Upload TRT CSV file", type="csv")
traffic_file = st.file_uploader("Upload Traffic CSV file", type="csv")
capacity_file = st.file_uploader("Upload Capacity CSV file", type="csv")
utilization_file = st.file_uploader("Upload Utilization CSV file", type="csv")
pre_berthing_file = st.file_uploader("Upload Pre-Berthing Detention CSV file", type="csv")
output_file = st.file_uploader("Upload Output per Ship Berth Day CSV file", type="csv")

# Function to read a CSV file and handle EmptyDataError
def safe_read_csv(file):
    try:
        return pd.read_csv(file)
    except pd.errors.EmptyDataError:
        st.write(f"Error: The uploaded file {file.name} is empty. Please upload a valid file.")
        return None

# Display the first few rows and check for missing values if uploaded
def display_data_info(df, name):
    if df is not None:
        if clean_data:
            df = clean_dataframe(df)
        st.write(f"{name} - First Few Rows:")
        st.dataframe(df.head())
        st.write(f"Missing Values in {name}:")
        st.write(df.isnull().sum())

# Read and display each dataset with error handling for empty files
if trt_file is not None:
    trt_df = safe_read_csv(trt_file)
    display_data_info(trt_df, "TRT DataFrame")

if traffic_file is not None:
    traffic_df = safe_read_csv(traffic_file)
    display_data_info(traffic_df, "Traffic DataFrame")

if capacity_file is not None:
    capacity_df = safe_read_csv(capacity_file)
    display_data_info(capacity_df, "Capacity DataFrame")

if utilization_file is not None:
    utilization_df = safe_read_csv(utilization_file)
    display_data_info(utilization_df, "Utilization DataFrame")

if pre_berthing_file is not None:
    pre_berthing_df = safe_read_csv(pre_berthing_file)
    display_data_info(pre_berthing_df, "Pre-Berthing Detention DataFrame")

if output_file is not None:
    output_df = safe_read_csv(output_file)
    display_data_info(output_df, "Output per Ship Berth Day DataFrame")

# Analysis section
if all([trt_file, traffic_file, capacity_file, utilization_file, output_file]):
    if all([trt_df is not None, traffic_df is not None, capacity_df is not None, utilization_df is not None, output_df is not None]):
        
        def analyze_port_statistics(port_name):
            if port_name not in capacity_df.columns:
                st.write(f"Port {port_name} not found in data.")
                return pd.DataFrame()

            # Extract yearly statistics for the port
            stats_df = pd.DataFrame({
                'Year': capacity_df['Year'],
                'Capacity': capacity_df[port_name],
                'Traffic': traffic_df[port_name],
                'Utilization': utilization_df[port_name]
            })

            # Calculate year-over-year growth rates and moving averages
            stats_df['Capacity_Growth'] = stats_df['Capacity'].pct_change() * 100
            stats_df['Traffic_Growth'] = stats_df['Traffic'].pct_change() * 100
            stats_df['Utilization_MA'] = stats_df['Utilization'].rolling(window=3).mean()

            return stats_df

        # Select a port for analysis
        port_name = st.selectbox(
            "Select Port",
            options=capacity_df.columns[1:],  # Assuming first column is 'Year'
            index=0
        )

        # Analyze and display statistics for the selected port
        stats_df = analyze_port_statistics(port_name)
        if not stats_df.empty:
            st.write(f"Statistics for Port: {port_name}")
            st.dataframe(stats_df)

# Plotting function
def plot_metric_trends(df, metric_name):
    plt.figure(figsize=(10, 6))
    for port in [col for col in df.columns if col not in ['Year', 'All Ports']]:
        plt.plot(df['Year'], df[port], label=port)
    plt.title(f'{metric_name} Trends Across Ports')
    plt.xlabel('Year')
    plt.ylabel(metric_name)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()  # Close the plot to free memory

# Plotting the trends for available data
if 'capacity_df' in locals() and capacity_df is not None:
    st.write("Capacity Trends Across Ports")
    plot_metric_trends(capacity_df, 'Capacity')

if 'utilization_df' in locals() and utilization_df is not None:
    st.write("Utilization Trends Across Ports")
    plot_metric_trends(utilization_df, 'Utilization')

if 'trt_df' in locals() and trt_df is not None:
    st.write("TRT Trends Across Ports")
    plot_metric_trends(trt_df, 'TRT')

if 'output_df' in locals() and output_df is not None:
    st.write("Output Trends Across Ports")
    plot_metric_trends(output_df, 'Output')

if 'pre_berthing_df' in locals() and pre_berthing_df is not None:
    st.write("Pre-Berthing Trends Across Ports")
    plot_metric_trends(pre_berthing_df, 'Pre-Berthing')

# Function to analyze port correlations
def analyze_port_correlations(port_name):
    if port_name not in capacity_df.columns:
        st.write(f"Port {port_name} not found in capacity data.")
        return pd.DataFrame()

    # Check if output_df is defined and contains the port
    if 'output_df' not in locals() or port_name not in output_df.columns:
        st.write(f"Port {port_name} not found in Output data.")
        return pd.DataFrame()

    # Create the correlation dataframe only if all necessary data is available
    correlation_df = pd.DataFrame({
        'Capacity': capacity_df[port_name],
        'Traffic': traffic_df[port_name],
        'Utilization': utilization_df[port_name],
        'TRT': trt_df[port_name],
        'Output': output_df[port_name]
    })

    return correlation_df.corr()

def plot_port_comparison(metric_df, year, metric_name):
    year_data = metric_df[metric_df['Year'] == year].melt(
        id_vars=['Year'],
        value_vars=[col for col in metric_df.columns if col not in ['Year', 'All Ports']]
    )

    plt.figure(figsize=(15, 6))
    sns.barplot(x='variable', y='value', data=year_data)
    plt.title(f'{metric_name} Comparison Across Ports ({year})')
    plt.xticks(rotation=45)
    plt.xlabel('Ports')
    plt.ylabel(metric_name)
    plt.tight_layout()
    st.pyplot(plt)  # Use Streamlit's built-in pyplot to display plots
    plt.close()

# Assuming you have the DataFrames like capacity_df, traffic_df, etc.

# Streamlit dropdown for selecting year
year_options = [str(year) for year in capacity_df['Year'].unique()]
selected_year = st.selectbox('Select Year:', year_options)

# Display and update plots based on selected year
if selected_year:
    st.write(f"Showing data for the year: {selected_year}")
    
    # Plot each metric for the selected year
    plot_port_comparison(capacity_df, selected_year, 'Capacity')
    plot_port_comparison(traffic_df, selected_year, 'Traffic')
    plot_port_comparison(utilization_df, selected_year, 'Utilization')
    plot_port_comparison(trt_df, selected_year, 'TRT')
    plot_port_comparison(output_df, selected_year, 'Output')
    plot_port_comparison(pre_berthing_df, selected_year, 'Pre-Berthing')
