import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Title
st.title("Benchmarking Indian Ports: \nA Data-Driven Analysis of Operational Efficiency\n\n By: Dipti Kothari-23070126040 and Eric Siqueira-23070126041")

# Function to clean the dataframe
def clean_dataframe(df):
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
    return df

# Checkbox to clean dataframes
clean_data = st.checkbox("Clean DataFrames (Remove spaces from column names and string values)")

# File uploaders for CSV files
trt_file = st.file_uploader("Upload TRT CSV file", type="csv")
traffic_file = st.file_uploader("Upload Traffic CSV file", type="csv")
capacity_file = st.file_uploader("Upload Capacity CSV file", type="csv")
utilization_file = st.file_uploader("Upload Utilization CSV file", type="csv")
pre_berthing_file = st.file_uploader("Upload Pre-Berthing Detention CSV file", type="csv")
output_file = st.file_uploader("Upload Output per Ship Berth Day CSV file", type="csv")

# Function to safely read CSV files
def safe_read_csv(file):
    try:
        return pd.read_csv(file)
    except pd.errors.EmptyDataError:
        st.write(f"Error: The uploaded file {file.name} is empty. Please upload a valid file.")
        return None

# Display information about the uploaded data
def display_data_info(df, name):
    if df is not None:
        if clean_data:
            df = clean_dataframe(df)
        st.write(f"{name} - First Few Rows:")
        st.dataframe(df.head())
        st.write(f"Missing Values in {name}:")
        st.write(df.isnull().sum())

# Read and display each dataset
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

# Function to plot port data comparisons by year
def plot_port_comparison(df, selected_year, metric_name):
    # Filter the data for the selected year
    df_year = df[df['Year'] == int(selected_year)]

    # If the year exists, plot the comparison for the metric
    if not df_year.empty:
        plt.figure(figsize=(12, 6))
        for port in [col for col in df.columns if col not in ['Year', 'All Ports']]:
            plt.plot(df_year['Year'], df_year[port], label=port)
        
        plt.title(f'{metric_name} Comparison for Year: {selected_year}')
        plt.xlabel('Port')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(loc='best')
        plt.grid(True)
        
        # Display the plot using Streamlit
        st.pyplot(plt)
        plt.close()  # Close the plot to free memory
    else:
        st.write(f"No data available for the selected year: {selected_year}")

# Streamlit dropdown for selecting year
if capacity_df is not None:
    year_options = [str(year) for year in capacity_df['Year'].unique()]
    selected_year = st.selectbox('Select Year:', year_options)

    # Display and update plots based on selected year
    if selected_year:
        st.write(f"Showing data for the year: {selected_year}")
        
        # Plot each metric for the selected year
        if capacity_df is not None:
            plot_port_comparison(capacity_df, selected_year, 'Capacity')
        if traffic_df is not None:
            plot_port_comparison(traffic_df, selected_year, 'Traffic')
        if utilization_df is not None:
            plot_port_comparison(utilization_df, selected_year, 'Utilization')
        if trt_df is not None:
            plot_port_comparison(trt_df, selected_year, 'TRT')
        if output_df is not None:
            plot_port_comparison(output_df, selected_year, 'Output')

# Analyze Turn Round Time (TRT) performance
def analyze_trt_performance(trt_df):
    # Calculate average TRT for each port
    port_cols = [col for col in trt_df.columns if col not in ['Year', 'All Ports']]
    avg_trt = trt_df[port_cols].mean().sort_values()

    # Calculate TRT trend (improvement rate)
    trt_trend = trt_df[port_cols].apply(lambda x: stats.linregress(range(len(x)), x)[0])

    # Create performance summary
    performance_summary = pd.DataFrame({
        'Average_TRT': avg_trt,
        'TRT_Trend': trt_trend
    })

    # Plot average TRT comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x=avg_trt.index, y=avg_trt.values)
    plt.title('Average Turn Round Time by Port')
    plt.xticks(rotation=45)
    plt.ylabel('Average TRT (days)')
    plt.xlabel('Port')
    plt.tight_layout()

    # Display plot using Streamlit
    st.pyplot(plt)

    return performance_summary

# Proceed if TRT data is available
if trt_df is not None:
    analyze_trt_performance(trt_df)

# Additional analysis for ports
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

# Function to plot trends across ports
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

# Plotting trends for available data
if 'capacity_df' in locals() and capacity_df is not None:
    st.write("Capacity Trends Across Ports")
    plot_metric_trends(capacity_df, 'Capacity')

if 'utilization_df' in locals() and utilization_df is not None:
    st.write("Utilization Trends Across Ports")
    plot_metric_trends(utilization_df, 'Utilization')

if 'traffic_df' in locals() and traffic_df is not None:
    st.write("Traffic Trends Across Ports")
    plot_metric_trends(traffic_df, 'Traffic')

if 'output_df' in locals() and output_df is not None:
    st.write("Output per Ship Berth Day Trends Across Ports")
    plot_metric_trends(output_df, 'Output')
