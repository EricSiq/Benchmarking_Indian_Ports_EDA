import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Streamlit Title
st.title("Benchmarking Indian Ports: \nA Data-Driven Analysis of Operational Efficiency\n\n By: Dipti Kothari-23070126040 and Eric Siqueira-23070126041")

# Define the cleaning function for the data
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

# Function to analyze TRT performance and trends
def analyze_trt_performance(trt_df):
    # Ensure trt_df has valid data
    if trt_df is None or trt_df.empty:
        st.error("TRT DataFrame is empty or not available.")
        return pd.DataFrame()

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

    # Display the plot using Streamlit's pyplot function
    st.pyplot(plt)

    return performance_summary

# Main section to handle file uploads and display data
if trt_file is not None:
    trt_df = safe_read_csv(trt_file)  # Load the TRT data
    if trt_df is not None:
        st.write("TRT DataFrame loaded successfully.")
        display_data_info(trt_df, "TRT DataFrame")  # Display basic info about the file

        # Call the analysis function to calculate and display performance summary
        performance_summary = analyze_trt_performance(trt_df)
        if not performance_summary.empty:
            st.write("TRT Performance Summary:")
            st.dataframe(performance_summary)

# Function to analyze port statistics
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
def plot_port_comparison(df, selected_year, metric_name):
    # Filter data for the selected year
    year_df = df[df['Year'] == int(selected_year)]

    if year_df.empty:
        st.write(f"No data available for the year {selected_year}.")
        return
    
    # Plot the data for each port in the selected year
    plt.figure(figsize=(10, 6))
    ports = [col for col in year_df.columns if col != 'Year']  # Exclude the 'Year' column
    for port in ports:
        plt.plot(year_df['Year'], year_df[port], label=port)
    plt.title(f'{metric_name} for Ports in {selected_year}')
    plt.xlabel('Port')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()  # Close the plot to free memory

# Main code for Streamlit app
if 'capacity_df' in locals() and capacity_df is not None:
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
# Streamlit dropdown for selecting year
if 'capacity_df' in locals() and capacity_df is not None:
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

# Check if all columns in trt_df are numeric, and convert them to numeric if needed
if trt_df is not None:
    # Convert all columns to numeric (ignoring errors for non-numeric)
    trt_df = trt_df.apply(pd.to_numeric, errors='ignore')
