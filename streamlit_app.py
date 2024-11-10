import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.title("Benchmarking Indian Ports: \nA Data-Driven Analysis of Operational Efficiency\n\n By:Dipti Kothari-23070126040  and Eric Siqueira-23070126041")
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

# Display the first few rows and check for missing values if uploaded
def display_data_info(df, name):
    if df is not None:
        if clean_data:
            df = clean_dataframe(df)
        st.write(f"{name} - First Few Rows:")
        st.dataframe(df.head())
        st.write(f"Missing Values in {name}:")
        st.write(df.isnull().sum())

if trt_file is not None:
    trt_df = pd.read_csv(trt_file)
    display_data_info(trt_df, "TRT DataFrame")

if traffic_file is not None:
    traffic_df = pd.read_csv(traffic_file)
    display_data_info(traffic_df, "Traffic DataFrame")

if capacity_file is not None:
    capacity_df = pd.read_csv(capacity_file)
    display_data_info(capacity_df, "Capacity DataFrame")

if utilization_file is not None:
    utilization_df = pd.read_csv(utilization_file)
    display_data_info(utilization_df, "Utilization DataFrame")

if pre_berthing_file is not None:
    pre_berthing_df = pd.read_csv(pre_berthing_file)
    display_data_info(pre_berthing_df, "Pre-Berthing Detention DataFrame")

if output_file is not None:
    output_df = pd.read_csv(output_file)
    display_data_info(output_df, "Output per Ship Berth Day DataFrame")
