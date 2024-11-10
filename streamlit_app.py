import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.title("Benchmarking Indian Ports: \nA Data-Driven Analysis of Operational Efficiency\n\n By:Dipti Kothari-23070126040  and Eric Siqueira-23070126041")
# Define a cleaning function
def clean_dataframe(df):
    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Apply string cleanup to all columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Remove leading/trailing spaces from string values
    df = df.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)  # Remove all spaces from strings

    return df
# File uploaders for each CSV file
trt_file = st.file_uploader("Upload TRT CSV file", type="csv")
traffic_file = st.file_uploader("Upload Traffic CSV file", type="csv")
capacity_file = st.file_uploader("Upload Capacity CSV file", type="csv")
utilization_file = st.file_uploader("Upload Utilization CSV file", type="csv")
pre_berthing_file = st.file_uploader("Upload Pre-Berthing Detention CSV file", type="csv")
output_file = st.file_uploader("Upload Output per Ship Berth Day CSV file", type="csv")

# Read and display the uploaded files
if trt_file is not None:
    trt_df = pd.read_csv(trt_file)
    if clean_data:
        trt_df = clean_dataframe(trt_df)
    st.write("TRT DataFrame - First Few Rows:")
    st.dataframe(trt_df.head())

if traffic_file is not None:
    traffic_df = pd.read_csv(traffic_file)
    if clean_data:
        traffic_df = clean_dataframe(traffic_df)
    st.write("Traffic DataFrame - First Few Rows:")
    st.dataframe(traffic_df.head())

if capacity_file is not None:
    capacity_df = pd.read_csv(capacity_file)
    if clean_data:
        capacity_df = clean_dataframe(capacity_df)
    st.write("Capacity DataFrame - First Few Rows:")
    st.dataframe(capacity_df.head())

if utilization_file is not None:
    utilization_df = pd.read_csv(utilization_file)
    if clean_data:
        utilization_df = clean_dataframe(utilization_df)
    st.write("Utilization DataFrame - First Few Rows:")
    st.dataframe(utilization_df.head())

if pre_berthing_file is not None:
    pre_berthing_df = pd.read_csv(pre_berthing_file)
    if clean_data:
        pre_berthing_df = clean_dataframe(pre_berthing_df)
    st.write("Pre-Berthing Detention DataFrame - First Few Rows:")
    st.dataframe(pre_berthing_df.head())

if output_file is not None:
    output_df = pd.read_csv(output_file)
    if clean_data:
        output_df = clean_dataframe(output_df)
    st.write("Output per Ship Berth Day DataFrame - First Few Rows:")
    st.dataframe(output_df.head())
