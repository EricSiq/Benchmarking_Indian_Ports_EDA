import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ipywidgets as widgets
from IPython.display import display, clear_output

st.title("Benchmarking Indian Ports: A Data-Driven Analysis of Operational Efficiency
")
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
    st.write("TRT DataFrame Preview:")
    st.dataframe(trt_df)

if traffic_file is not None:
    traffic_df = pd.read_csv(traffic_file)
    st.write("Traffic DataFrame Preview:")
    st.dataframe(traffic_df)

if capacity_file is not None:
    capacity_df = pd.read_csv(capacity_file)
    st.write("Capacity DataFrame Preview:")
    st.dataframe(capacity_df)

if utilization_file is not None:
    utilization_df = pd.read_csv(utilization_file)
    st.write("Utilization DataFrame Preview:")
    st.dataframe(utilization_df)

if pre_berthing_file is not None:
    pre_berthing_df = pd.read_csv(pre_berthing_file)
    st.write("Pre-Berthing Detention DataFrame Preview:")
    st.dataframe(pre_berthing_df)

if output_file is not None:
    output_df = pd.read_csv(output_file)
    st.write("Output per Ship Berth Day DataFrame Preview:")
    st.dataframe(output_df)
