import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained KMeans model and scaler
model = pkl.load(open('kmeans_model.pkl', 'rb'))
scaler = pkl.load(open('scaler.pkl', 'rb'))  # Load pre-trained scaler

# Streamlit App
st.title("Customer Segmentation Using KMeans")
st.write("Enter customer details to predict the cluster:")

# User Input Fields
amount = st.number_input("Enter Amount Spent:", min_value=0.0, step=100.0)
frequency = st.number_input("Enter Purchase Frequency:", min_value=1, step=1)
recency = st.number_input("Enter Days Since Last Purchase:", min_value=0, step=1)

# Prediction
if st.button("Predict Cluster"):
    # Prepare input data
    input_data = np.array([[amount, frequency, recency]], dtype=float)  # Ensure it's a NumPy array
    scaled_data = scaler.transform(input_data)  # Transform using pre-trained scaler
    
    # Predict cluster
    cluster = int(model.predict(scaled_data)[0])  # Ensure it's an integer

    # Define cluster names
    cluster_names = {
        0: "Loyal Customers (वफादार ग्राहक)",
        1: "Lost Customers (खोए हुए ग्राहक)",
        2: "Frequent Buyers (बार-बार खरीदने वाले ग्राहक)",
    }

    # Display the result
    st.write(f"The customer belongs to Cluster: **{cluster_names.get(cluster, 'Unknown')}**")
