# This cell uses the %%writefile magic to save the code below into a file named 'app.py'.
# You can then download this file from the Colab file explorer on the left.

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load all saved model artefacts ---
@st.cache_resource
def load_models():
    try:
        loaded_kmeans   = joblib.load('kmeans_rfm_model.pkl')
        loaded_scaler   = joblib.load('rfm_scaler.pkl')
        loaded_sim_df   = joblib.load('item_similarity_matrix.pkl')
        loaded_labels   = joblib.load('segment_labels.pkl')
        loaded_features = joblib.load('cluster_features.pkl')
        return loaded_kmeans, loaded_scaler, loaded_sim_df, loaded_labels, loaded_features
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure all .pkl files are in the same directory as the app.py.")
        st.stop()

kmeans_model, rfm_scaler, item_sim_df, segment_labels_map, cluster_features = load_models()

# --- 2. Helper Functions ---
def rfm_preprocessing_predict(recency, frequency, monetary, scaler, kmeans_model, segment_labels, cluster_features):
    new_customer_df = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})
    new_customer_df['Log_Frequency'] = np.log1p(new_customer_df['Frequency'])
    new_customer_df['Log_Monetary']  = np.log1p(new_customer_df['Monetary'])
    X_new_customer_scaled = scaler.transform(new_customer_df[cluster_features])
    cluster_id = kmeans_model.predict(X_new_customer_scaled)[0]
    return segment_labels.get(cluster_id, f"Cluster {cluster_id}")

def get_recommendations(product_name, sim_df, top_n=5):
    p = str(product_name).strip().upper()
    if p not in sim_df.index:
        matches = [x for x in sim_df.index if p in x]
        if not matches: return f"Product '{product_name}' not found."
        p = matches[0]
        st.info(f"Closest match found: '{p}'")
    scores = sim_df[p].sort_values(ascending=False).iloc[1:top_n+1]
    return scores.reset_index().rename(columns={'Description':'Product', p:'Similarity'})

# --- 3. Streamlit App Layout ---
st.set_page_config(page_title="Shopper Spectrum ML App", layout="wide")
st.title("🛒 Shopper Spectrum: Customer Segmentation & Recommendations")

tab1, tab2 = st.tabs(["Customer Segmentation", "Product Recommendations"])

with tab1:
    st.header("Predict Segment")
    r = st.number_input("Recency", value=30)
    f = st.number_input("Frequency", value=5)
    m = st.number_input("Monetary", value=250.0)
    if st.button("Predict"):
        st.success(f"Segment: {rfm_preprocessing_predict(r, f, m, rfm_scaler, kmeans_model, segment_labels_map, cluster_features)}")

with tab2:
    st.header("Recommendations")
    prod = st.text_input("Product Name")
    if st.button("Get Recommendations"):
        recs = get_recommendations(prod, item_sim_df)
        st.dataframe(recs) if not isinstance(recs, str) else st.warning(recs)
