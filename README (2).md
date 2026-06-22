# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

A production-ready Machine Learning web application that segments e-commerce customers using **RFM Analysis + KMeans Clustering** and provides **intelligent product recommendations** using item-based collaborative filtering.

🔗 **Live App:** [shopperspectrum-glcxq3eezpqqufr4lrwudw.streamlit.app](https://shopperspectrum-glcxq3eezpqqufr4lrwudw.streamlit.app)

---

## 📌 Problem Statement

E-commerce businesses struggle with blanket marketing — sending the same message to every customer regardless of their behaviour. This project solves that by:
- Automatically segmenting customers based on their purchase behaviour
- Recommending relevant products to each customer persona
- Providing a live inference interface for real-time predictions

---

## 🧠 ML Pipeline

### 1. RFM Feature Engineering
Each customer is profiled using three behavioural metrics:
| Feature | Description |
|---|---|
| **Recency** | Days since last purchase |
| **Frequency** | Number of transactions |
| **Monetary** | Total spend (£) |

Log transformations applied to Frequency and Monetary to handle skewness.

### 2. Customer Segmentation — KMeans Clustering
- Features scaled using `StandardScaler`
- Optimal K selected via **Elbow Method** + **Silhouette Score**
- Customers labelled into meaningful business segments:
  - 🏆 High-Value
  - 🔄 Loyal
  - ⚠️ At-Risk
  - 💤 Dormant

### 3. Product Recommendations — Item-Based Collaborative Filtering
- Built a **customer-product matrix** from transaction data
- Computed **cosine similarity** between all product pairs
- Returns top-N similar products for any given product name

---

## 🖥️ App Features

| Tab | Feature |
|---|---|
| **Customer Segmentation** | Input Recency, Frequency, Monetary → Get segment prediction instantly |
| **Product Recommendations** | Input any product name → Get top 5 similar products |

---

## 🗂️ Project Structure

```
shopper_spectrum/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── kmeans_rfm_model.pkl        # Trained KMeans model
├── rfm_scaler.pkl              # Fitted StandardScaler
├── segment_labels.pkl          # Cluster ID → Segment name mapping
├── cluster_features.pkl        # Feature names used during training
└── README.md
```

> `item_similarity_matrix.pkl` is hosted on Google Drive and downloaded automatically at runtime due to its size (~116 MB).

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/paradkarharsh/shopper_spectrum.git
cd shopper_spectrum

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📦 Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data processing & RFM computation |
| `scikit-learn` | KMeans clustering, StandardScaler |
| `joblib` | Model serialisation |
| `gdown` | Large file download from Google Drive |
| `streamlit` | Web application & deployment |

---

## 📊 Business Impact

- Enables **precision customer targeting** — moving from blanket to personalised marketing
- Reduces **Customer Acquisition Cost** by focusing spend on high-value segments
- Improves **retention rates** by identifying at-risk customers early
- Increases **Average Order Value** through intelligent product personalisation

---

## 👤 Author

**Harshvardhan Paradkar**
B.Tech CSE (AI & ML) — IPS Academy, Indore
Intern @ Vault of Codes

[![GitHub](https://img.shields.io/badge/GitHub-paradkarharsh-black?logo=github)](https://github.com/paradkarharsh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/paradkarharsh)
