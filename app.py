import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#  PAGE CONFIG
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.markdown("""
<style>
/* Make all headings and text black */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)
# APP TITLE
st.title("Customer Segmentation Dashboard")
st.markdown("""
This system uses **K-Means Clustering** to group customers based on their purchasing behavior and similarities.  
👉 Discover hidden customer groups without predefined labels.

**Dataset:** [Wholesale Customers Data](https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set)
""")

#  LOAD DATA 
df = pd.read_csv('Wholesale customers data.csv')
numerical_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

#  INPUT SECTION 
st.subheader("Clustering Inputs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    feature1 = st.selectbox("Select Feature 1", numerical_features, index=0)
with col2:
    feature2 = st.selectbox("Select Feature 2", numerical_features, index=1)
with col3:
    k = st.number_input("Number of Clusters (K)", min_value=2, max_value=10, value=5, step=1)
with col4:
    random_state = st.number_input("Random State (optional)", value=42, step=1)

# RUN CLUSTERING 
if st.button("🟦 Run Clustering"):

    # Prepare data for clustering
    X = df[[feature1, feature2]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    df['Cluster'] = kmeans.labels_

    #  VISUALIZATION 
    st.subheader("Cluster Visualization")

    # Create two columns, plot in the first one (half width)
    col_plot, col_empty = st.columns([1, 1])  # 50% width for plot, 50% empty

    with col_plot:
        fig, ax = plt.subplots(figsize=(4, 3))  # smaller figure

        colors = plt.cm.get_cmap('tab10', k)

        for i in range(k):
            ax.scatter(X_scaled[df['Cluster'] == i, 0],
                       X_scaled[df['Cluster'] == i, 1],
                       s=40, c=[colors(i)], label=f'Cluster {i}', alpha=0.7)

        # Plot centroids
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   s=80, c='gold', label='Centroid', edgecolor='black')

        ax.set_xlabel(f'{feature1} (scaled)', fontsize=8)
        ax.set_ylabel(f'{feature2} (scaled)', fontsize=8)
        ax.set_title(f'K-Means Clusters (K={k})', fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    #  CLUSTER SUMMARY
    st.subheader("Cluster Summary")
    summary = df.groupby('Cluster')[[feature1, feature2]].agg(['mean', 'count'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    st.dataframe(summary, height=200)

    # BUSINESS INTERPRETATION 
    st.subheader("Business Insights")
    for i in range(k):
        cluster_data = df[df['Cluster'] == i]
        dominant = cluster_data[[feature1, feature2]].mean().idxmax()
        st.markdown(f"🟢 **Cluster {i}:** Customers spend most on **{dominant}**. "
                    f"Average spending: {cluster_data[[feature1, feature2]].mean().to_dict()}")

    st.markdown("""
📌 **Note:** Customers in the same cluster exhibit similar purchasing behaviour and can be targeted with similar business strategies.
""")
