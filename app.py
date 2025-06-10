
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(page_title="Moon Landing Image Analysis", layout="wide")
st.title("Moon Landing: Image Collection & Analysis")

st.header("Image Collection")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    st.image("images/sample_real_original.jpg", caption="Real - Original", use_column_width=True)
with col2:
    st.subheader("Processed Grayscale Image")
    st.image("images/sample_real_gray.jpg", caption="Real - Grayscale", use_column_width=True)

st.markdown("---")
st.header("Image Analysis")

analysis_type = st.selectbox("Select Analysis Feature:", ["GLCM Texture", "Shadow Angles", "Contour Complexity", "Lander Metrics"])

def load_histogram_data(filepath):
    return np.load(filepath)

if analysis_type == "GLCM Texture":
    st.subheader("GLCM Texture Comparison")
    feature = st.selectbox("Choose texture feature:", ["Contrast", "Homogeneity", "Energy", "Correlation"])
    real = load_histogram_data(f"histogram_data/real_{feature.lower()}.npy")
    fake = load_histogram_data(f"histogram_data/fake_{feature.lower()}.npy")
    fig, ax = plt.subplots()
    ax.hist(real, bins=20, alpha=0.6, label='Real', color='skyblue')
    ax.hist(fake, bins=20, alpha=0.6, label='Fake', color='salmon')
    ax.set_title(f'{feature} Histogram')
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

elif analysis_type == "Shadow Angles":
    st.subheader("Shadow Angle Distribution")
    real = load_histogram_data("histogram_data/real_shadow_angles.npy")
    fake = load_histogram_data("histogram_data/fake_shadow_angles.npy")
    fig, ax = plt.subplots()
    ax.hist(real, bins=30, alpha=0.6, label='Real', color='green')
    ax.hist(fake, bins=30, alpha=0.6, label='Fake', color='red')
    ax.set_title("Shadow Angle Distribution")
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

elif analysis_type == "Contour Complexity":
    st.subheader("Contour Complexity")
    real = load_histogram_data("histogram_data/real_contours.npy")
    fake = load_histogram_data("histogram_data/fake_contours.npy")
    fig, ax = plt.subplots()
    ax.hist(real, bins=20, alpha=0.6, label='Real', color='purple')
    ax.hist(fake, bins=20, alpha=0.6, label='Fake', color='orange')
    ax.set_title("Contour Count Comparison")
    ax.set_xlabel("Number of Contours")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

elif analysis_type == "Lander Metrics":
    metric = st.selectbox("Choose metric:", ["Max Area", "Average Perimeter", "Number of Contours"])
    file_key = metric.lower().replace(" ", "_")
    real = load_histogram_data(f"histogram_data/real_{file_key}.npy")
    fake = load_histogram_data(f"histogram_data/fake_{file_key}.npy")
    fig, ax = plt.subplots()
    ax.hist(real, bins=20, alpha=0.6, label='Real', color='blue')
    ax.hist(fake, bins=20, alpha=0.6, label='Fake', color='orange')
    ax.set_title(f"Lander Metric: {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
