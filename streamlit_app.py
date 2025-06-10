import streamlit as st

st.set_page_config(page_title="Moon Landing Conspiracy", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>Moon Landing Conspiracy</h1>", unsafe_allow_html=True)
st.markdown("---")

# Tabs for sections
tabs = st.tabs(["Overview", "Image Collection", "Image Analysis", "Machine Learning", "Video Detection", "Statistical Testing", "Conclusion"])

# ---- Tab 1: Overview ----
with tabs[0]:
    st.header("Overview")
    st.write("""
    This section provides a background on the moon landing conspiracy theory,
    and introduces the use of modern technology like machine learning and video analysis
    to explore the authenticity of related video footage.
    """)

# ---- Tab 2: Image Collection ----
with tabs[1]:
    st.header("📸 Image Collection")
    st.markdown("""
    We collected **200 grayscale-processed images**, equally split between `real` and `fake`, and categorized into four feature-based regions:
    - Surface
    - Shadow
    - Flag
    - Lander
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/real_sample.jpg", caption="Real Image (Grayscale)", use_column_width=True)
    with col2:
        st.image("images/fake_sample.jpg", caption="Fake Image (Grayscale)", use_column_width=True)

# ---- Tab 3: Image Analysis ----
with tabs[2]:
    st.header("🧩 Image Analysis")

    # Surface Texture Features
    st.subheader("GLCM Texture Features (Surface)")
    texture_features = ['contrast', 'homogeneity', 'energy', 'correlation']
    for feature in texture_features:
        st.markdown(f"**{feature.capitalize()}**")
        fig, ax = plt.subplots(figsize=(6, 4))
        real_data = np.load(f"histogram_data/real_surface_glcm_{feature}.npy")
        fake_data = np.load(f"histogram_data/fake_surface_glcm_{feature}.npy")
        min_val = min(min(real_data), min(fake_data))
        max_val = max(max(real_data), max(fake_data))
        ax.hist(real_data, bins=20, alpha=0.6, label='Real', color='skyblue', range=(min_val, max_val))
        ax.hist(fake_data, bins=20, alpha=0.6, label='Fake', color='salmon', range=(min_val, max_val))
        ax.legend()
        st.pyplot(fig)

    # Shadow Angle Distribution
    st.subheader("Shadow Angle Distribution")
    real_angles = np.load("histogram_data/real_shadow_angles.npy")
    fake_angles = np.load("histogram_data/fake_shadow_angles.npy")
    fig, ax = plt.subplots(figsize=(6, 4))
    angle_min = min(min(real_angles), min(fake_angles))
    angle_max = max(max(real_angles), max(fake_angles))
    ax.hist(real_angles, bins=30, alpha=0.6, label='Real Shadows', color='green', range=(angle_min, angle_max))
    ax.hist(fake_angles, bins=30, alpha=0.6, label='Fake Shadows', color='red', range=(angle_min, angle_max))
    ax.legend()
    st.pyplot(fig)

    # Lander Contour Features
    st.subheader("Lander Contour Features")
    lander_features = ['num_contours', 'max_area', 'avg_perimeter']
    for metric in lander_features:
        st.markdown(f"**{metric.replace('_', ' ').title()}**")
        real_data = np.load(f"histogram_data/real_lander_{metric}.npy")
        fake_data = np.load(f"histogram_data/fake_lander_{metric}.npy")
        min_val = min(min(real_data), min(fake_data))
        max_val = max(max(real_data), max(fake_data))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(real_data, bins=20, alpha=0.6, label='Real', color='lightblue', range=(min_val, max_val))
        ax.hist(fake_data, bins=20, alpha=0.6, label='Fake', color='blue', range=(min_val, max_val))
        ax.legend()
        st.pyplot(fig)

# ---- Tab 3: Machine Learning ----
with tabs[3]:
    st.header("Machine Learning")
    st.write("""
    A simple machine learning model (Random Forest) trained on handcrafted features
    to distinguish real vs fake moon landing footage.
    """)

# ---- Tab 4: Video Detection ----
with tabs[4]:
    st.header("Video Detection")
    st.write("This section demonstrates three methods for detecting fake videos.")

# ---- Tab 5: Statistical Testing ----
with tabs[5]:
    st.header("Statistical Testing")
    st.write("""
    Use statistical methods (e.g., t-tests) to compare features like brightness or edge density
    between real and fake video frames to test for significant differences.
    """)

# ---- Tab 6: Conclusion ----
with tabs[6]:
    st.header("Conclusion")
    st.write("""
    Based on frame features, machine learning, and deepfake detection models,
    there's strong potential to detect video manipulation using computational techniques.
    """)
