import streamlit as st

st.set_page_config(page_title="Moon Landing Conspiracy", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>Moon Landing Conspiracy</h1>", unsafe_allow_html=True)
st.markdown("---")

# Tabs for sections
tabs = st.tabs(["Overview", "Data Analysis", "Machine Learning", "Video Detection", "Statistical Testing", "Conclusion"])

# ---- Tab 1: Overview ----
with tabs[0]:
    st.header("Overview")
    st.write("""
    This section provides a background on the moon landing conspiracy theory,
    and introduces the use of modern technology like machine learning and video analysis
    to explore the authenticity of related video footage.
    """)

# ---- Tab 2: Data Analysis ----
with tabs[1]:
    st.header("Data Analysis")
    st.write("Explore and visualize key frame-level metrics such as brightness, edge density, and noise.")

# ---- Tab 3: Machine Learning ----
with tabs[2]:
    st.header("Machine Learning")
    st.write("""
    A simple machine learning model (Random Forest) trained on handcrafted features
    to distinguish real vs fake moon landing footage.
    """)

# ---- Tab 4: Video Detection ----
with tabs[3]:
    st.header("Video Detection")
    st.write("This section demonstrates three methods for detecting fake videos.")

# ---- Tab 5: Statistical Testing ----
with tabs[4]:
    st.header("Statistical Testing")
    st.write("""
    Use statistical methods (e.g., t-tests) to compare features like brightness or edge density
    between real and fake video frames to test for significant differences.
    """)

# ---- Tab 6: Conclusion ----
with tabs[5]:
    st.header("Conclusion")
    st.write("""
    Based on frame features, machine learning, and deepfake detection models,
    there's strong potential to detect video manipulation using computational techniques.
    """)