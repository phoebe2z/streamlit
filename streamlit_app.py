import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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
        st.image("images/fake_sample.png", caption="Fake Image (Grayscale)", use_column_width=True)

# ---- Tab 3: Image Analysis ----
with tabs[2]:
    st.header("🧩 Image Analysis")

    # Simulated histogram plotting functions per category (replace with real data)
def show_shadow_histogram():
    st.subheader("Shadow Angle Distribution")
    shadow_fake = np.random.normal(90, 5, 100)
    shadow_real = np.random.normal(90, 15, 100)
    plt.hist(shadow_real, bins=30, alpha=0.6, label='Real')
    plt.hist(shadow_fake, bins=30, alpha=0.6, label='Fake')
    plt.xlabel("Shadow Angle (°)")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)

def show_flag_histogram():
    st.subheader("Flag Contour Complexity")
    real = np.random.normal(30000, 5000, 100)
    fake = np.random.normal(23000, 2000, 100)
    plt.hist(real, bins=30, alpha=0.6, label='Real')
    plt.hist(fake, bins=30, alpha=0.6, label='Fake')
    plt.xlabel("Contour Complexity")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)

def show_astronaut_histogram():
    st.subheader("Astronaut Contour Complexity")
    real = np.random.normal(30000, 5000, 100)
    fake = np.random.normal(15000, 3000, 100)
    plt.hist(real, bins=30, alpha=0.6, label='Real')
    plt.hist(fake, bins=30, alpha=0.6, label='Fake')
    plt.xlabel("Complexity Score")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)

def show_horizon_histogram():
    st.subheader("Horizon Line Angles")
    real = np.random.normal(10, 10, 100)
    fake = np.concatenate([np.random.normal(90, 5, 50), np.random.normal(-90, 5, 50)])
    plt.hist(real, bins=30, alpha=0.6, label='Real')
    plt.hist(fake, bins=30, alpha=0.6, label='Fake')
    plt.xlabel("Line Angle (°)")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)

def show_surface_histogram():
    st.subheader("Surface Texture Features: Energy")
    real = np.random.normal(0.2, 0.05, 100)
    fake = np.random.normal(0.5, 0.1, 100)
    plt.hist(real, bins=30, alpha=0.6, label='Real')
    plt.hist(fake, bins=30, alpha=0.6, label='Fake')
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)

def show_lander_histogram():
    st.subheader("Lander Contour Features")
    real = np.random.normal(10000, 3000, 100)
    fake = np.random.normal(6000, 2000, 100)
    plt.hist(real, bins=30, alpha=0.6, label='Real')
    plt.hist(fake, bins=30, alpha=0.6, label='Fake')
    plt.xlabel("Contour Area")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)

# Mapping selection to functions
analysis_map = {
    "Shadow": show_shadow_histogram,
    "Flag": show_flag_histogram,
    "Astronaut": show_astronaut_histogram,
    "Horizon": show_horizon_histogram,
    "Surface": show_surface_histogram,
    "Lander": show_lander_histogram,
}

# Streamlit UI
st.title("AI-Based Lunar Image Analysis")
choice = st.selectbox("Choose a category for analysis:", list(analysis_map.keys()))
analysis_map[choice]()

# Explanation
explanations = {
    "Shadow": "Consistent shadow angles near 90° in fake images reveal synthetic lighting. Real shadows vary due to terrain and sunlight.",
    "Flag": "Fake flags tend to cluster around mid-complexity, possibly due to simulated waving. Real ones are more variable.",
    "Astronaut": "Real astronaut figures show higher contour complexity due to real texture details, while fake images are smoother.",
    "Horizon": "Fake images show unnatural horizon lines around ±90°, indicating possible rendering flaws or AI artifacts.",
    "Surface": "Fake surfaces often show higher texture energy and uniformity, suggesting artificial patterning.",
    "Lander": "Fake landers have fewer and simpler contours; real landers exhibit detailed and diverse shapes.",
}

st.markdown(f"**Conclusion**: {explanations[choice]}")

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
