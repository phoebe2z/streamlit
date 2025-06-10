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
    st.header("🌗 Shadow Angle Distribution")
    real_angles = np.load("histogram_data/real_shadow_angles.npy")
    fake_angles = np.load("histogram_data/fake_shadow_angles.npy")

    fig, ax = plt.subplots(figsize=(7, 4))
    angle_min = min(min(real_angles), min(fake_angles))
    angle_max = max(max(real_angles), max(fake_angles))
    ax.hist(real_angles, bins=30, alpha=0.6, label="Real Shadows", color="green", range=(angle_min, angle_max))
    ax.hist(fake_angles, bins=30, alpha=0.6, label="Fake Shadows", color="red", range=(angle_min, angle_max))
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

def show_flag_histogram():
    st.subheader("Flag Contour Complexity")
    # Run analysis
    real_complexities = np.load("histogram_data/real_flag_complexity.npy")
    fake_complexities = np.load("histogram_data/fake_flag_complexity.npy")
    
    # Plotting the results
    fig, ax = plt.subplots(figsize=(7, 4))
    min_complexity = min(min(real_complexities), min(fake_complexities))
    max_complexity = max(max(real_complexities), max(fake_complexities))
    ax.hist(real_complexities, bins=20, alpha=0.6, color='purple', label='Real Flags', range=(min_complexity, max_complexity))
    ax.hist(fake_complexities, bins=20, alpha=0.6, color='red', label='Fake Flags', range=(min_complexity, max_complexity))
    ax.set_xlabel('Contour Complexity')
    ax.set_ylabel('Frequency')
    ax.set_title('Contour Complexity of Flags')
    ax.legend()
    ax.grid(True)
    ax.tight_layout()
    st.pyplot(fig)

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
st.title("AI-Based Moon Landing Image Analysis")
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

explanations["Shadow"] = """\\n
- The fake images have shadow angles highly concentrated around 90°, indicating a consistent light direction, so possibly a result of AI or manually configured lighting.
- The real images also cluster near 90°, but with a wider spread, reflecting natural variations in shadow direction due to terrain, camera angle, and real sunlight.
- A few extreme angles in the real images may result from terrain features, multiple reflected lights, or shadow detection errors."""

explanations["Flag"] = """\\n
- Fake flag images tend to cluster around mid-level contour complexity, likely due to simulated flag waving with limited variability.
- Real flag images show a wider distribution of complexity, reflecting natural physical dynamics like fabric motion, lighting, and camera angle variations.
- This distribution suggests that real flag photos are less uniform and more organically complex, while fake ones are more constrained and repetitive."""


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
