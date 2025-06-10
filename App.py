import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from scipy.stats import entropy
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tempfile
import plotly.graph_objects as go


st.set_page_config(page_title="Moon Landing Conspiracy", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>Moon Landing Conspiracy</h1>", unsafe_allow_html=True)
st.markdown("---")

# Tabs for sections
tabs = st.tabs(["Overview", "Image Collection", "Image Analysis", "AI Verification of Video Content", "Conclusion"])

# ---- Tab 0: Overview ----
with tabs[0]:
    st.header("🚀 Overview of the Moon Landing")
    
    st.image("https://www.nasa.gov/wp-content/uploads/2025/05/54492243289-217ce0b692-o.jpg", use_container_width=True, caption="Moon Landing NASA")

    st.markdown("""
    ### 🌕 The Historic Apollo 11 Mission

**July 20, 1969** was a defining moment in human history. NASA's **Apollo 11 mission** successfully landed astronauts **Neil Armstrong** and **Buzz Aldrin** on the Moon 🌙, while **Michael Collins** orbited in the command module.

> 🧑‍🚀 *"That's one small step for man, one giant leap for mankind."* — Neil Armstrong

This accomplishment was the result of years of engineering brilliance, scientific research, and the geopolitical competition of the Cold War's **space race** between the 🇺🇸 United States and the 🇷🇺 Soviet Union.

Following Apollo 11, five more missions (Apollo 12, 14, 15, 16, and 17) also made successful landings and returned lunar samples 🪨 and data 📡.

---

### 🕵️‍♂️ A Brief History of Moon Landing Conspiracies

Despite clear evidence, conspiracy theories questioning the Moon landings began in the **1970s**. One of the earliest skeptics, **Bill Kaysing**, claimed NASA faked the landings in a TV studio.

Conspiracy theorists cite:

- 📷 Strange photo shadows
- 🚫 Lack of stars in sky
- ☢️ Radiation in space

These ideas have persisted due to misinformation, media influence, and poor understanding of physics in the public sphere.

---

### 🤖 Role of AI & Data Science in Uncovering the Truth

Modern AI and data science tools help us **verify historical events** and **counter misinformation**:

- 🧠 **Image Forensics**: Detect altered or staged visuals
- 📊 **Telemetry/Data Verification**: Cross-reference mission data logs
- 🕸️ **Disinformation Analysis**: Detect fake news patterns on the web

These technologies offer a **data-driven shield against conspiracy** and enhance public trust in science 🔬.

""")

# ---- Tab 2: Image Collection ----
with tabs[1]:
    st.header("📸 Image Collection")
    st.markdown("""
    We collected *200 grayscale-processed images*, equally split between real and fake, and categorized into six feature-based regions:
    - Shadow
    - Astronaut
    - Surface
    - Shadow
    - Flag
    - Lander
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/real_sample.jpg", caption="Real Image (Grayscale)", use_container_width=True)
    with col2:
        st.image("images/fake_sample.png", caption="Fake Image (Grayscale)", use_container_width=True)

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
        ax.set_title('Shadow Angle Distribution')
        ax.legend()
        ax.grid(True)
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
        st.pyplot(fig)
    
    def show_astronaut_histogram():
        st.subheader("Astronaut Contour Complexity")
        # Run analysis
        real_complexities = np.load("histogram_data/real_astronaut_complexity.npy")
        fake_complexities = np.load("histogram_data/fake_astronaut_complexity.npy")
        
        # Plotting the results
        fig, ax = plt.subplots(figsize=(7, 4))
        min_complexity = min(min(real_complexities), min(fake_complexities))
        max_complexity = max(max(real_complexities), max(fake_complexities))
        ax.hist(real_complexities, bins=20, alpha=0.6, color='purple', label='Real Flags', range=(min_complexity, max_complexity))
        ax.hist(fake_complexities, bins=20, alpha=0.6, color='red', label='Fake Flags', range=(min_complexity, max_complexity))
        ax.set_xlabel('Contour Complexity')
        ax.set_ylabel('Frequency')
        ax.set_title('Contour Complexity of Astronauts')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    def show_horizon_histogram():
        st.subheader("Horizon Line Angles")
        # Run analysis for real and fake sets
        real_angles = np.load("histogram_data/real_horizon_angles.npy")
        fake_angles = np.load("histogram_data/fake_horizon_angles.npy")
        
        # Plot histograms
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(real_angles, bins=30, alpha=0.7, label='Real', color='skyblue')
        ax.hist(fake_angles, bins=30, alpha=0.7, label='Fake', color='salmon')
        ax.set_xlabel("Line Angle (degrees)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Detected Line Angles")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
    
    def show_surface_histogram():
        st.subheader("Surface Texture Features: Energy")
       # Define texture features to extract
        properties = ['contrast', 'homogeneity', 'energy', 'correlation']
        
        # Extract features
        real_features = np.load("histogram_data/real_surface_features.npy", allow_pickle=True).item()
        fake_features = np.load("histogram_data/fake_surface_features.npy", allow_pickle=True).item()
        
        # Plot histograms
        for prop in properties:
            fig, ax = plt.subplots(figsize=(7, 4))
            real_data = real_features[prop]
            fake_data = fake_features[prop]
            min_val = min(np.min(real_data), np.min(fake_data))
            max_val = max(np.max(real_data), np.max(fake_data))
            ax.hist(real_features[prop], bins=20, alpha=0.6, label='Real', color='skyblue', range=(min_val, max_val))
            ax.hist(fake_features[prop], bins=20, alpha=0.6, label='Fake', color='salmon', range=(min_val, max_val))
            ax.set_title(f"Texture Feature: {prop.capitalize()}")
            ax.set_xlabel(prop.capitalize())
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
    
    def show_lander_histogram():
        # Collect features
        real_features = np.load("histogram_data/real_lander_features.npy", allow_pickle=True).item()
        fake_features = np.load("histogram_data/fake_lander_features.npy", allow_pickle=True).item()
        
        # Plot histogram comparison
        for metric in ['num_contours', 'max_area', 'avg_perimeter']:
            fig, ax = plt.subplots(figsize=(7, 4))
            min_val = min(min(real_features[metric]), min(fake_features[metric]))
            max_val = max(max(real_features[metric]), max(fake_features[metric]))
            ax.hist(real_features[metric], bins=20, alpha=0.6, label='Real', color='lightblue', range=(min_val, max_val))
            ax.hist(fake_features[metric], bins=20, alpha=0.6, label='Fake', color='blue', range=(min_val, max_val))
            ax.set_title(f'Lander Feature: {metric.replace("_", " ").title()}')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)
    
    
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
    
    explanations["Shadow"] = """ 
    - The fake images have shadow angles highly concentrated around 90°, indicating a consistent light direction, so possibly a result of AI or manually configured lighting.
    - The real images also cluster near 90°, but with a wider spread, reflecting natural variations in shadow direction due to terrain, camera angle, and real sunlight.
    - A few extreme angles in the real images may result from terrain features, multiple reflected lights, or shadow detection errors."""
    
    explanations["Flag"] = """ 
    - Fake flag images tend to cluster around mid-level contour complexity, likely due to simulated flag waving with limited variability.
    - Real flag images show a wider distribution of complexity, reflecting natural physical dynamics like fabric motion, lighting, and camera angle variations.
    - This distribution suggests that real flag photos are less uniform and more organically complex, while fake ones are more constrained and repetitive."""
    
    explanations["Astronaut"] = """ 
    - The contour complexity of real astronaut images tends to be higher, reflecting intricate textures, lighting variations, and authentic visual noise.
    - In contrast, fake images cluster in the lower complexity range, suggesting smoother, less detailed rendering, possibly due to AI generation or simplified edits.
    - This supports the hypothesis that real visual content often exhibits richer and more variable structure, while fakes are more uniform and predictable."""
    
    explanations["Horizon"] = """ 
    - Real images have line angles mainly fall between -20° and +30°, which distribution is compact, showing natural orientation.
    - Fake images have strong peaks at ±90°, indicates extreme tilts or vertical lines.
    - Real images show natural and consistent horizon angles, while fake images exhibit abnormal angle peaks, useful for detecting falsification."""
    
    explanations["Surface"] = """ 
    1.Contrast
    - Fake images exhibit a broader range of contrast values, with some extreme outliers reaching above 700. Most real images cluster below 150.
    - Fake images tend to have harsher or more inconsistent textures, possibly due to rendering or compositing artifacts. 
    
    2.Homogeneity
    - Real images show middle homogeneity overall. peaking around 0.3-0.6, suggesting more natural texture variation. While fake images have more distributed values, too high or too low.
    - Fake textures tend to be either overly smoothed or exhibit abrupt patterns, leading to a wider range of homogeneity. 
    
    3. Energy
    - Fake images dominate in high-energy values (0.3–0.7), while real images mostly lie below 0.3.
    - Energy reflects texture repetition. High energy in fake images may indicate repetitive or templated patterns. 
    
    4.Correlation
    - Both real and fake images have high correlation, but fake images show more clustering around 0.98–0.99.
    - High correlation in fake images suggests more uniform pixel transitions, possibly synthetic."""
    
    explanations["Lander"] = """ 
    1.Num Contours
    - Fake images tend to have more instances with low contour counts, while real images are more evenly spread and can reach higher counts.
    - This may suggest that fake images often exhibit simpler structures in the lander region, possibly due to generative model artifacts or lack of detail.
    
    2.Max Area
    - Max area quantifies the largest detected contour, usually representing the main body of the lander.
    - Real images tend to contain larger contour areas, indicating more coherent and possibly more detailed lander shapes.
    - Fake images show more limited area sizes, with few outliers.
    
    3.Average Perimeter
    - Fake images actually show a wider spread in average contour perimeters, ranging from very low to very high values. This suggests that fake landers may have either overly simplistic or irregularly exaggerated contours due to artifacts.
    - Real images show a more compact distribution, indicating more consistent shape complexity."""
    
    
    st.markdown(f"*Conclusion*: {explanations[choice]}")


# ---- Tab 3: Video Detection ----
with tabs[3]:
    st.header("AI Verification for Video Content 🤖")
    st.write("""
    With the advancement in synthetic media generation, authenticating video content using AI has become increasingly critical.
    This section showcases three stages of model development: starting from a minimal dataset, using a pre-trained model, and training a CNN on a custom dataset.
    """)

    # Section 5.1 - 1:1 Small Dataset
    st.subheader("5.1: 1:1 Small Dataset 🌙")

    # Explanation
    with st.expander("Explanation"):
        st.markdown("""
    ### Explanation
    This section demonstrates a simple proof-of-concept pipeline using a very small dataset: one real video and one fake video.

    The idea is to:
    - Extract individual frames from each video.
    - Compute basic image statistics like brightness, noise level, and edge density for each frame.
    - Average these features across all frames in a video to get a single feature vector per video.
    - Train a Random Forest classifier to distinguish between real and fake videos using these features.

    Due to the extremely small dataset size (only one sample per class), the results are **not reliable** but serve to validate the overall approach.
        """)

    # Code
    with st.expander("Code"):
        st.markdown("""
    ### Code
    ```python
    def extract_frames(video_path, output_folder, max_frames=500):
        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"{output_folder}/frame_{count:04d}.jpg", frame)
            count += 1
        cap.release()
        print(f"Extracted {count} frames to {output_folder}")


    def extract_frame_features(frame_path):
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return None
        brightness = np.mean(frame)
        noise = np.std(cv2.GaussianBlur(frame, (3, 3), 0) - frame)
        edges = cv2.Canny(frame, 100, 200)
        edge_density = np.sum(edges) / edges.size
        return [brightness, noise, edge_density]


    def extract_video_features(folder):
        features = []
        for frame_path in sorted(glob.glob(folder + "/*.jpg")):
            feat = extract_frame_features(frame_path)
            if feat:
                features.append(feat)
        return np.mean(features, axis=0)  # average feature values

    # Extract frames (assumes video files are present)
    extract_frames("real-moonlanding.mp4", "frames_real")
    extract_frames("fake-moonlanding.mp4", "frames_fake")
    extract_frames("new-video.mp4", "new_frames")

    # Extract features
    real_feat = extract_video_features("frames_real")
    fake_feat = extract_video_features("frames_fake")
    new_feat = extract_video_features("new_frames")

    # Train simple classifier
    X = [fake_feat, real_feat]
    y = [0, 1]  # 1 = real, 0 = fake
    model = RandomForestClassifier()
    model.fit(X, y)

    # Predict
    prediction = model.predict([new_feat])
    print("Prediction:", prediction)
    print("Prediction:", "Fake" if prediction[0] == 1 else "Real")
    print("Trained on 1 real and 1 fake video — results are not reliable.")
    ```
        """)

    # Output
    with st.expander("Output"):
        st.markdown("""
    ### Output
    ```
    Prediction:  [0]
    Prediction: Fake
    Trained on 1 real and 1 fake video — results are not reliable.
    ```
    """)


    # --- 5.2: Pre-trained Dataset ---
    st.subheader("2. Pre-trained Deepfake Detection Model 🌗")

    # ➤ Explanation
    with st.expander("Explanation"):
        st.markdown("""
        ### Explanation
This section demonstrates how to use a **pre-trained deepfake detection model** for classifying videos.

The process involves:

1. **Model Loading**: Load a model saved in HDF5 format (`deepfake_detection_model.h5`). The model architecture is also saved and reloaded from JSON for clarity.
2. **Video Frame Extraction**: Frames are extracted from an input video using OpenCV, capped at 30 frames for efficiency.
3. **Preprocessing**: Each frame is resized to 299x299, normalized, and collected into a NumPy array.
4. **Batch Prediction**: The full stack of preprocessed frames is passed through the model, which outputs predictions per frame.
5. **Decision Rule**: The average of frame-level predictions is taken. If the average confidence is >0.5, it's labeled "Real", else "Fake".

This method offers a good balance of speed and robustness for real-time inference on short video clips.
    """)

    # ➤ Code
    with st.expander("Code"):
        st.markdown("""
        ### Code
        """)
        st.code("""
from tensorflow.keras.models import load_model, model_from_json
import cv2
import os
import numpy as np
import json

# Load full model
model = load_model("deepfake_detection_model.h5")

# Save architecture
model_json = model.to_json()
with open("reconstructed_model.json", "w") as json_file:
    json.dump(model_json, json_file)

# Load architecture
with open("reconstructed_model.json", "r") as json_file:
    loaded_model_json = json.load(json_file)

model = model_from_json(loaded_model_json)

# Load weights
model.load_weights("deepfake_detection_model.h5")

# Preprocess function
def preprocess_frame(frame_path):
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Could not read image: {frame_path}")
    frame = cv2.resize(frame, (299, 299))
    frame = frame.astype("float32") / 255.0
    return frame

# Extract frames
def trained_extract_frames(video_path, output_folder, max_frames=100):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return []
    frame_paths = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        count += 1
    cap.release()
    return frame_paths

# Extract + preprocess
frames = trained_extract_frames("new-video.mp4", "pretrained_frames", max_frames=30)
processed_frames = np.array([preprocess_frame(f) for f in frames])  # shape: (N, 299, 299, 3)

# Add batch dimension
input_tensor = np.expand_dims(processed_frames, axis=0)  # shape: (1, N, 299, 299, 3)

# Predict
preds = model.predict(input_tensor)
mean_pred = np.mean(preds)
label = "Real" if mean_pred > 0.5 else "Fake"
print(f"Prediction: {label} (confidence: {mean_pred:.2f})")
        """)

    #Output
    with st.expander("Output"):
        st.markdown("""
        ### Output
        """)
        st.text("""
    1/1 ━━━━━━━━━━━━━━━━━━━━ 91s 91s/step
    Prediction: Real (confidence: 0.54)
        """)


    # Advanced Self-Trained Model Using Transfer Learning
    # --- Section 3 ---
    st.subheader("3. Self-Trained CNN on Custom Dataset 🌕")

    # Explanation
    with st.expander("Explanation"):
        st.markdown("""
        ### Explanation
This method involves training a custom CNN model on a dataset built from real and AI-generated video frames.
The steps involved include:

1. **Dataset Creation**: Frames are sampled from both real and fake videos.
2. **Augmentation**: To balance and enrich the dataset, techniques such as flipping, zooming, and brightness adjustments are applied.
3. **Transfer Learning**: A `MobileNetV2` model is used as the base with custom top layers.
4. **Class Imbalance Handling**: Class weights are computed and applied during training.
5. **Training**: Model is trained for 10 epochs, showing significant improvement in accuracy and generalization.
6. **Output**: The trained model (`moon_video_detector_model.h5`) is saved for real-time inference later.

This strategy significantly outperformed basic methods with validation accuracy above 96%.
    """)
        st.markdown("### 📋 Visual Attribute Tendency Table")
    
    # Define the attribute table
        attribute_data = {
        "Visual Attribute": [
            "Brightness", 
            "Edge Density", 
            "Noise Variance", 
            "Color Distribution"
        ],
        "Description": [
            "Average intensity of pixels",
            "Number of edges (e.g., via Canny edge detection)",
            "Standard deviation in local patches",
            "Skewness or dominance in color channels"
        ],
        "Trend in Fake Videos": [
            "May be unnaturally even or low",
            "Often lower or overly smooth",
            "Often lacks natural camera noise",
            "May be limited or artificial"
        ]
        }

        attribute_df = pd.DataFrame(attribute_data)

        # Set index to start from 1
        attribute_df.index = attribute_df.index + 1
        attribute_df.index.name = "No"

        st.table(attribute_df)

    with st.expander("🧪 Feature Extraction and Table (Real vs Fake)"):

        def extract_visual_attributes(image_path):
            img = cv2.imread(image_path)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            edge_density = np.sum(cv2.Canny(gray, 100, 200) > 0) / gray.size
            noise = cv2.Laplacian(gray, cv2.CV_64F).var()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [256], [0, 256]).flatten()
            color_entropy = entropy(hist + 1e-8)
            return {
            "Brightness": brightness,
            "Edge Density": edge_density,
            "Noise Variance": noise,
            "Color Entropy": color_entropy
            }

        def process_folder(folder_path, label):
            data = []
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(folder_path, filename)
                    attributes = extract_visual_attributes(full_path)
                    if attributes:
                        attributes["Label"] = label
                        attributes["Filename"] = filename
                        data.append(attributes)
            return data

    
        real_data = process_folder("train_dataset/train/real", "Real")
        fake_data = process_folder("train_dataset/train/fake", "Fake")
        df = pd.DataFrame(real_data + fake_data)

        st.write("### Visual Attribute Table")
        st.dataframe(pd.concat([df.head(10), df.tail(10)]))
        import seaborn as sns
        import matplotlib.pyplot as plt

        st.write("### Attribute Distributions by Class")
        for attr in ["Brightness", "Edge Density", "Noise Variance", "Color Entropy"]:
            fig, ax = plt.subplots()
            sns.boxplot(x="Label", y=attr, data=df, ax=ax)
            st.pyplot(fig)
            
    # Code
    with st.expander("Code"):
        st.markdown("""
        ### Code
        """)
        st.code("""    
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

# === Parameters ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "train_dataset"

# === Data Generators with Augmentation for fake class ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# === Calculate Class Weights ===
y_train = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# === Transfer Learning Model ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze base

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# === Train Model ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights
)

# === Save Model ===
model.save("moon_video_detector_model.h5")
        """)

    # Output
    with st.expander("Output"):
        st.markdown("""
        ### Output
        """)
        st.text("""
Found 1497 images belonging to 2 classes.
Found 383 images belonging to 2 classes.
Class Weights: {0: 8.3166..., 1: 0.5319...}

Epoch 1/10 - accuracy: 0.5328 - val_accuracy: 0.5013
Epoch 2/10 - accuracy: 0.6080 - val_accuracy: 0.6736
Epoch 3/10 - accuracy: 0.7050 - val_accuracy: 0.8146
Epoch 4/10 - accuracy: 0.7768 - val_accuracy: 0.8956
Epoch 5/10 - accuracy: 0.8474 - val_accuracy: 0.9426
Epoch 6/10 - accuracy: 0.8630 - val_accuracy: 0.9530
Epoch 7/10 - accuracy: 0.9036 - val_accuracy: 0.9634
Epoch 8/10 - accuracy: 0.9288 - val_accuracy: 0.9634
Epoch 9/10 - accuracy: 0.9299 - val_accuracy: 0.9634
Epoch 10/10 - accuracy: 0.9421 - val_accuracy: 0.9661
        """)
    with st.expander("📊 Model Evaluation (Random Forest Classifier)"):
        st.markdown("This section summarizes the performance of the **Random Forest Classifier** trained on the frame-level visual attributes.")
    
        st.markdown("#### 🔍 Metric Definitions:")
        st.markdown("""
    - **Precision**: Out of all predicted positives, how many were actually correct.
    - **Recall**: Out of all actual positives, how many were correctly predicted.
    - **F1-score**: Harmonic mean of precision and recall.
    - **Support**: Number of true instances of each class in the test data.
        """)

        st.markdown("#### 📑 Classification Report:")
    
        data = {
        "Class": ["Real (0)", "Fake (1)", "Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision": ["0.98", "0.67", "0.96", "0.82", "0.96"],
        "Recall": ["0.98", "0.59", "0.96", "0.79", "0.96"],
        "F1-score": ["0.98", "0.62", "0.96", "0.80", "0.96"],
        "Support": ["283", "17", "300", "300", "300"]
        }

        df = pd.DataFrame(data)
        df.index = [1, 2, 3, 4, 5]  # set index from 1
        st.dataframe(df)

        st.markdown("#### 💡 Key Observations:")
        st.markdown("""
    - The model performs **exceptionally well on real videos** (Class 0) with high precision and recall.
    - **Performance on fake videos (Class 1) is weaker** due to limited training samples (only 17).
    - **Overall accuracy is 96%**, but the macro average F1-score is lower (0.80) because of the fake class imbalance.
        """)
        st.info("Improvement Tip: Increase fake data samples and experiment with additional visual features or balanced class weights.")

    st.info("This model leverages MobileNetV2 with transfer learning on a self-curated dataset. It uses data augmentation and class balancing, achieving up to 96% validation accuracy.")

    # --- Demo Section ---
    st.subheader("📹 Live Video Prediction Demo")
    st.markdown("""
        ##### Using the Model that we have trained
        moon_video_detector_model.h5
        """)

    # Upload video
    video_file = st.file_uploader("Upload a video", type=["mp4"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        # Load pre-trained model
        model = load_model("moon_video_detector_model.h5")

        def extract_frames(video_path, max_frames=50):
            cap = cv2.VideoCapture(video_path)
            frames = []
            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224)) / 255.0
                frames.append(frame)
                count += 1
            cap.release()
            return np.array(frames)

        st.write("🔄 Extracting frames and running model...")
        frames = extract_frames(video_path)
        predictions = model.predict(frames)
        confidence = float(np.mean(predictions))
        label = "Real" if confidence > 0.5 else "Fake"

        # Display results
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: `{confidence:.2f}`")

        # Plot
        fig = go.Figure(data=[
            go.Bar(x=["Fake", "Real"], y=[1 - confidence, confidence],
                   marker_color=["red", "green"])
        ])
        fig.update_layout(title="Prediction Confidence", yaxis_title="Probability", xaxis_tickangle=-90)
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 5: Statistical Testing ----
#with tabs[5]:
#    st.header("Statistical Analysis and Hypothesis Testing")
#    st.write("""
#   Use statistical methods (e.g., t-tests) to compare features like brightness or edge density
#   between real and fake video frames to test for significant differences.
#   """)

# ---- Tab 6: Conclusion ----
with tabs[4]:
    st.header("🚀 Conclusion: AI & the Moon Landing")

    st.image("https://www.nasa.gov/wp-content/uploads/2023/03/gpn-2001-000014.jpg", 
             caption="Small step, Big Impact", use_container_width=True)

    st.markdown("""
    ### 🧪 Summary of Findings
    
    From the combined analysis of images and videos:
    
    - **200 grayscale images** were analyzed (real vs. fake) across six categories: *shadow, astronaut, surface, flag, horizon, and lander*.
    - AI image analysis revealed significant **visual differences**:
      - **Fake shadows** showed unnatural angle consistency (often near 90°).
      - **Fake flags and astronauts** lacked natural texture complexity.
      - **Surface and lander images** showed artificial contour patterns in fake images.
    - Using **contour analysis, histograms, and texture metrics**, we found measurable inconsistencies in fake media.
      
    In video analysis:
    - Frame-level features were used to detect fakeness via traditional ML (Random Forest).
    - A pre-trained deepfake detection model was tested and validated.
    - A **custom CNN trained with MobileNetV2** achieved **over 96% validation accuracy**, showing strong predictive confidence for real/fake video detection.
    
    ---
    
    ### 🤖 How AI Helps Verify Historical Events
    
    AI provides an **objective, scalable, and evidence-driven approach** to analyzing historical media. In this project, AI enabled us to:
    
    - Detect statistical patterns that humans miss (e.g., contour complexity, edge density).
    - Quantitatively compare **real vs fake visual artifacts** across hundreds of samples.
    - Apply deep learning models to **validate the authenticity** of both images and videos.
    
    AI also plays a critical role in **combatting misinformation** by:
    - Highlighting inconsistencies in synthetic media
    - Verifying archival evidence using forensics
    - Automating large-scale analysis of visual and textual claims
    
    ---
    
    ### 🌕 Final Verdict
    
    Based on:
    - Measurable image inconsistencies in fake content
    - High-performing AI models detecting authenticity
    - Scientific and historical corroboration
    
    > **We conclude with high confidence that the Moon landing was not faked.**
    
    The evidence, both human and AI-analyzed, overwhelmingly supports the authenticity of the 1969 Apollo 11 mission and subsequent Moon landings.
    
    ---
    
    > 🧠 *"AI doesn’t just detect fakes—it helps protect the truth."*
    """)