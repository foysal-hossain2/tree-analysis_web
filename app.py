import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Page custom layout and design
st.set_page_config(
    page_title="Seasonal Growth & Plant Species Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic and clean UI
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fb;
    }
    h1 {
        color: #1e4620;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .report-text {
        font-size: 14px;
        color: #555555;
        background-color: #e8f5e9;
        padding: 15px;
        border-left: 5px solid #2e7d32;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Project Information Title
st.title("🌿 Seasonal Growth Modulation & Plant Classifier")
st.markdown("### Dhaka Periurban Tree Zone Analysis (FYDP Project)")

# Sidebar Information
st.sidebar.image("https://img.icons8.com/clouds/200/000000/plant_under_sun.png", use_container_width=True)
st.sidebar.header("Project Details")
st.sidebar.info(
    "Eti ekti Final Year Design Project (FYDP) application ja Hidden Markov Model "
    "ebong Feature Extraction er sahajye periurban tree zone classification r "
    "growth modulation visual korte sahajyo kore."
)

# Model 29 Classes defined in tree-cnn.ipynb
CLASS_NAMES = [
    'agave_angustifolia', 'antigonon_leptopus', 'azadirachta_indica', 'baby_bamboo',
    'beach_morning_glory', 'boston_fern', 'boxwood_shrubs', 'buxus_sempervirens',
    'chattim', 'euphorbia_trigona_cactus', 'golden_panda', 'guava_plant',
    'heliconia_psittacorum', 'henna', 'jackfruit', 'murraya_paniculata',
    'natal_plum', 'neem', 'orchid_tree', 'panda_grass',
    'piper_excelsum', 'plumeria_tree', 'rough_cocklebur', 'sago_palm',
    'selaginella_kraussiana', 'sugar_apple', 'ti_plant', 'tree_of_heaven',
    'yellow_walking_iris'
]

# Cache model load to prevent memory leaks
@st.cache_resource
def load_keras_model():
    # Looks for plant_model.h5 or plant_model.keras in the same directory
    try:
        model = tf.keras.models.load_model('plant_model.h5')
    except Exception:
        model = tf.keras.models.load_model('plant_model.keras')
    return model

with st.spinner("Model load hocche... Anugraho kore opekkha korun..."):
    try:
        model = load_keras_model()
        st.sidebar.success("✅ Model Safolbhabe Loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Model load hote somossa hoyeche: {e}")
        st.stop()

# File Uploader Section
st.markdown("#### 📸 Gach ba Patar Chobi Upload Korun")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image beautifully
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Chobi", use_container_width=True)
        
    with col2:
        st.markdown("#### ⏳ Analysis Cholicche...")
        # Preprocessing matching IMAGE_SIZE = (224, 224) from tree-cnn.ipynb
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        
        # Expand dimensions to create batch size of 1
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_tensor)[0]
        
        # Handle softmax vs sigmoid output safety
        if len(predictions) == 1: # Binary fallback safety
            score = tf.nn.sigmoid(predictions[0]).numpy()
            predicted_class = CLASS_NAMES[1] if score > 0.5 else CLASS_NAMES[0]
            confidence = score if score > 0.5 else 1 - score
        else:
            # Apply softmax safety if model output layer doesn't have it explicitly
            exp_preds = np.exp(predictions - np.max(predictions))
            probabilities = exp_preds / exp_preds.sum()
            
            top_index = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[top_index]
            confidence = probabilities[top_index]

        # Present Results beautifully
        st.balloons()
        st.success(f"**Result:** This is likely **{predicted_class.replace('_', ' ').title()}**")
        st.metric(label="Confidence Level Score", value=f"{confidence * 100:.2f}%")

    # Plot top 5 Predictions using Plotly Interactive Charts
    st.markdown("---")
    st.markdown("#### 📊 Prediction Confidence Breakdown (Top 5)")
    
    if len(predictions) > 1:
        top_5_idx = np.argsort(probabilities)[-5:][::-1]
        top_5_labels = [CLASS_NAMES[i].replace('_', ' ').title() for i in top_5_idx]
        top_5_scores = [probabilities[i] * 100 for i in top_5_idx]

        fig = go.Figure(go.Bar(
            x=top_5_scores,
            y=top_5_labels,
            orientation='h',
            marker=dict(color='#388e3c')
        ))
        fig.update_layout(
            xaxis_title="Confidence Percentage (%)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=50, r=50, t=30, b=30),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # App landing info box
    st.info("💡 Prothome upore chobi upload korun jeno model seti predict korte pare.")
    st.markdown(
        "<div class='report-text'>"
        "<strong>Project Background (FYDP Report Summary):</strong><br>"
        "Dhaka periurban tree zone a rtu-bhed (seasonal) gachpalar briddhi nirupor, ecosystem cascade effects "
        "ebong precise Deep Learning / CNN models babohar kore chobi theke gach chenar "
        "jonno ai system toiri kora hoyeche."
        "</div>", 
        unsafe_allow_html=True
    )