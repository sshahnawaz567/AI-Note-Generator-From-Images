import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import base64
from torchvision import models
from io import BytesIO
import os
import random
from llm import get_weather, get_cuisine_info, get_local_attractions, get_wikipedia_info, generate_notes  # Import your functions


# List of class names for your landmarks
class_names = ['Ajanta Caves', 'Ellora Caves', 'Fatehpur Sikri',
       'Gateway of India', 'Hawa mahal', 'Khajuraho', 'Sun Temple Konark',
       'alai_darwaza', 'alai_minar', 'basilica_of_bom_jesus', 'charminar',
       'golden temple', 'iron_pillar', 'jamali_kamali_tomb',
       'lotus_temple', 'mysore_palace', 'qutub_minar', 'tajmahal',
       'tanjavur temple', 'victoria memorial']

# --- Load Model ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = models.densenet121(pretrained=False)
        num_classes = len(class_names)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes))

        # Load model weights
        model.load_state_dict(torch.load("model/DenseNet121(2).pth", map_location=device))    
        model.eval()  # Set the model to evaluation mode
        model.to(device)  # Move model to device
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

# --- Image Preprocessing & Prediction ---
def predict_image(model, device, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Move the image to the correct device

    try:
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            return class_names[predicted.item()]  # Return the predicted class name
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        raise


st.set_page_config(page_title="EduNote 🌍", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        /* Global Styles */
        .main {
            background-color: #f7f7f7;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            font-family: 'Roboto', sans-serif;
            color: #4B8BBE;
            text-align: center;
        }
        h4 {
            font-family: 'Arial', sans-serif;
            color: #4B8BBE;
            text-align: center;
        }
        p {
            font-size: 16px;
            color: #555;
        }
        .stFileUploader {
            border: 2px dashed #4B8BBE;
            padding: 15px;
            margin-top: 10px;
            text-align: center;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #4B8BBE;
            color: white;
            border-radius: 5px;
            padding: 10px 15px;
            font-size: 16px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #357ABD;
        }
        .highlight-section {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .uploaded_image {
            transition: transform 0.2s ease;
        }
        .uploaded_image:hover {
            transform: scale(1.05);
        }
        .note-card {
            background-color: #ffffff;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-content {
            background-color: #4B8BBE;
            color: white;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to convert image to base64 string
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Title and subtitle
st.title("📚✨ EduNote - Discover, Learn, Download!")
st.markdown("""
<h4 style='color: #4B8BBE;'>📷 Upload an image to discover the place 🌍, see a reference photo 🖼️, and read travel notes ✈️</h4>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
option = st.sidebar.selectbox("🔽 Choose an option", ["📤 Image Upload", "📝 Generated Notes", "🏛️ Attractions"])

# Upload Section
st.markdown("## 📤 Upload Your Travel Image")
uploaded_file = st.file_uploader("Upload a photo (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Convert the image to base64 string
    img_base64 = image_to_base64(image)
    
    # Display the uploaded image with custom HTML styling
    st.markdown(f"""
        <div class="uploaded_image">
            <img src="data:image/png;base64,{img_base64}" alt="Uploaded Image" />
        </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("🔍 Analyzing the image... Please wait."):

            model, device = load_model()
            prediction = predict_image(model, device, image)
            formatted_name = prediction.strip().replace(" ", "_").lower()
            place_folder_path = os.path.join("database", formatted_name)

            st.success(f"📍 **Predicted Place:** `{prediction}`")

            # Reference Image
            if os.path.exists(place_folder_path):
                images = [f for f in os.listdir(place_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    random_image = random.choice(images)
                    image_path = os.path.join(place_folder_path, random_image)
                    st.image(image_path, caption=f"🖼️ Reference Image for {prediction}", use_column_width=True)
                else:
                    st.warning("⚠️ No reference images found for this location.")
            else:
                st.error("🚫 Reference folder not found for this location.")

            # Travel Note Section
            st.markdown("### 📝 Travel Note")
            st.markdown("<div class='note-card'>" + generate_notes(prediction) + "</div>", unsafe_allow_html=True)

            # Extra Info Section
            st.markdown("### 🌤️ Weather  | 🧭 Local Attractions")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"🌤️ **Weather**: {get_weather(prediction)}")
                # st.write(f"🍴 **Cuisine**: {get_cuisine_info(prediction)}")
                st.write(f"📍 **Attractions**: {get_local_attractions(prediction)}")

            with col2:
                st.write(f"📚 **Wikipedia**: {get_wikipedia_info(prediction)}")

    except Exception as e:
        st.error(f"❌ Oops! Something went wrong: {str(e)}")

# Sidebar navigation with styled buttons
st.sidebar.markdown("""
    <style>
        .stButton>button {
            background-color: #4B8BBE;
            color: white;
            border-radius: 5px;
            padding: 10px 15px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #357ABD;
        }
    </style>
""", unsafe_allow_html=True)