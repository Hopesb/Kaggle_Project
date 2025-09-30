import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ---------------------------
# Configuration
# ---------------------------
model_path = "vit_model.pth"  # Change if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

# Class labels
class_labels = [
    "dog", "horse", "elephant", "butterfly", "chicken", 
    "cat", "cow", "sheep", "spider", "squirrel",
]

# ---------------------------
# Model Setup
# ---------------------------
@st.cache_resource
def load_model():
    vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit_in_features = vit_model.heads.head.in_features

    for param in vit_model.parameters():
        param.requires_grad = False

    classifier = torch.nn.Sequential(
        torch.nn.Linear(vit_in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(256, 10),
    )

    vit_model.heads.head = classifier
    vit_model.to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        vit_model.load_state_dict(checkpoint["model_state_dict"])
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load model: {e}")

    vit_model.eval()
    return vit_model

model = load_model()

# ---------------------------
# Preprocessing
# ---------------------------
class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

mean, std = ([0.5179, 0.5005, 0.4128], [0.2660, 0.2611, 0.2786])

transform_norm = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Prediction function
def predict(model, img, device, class_labels):
    img = transform_norm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    result = {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}
    return result

# ---------------------------
# App Layout
# ---------------------------
st.set_page_config(
    page_title="Philip Code Academy — Image Classification Demo",
    page_icon="🐶",
    layout="centered"
)

# Sidebar Navigation
st.sidebar.title("📍 Navigation")
section = st.sidebar.radio(
    "Go to", 
    ["🏠 Home", "🧠 About Project", "📊 Try the Model", "🎓 Learn", "📞 Contact"]
)

# ---------------------------
# Home
# ---------------------------
if section == "🏠 Home":
    st.title("🐾 Welcome to Philip Code Academy!")
    st.markdown("""
    ### 👋 Greetings!
    This application demonstrates a **real-time Image Classification Model** built with PyTorch and Vision Transformer (ViT).  

    🚀 At **Philip Code Academy**, we groom minds in:
    - Python Programming  
    - Data Analysis (Python, SQL, Power BI)  
    - Data Science & Machine Learning  
    """)
    st.markdown("---")
    st.info("💡 Explore other pages using the sidebar navigation.")

# ---------------------------
# About Project
# ---------------------------
elif section == "🧠 About Project":
    st.title("🧠 Image Classification with CNNs and Transfer Learning")
    st.markdown("""
    **Author:** SHODOLAMU OPEYEMI PHILIP

    ### 📌 1. Introduction  
    Image classification is a core task in computer vision. It involves training models to categorize images into predefined classes.

    Real-world applications:
    - Wildlife research and species identification  
    - Automated farm animal monitoring  
    - Smart gallery categorization  

    ### 📂 Dataset & Objective  
    - ~25,000 animal images across 10 categories  
    - Goal: Build a model that accurately classifies them  

    ### 📊 Expected Outcomes  
    - A robust model for animal classification  
    - Comparative analysis of architectures  
    - High classification accuracy

    ### 📈 Evaluation Metrics  
    - Accuracy  
    - Confusion Matrix  
    - Loss & Accuracy Curves
    """)

# ---------------------------
# Try the Model
# ---------------------------
elif section == "📊 Try the Model":
    st.title("📊 Try the Animal Classifier")
    st.write("Upload an image and let the model classify it into one of the 10 animal categories.")

    uploaded_file = st.file_uploader("📤 Upload an animal image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", width="content")

        if st.button("🔍 Classify"):
            with st.spinner("Classifying... ⏳"):
                result = predict(model, image, device, class_labels)
                sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

                st.success("✅ Classification Complete!")
                top_label = max(sorted_result, key=sorted_result.get)
                st.markdown(f"**🧠 Top Prediction:** {top_label.upper()} ({sorted_result[top_label]*100:.2f}%)")

                st.markdown("### 🔎 Detailed Probabilities:")
                st.bar_chart(sorted_result)
    else:
        st.info("👆 Upload an image to get started.")

# ---------------------------
# Learn
# ---------------------------
elif section == "🎓 Learn":
    st.title("🎓 Learn with Philip Code Academy")
    st.markdown("""
    We offer **industry-ready training** in:
    - Python Programming (Beginner → Advanced)
    - Data Analysis (Python, SQL, Power BI)
    - Data Science & Machine Learning

    **Benefits:**
    - Real-world projects & portfolio-ready work  
    - Certificates upon completion  
    - Mentorship & job placement guidance

    📅 **Next Cohort:** October 2025  
    👉 [Register Here](https://forms.gle/q84rXMSFexkNnSji6)
    """)

# ---------------------------
# Contact
# ---------------------------
elif section == "📞 Contact":
    st.title("📞 Contact & Community")
    st.markdown("""
    - 📧 [Email Us](mailto:hope.sb001@gmail.com)  
    - 💬 [WhatsApp](https://wa.me/2348184801629)  
    """)
    st.markdown("---")
    st.write("© 2025 Philip Code Academy | Empowering Minds with Code")
