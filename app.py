import streamlit as st
import pickle
import requests
from io import BytesIO
from PIL import Image
import pytesseract

# =============================
# TESSERACT PATH (Windows)
# =============================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =============================
# LOAD MODEL & VECTORIZER
# =============================
with open("models/fake_job_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =============================
# IMPORT EXPLAIN FUNCTION
# =============================
from explain import explain_prediction

# =============================
# OCR FUNCTION FOR IMAGE URL
# =============================
def extract_text_from_image_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(image)
        return text
    except:
        return ""

# =============================
# STREAMLIT UI
# =============================
st.title("Fake Job Posting Detector")
st.write("Enter **job description text**, paste an **image URL**, or upload an **image**")

# ✅ INPUT HANDLING LIKE CODE 1
user_input = st.text_area("Text or Image URL", height=200)
uploaded_image = st.file_uploader("Optional: Upload Job Image", type=["png", "jpg", "jpeg"])

# =============================
# PREDICT BUTTON
# =============================
if st.button("Predict"):
    final_text = ""

    # Priority 1: Uploaded Image
    if uploaded_image is not None:
        try:
            final_text = pytesseract.image_to_string(Image.open(uploaded_image))
            st.subheader("Extracted Text from Image")
            st.write(final_text)
        except:
            st.error("Unable to read uploaded image")

    # Priority 2: Text or Image URL
    elif user_input.strip() != "":
        if user_input.lower().startswith(("http://", "https://")):
            extracted_text = extract_text_from_image_url(user_input)
            if extracted_text == "":
                st.error("Unable to read image from URL")
            else:
                st.subheader("Extracted Text from Image URL")
                st.write(extracted_text)
                final_text = extracted_text
        else:
            final_text = user_input

    else:
        st.warning("Please enter text, image URL, or upload an image")

    # =============================
    # MODEL PREDICTION
    # =============================
    if final_text.strip() != "":
        result = explain_prediction(final_text, model, vectorizer)

        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {result['prediction']}")
        st.write(f"**Model Confidence:** {result['model_confidence']}%")
        st.write(f"**Risk Percentage:** {result['risk_percentage']}%")

        st.subheader("Key Influencing Words")
        for word in result["keywords"]:
            st.write(f"- {word}")

