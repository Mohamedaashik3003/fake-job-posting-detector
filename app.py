import streamlit as st
import pickle
import requests
from io import BytesIO
from PIL import Image
import pytesseract
from explain import explain_prediction

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="wide"
)

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
# OCR FUNCTION
# =============================
def extract_text_from_image_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return pytesseract.image_to_string(image)
    except:
        return ""

# =============================
# HEADER
# =============================
st.title("🕵️ Fake Job Posting Detector")
st.markdown(
    "Detect **fake job postings** using AI. "
    "You can provide **text**, **image URL**, or **upload an image**."
)

st.divider()

# =============================
# LAYOUT
# =============================
left_col, right_col = st.columns(2)

# =============================
# INPUT SECTION
# =============================
with left_col:
    st.subheader("📥 Input")

    user_input = st.text_area(
        "Text or Image URL",
        height=200,
        placeholder="Paste job description OR image URL here"
    )

    uploaded_image = st.file_uploader(
        "Optional: Upload job image",
        type=["png", "jpg", "jpeg"]
    )

    predict_btn = st.button("🔍 Predict")

# =============================
# PREDICTION LOGIC
# =============================
with right_col:
    st.subheader("📊 Result")

    if predict_btn:
        final_text = ""

        # Priority 1: Uploaded image
        if uploaded_image is not None:
            try:
                final_text = pytesseract.image_to_string(
                    Image.open(uploaded_image)
                )
                with st.expander("📄 Extracted Text (Uploaded Image)"):
                    st.write(final_text)
            except:
                st.error("Unable to read uploaded image")

        # Priority 2: Text or image URL
        elif user_input.strip() != "":
            if user_input.lower().startswith(("http://", "https://")):
                extracted_text = extract_text_from_image_url(user_input)
                if extracted_text == "":
                    st.error("Unable to read image from URL")
                else:
                    final_text = extracted_text
                    with st.expander("📄 Extracted Text (Image URL)"):
                        st.write(extracted_text)
            else:
                final_text = user_input

        else:
            st.warning("Please enter text, image URL, or upload an image")

        # =============================
        # MODEL OUTPUT
        # =============================
        if final_text.strip() != "":
            result = explain_prediction(final_text, model, vectorizer)

            prediction = result["prediction"]
            confidence = result["model_confidence"]
            risk = result["risk_percentage"]

            if prediction == "Fake":
                st.error(f"🚨 Prediction: FAKE JOB")
            else:
                st.success(f"✅ Prediction: REAL JOB")

            st.metric("Model Confidence", f"{confidence}%")
            st.metric("Risk Percentage", f"{risk}%")

            st.subheader("🔑 Key Influencing Words")
            for word in result["keywords"]:
                st.write(f"• {word}")
