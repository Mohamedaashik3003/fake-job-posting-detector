import streamlit as st
import pickle
import requests
from io import BytesIO
from PIL import Image
import pytesseract
from explain import explain_prediction
from datetime import datetime

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
# OCR FUNCTION (IMAGE URL)
# =============================
def extract_text_from_image_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            return ""
        image = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except:
        return ""

# =============================
# GENERATE REPORT FUNCTION
# =============================
def generate_report(input_type, input_text, result, job_title="Not Provided"):
    date_now = datetime.now().strftime("%d-%m-%Y")

    report = f"""
FAKE JOB POSTING DETECTOR – AUTOMATIC ANALYSIS REPORT

Job Posting Analysis Report

Job Title: {job_title}
Platform Source: {input_type}
Date of Analysis: {date_now}

Prediction Result

Job Authenticity Status:
{"Fake Job" if result["prediction"] == "Fake" else "Real Job"}

Risk Percentage:
{result["risk_percentage"]}% probability of being Fake

Model Confidence:
The model is {result["model_confidence"]}% confident in this prediction.

Summary

This job posting was analyzed using a machine learning–based Fake Job Posting Detector.
The system examined the job description text obtained from user input and identified
patterns commonly associated with fraudulent job advertisements.

Key Indicators Identified
"""

    for word in result["keywords"]:
        report += f"- {word}\n"

    report += """
Model Details

Text Processing:
Stopword removal, punctuation removal, lowercase conversion

Feature Extraction:
TF-IDF Vectorization

Model Used:
Random Forest Classifier

Prediction Type:
Probability-based classification

Risk Interpretation

0–10% → Very Low Risk (Highly Likely Real Job)
11–30% → Low Risk (Likely Real Job)
31–60% → Medium Risk (Caution Advised)
61–100% → High Risk (Likely Fake Job)

Conclusion

Based on the analysis, users are advised to verify company details, avoid sharing
personal information, and proceed cautiously if the risk percentage is high.
"""
    return report

# =============================
# HEADER
# =============================
st.title("🕵️ Fake Job Posting Detector")
st.markdown(
    """
Detect **fake job postings** using Machine Learning.

**You can provide:**
- 📝 Job description text  
- 🌐 Job post URL (LinkedIn, Indeed, etc.)  
- 🖼 Upload a job image (poster / screenshot)
"""
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

    # Job Title Input
    job_title = st.text_input(
        "Job Title (optional, will appear in report)",
        placeholder="Enter job title here"
    )

    # Main Input: Text or URL
    user_input = st.text_area(
        "Text or Job Post URL",
        height=200,
        placeholder="Paste job description OR job post URL here"
    )

    # Optional uploaded image
    uploaded_image = st.file_uploader(
        "Optional: Upload job image",
        type=["png", "jpg", "jpeg"]
    )

    predict_btn = st.button("🔍 Predict")

# =============================
# RESULT SECTION
# =============================
with right_col:
    st.subheader("📊 Result")

    if predict_btn:
        final_text = ""

        # Priority 1: Uploaded Image
        if uploaded_image is not None:
            try:
                final_text = pytesseract.image_to_string(
                    Image.open(uploaded_image)
                )
                with st.expander("📄 Extracted Text (Uploaded Image)"):
                    st.write(final_text)
            except:
                st.error("❌ Unable to read uploaded image")

        # Priority 2: Text or Job Post URL
        elif user_input.strip() != "":
            final_text = user_input  # Direct prediction from text or URL

        else:
            st.warning("⚠️ Please enter text, job URL, or upload an image")

        # MODEL PREDICTION
        if final_text.strip() != "":
            result = explain_prediction(final_text, model, vectorizer)

            prediction = result["prediction"]
            confidence = result["model_confidence"]
            risk = result["risk_percentage"]

            if prediction == "Fake":
                st.error("🚨 Prediction: FAKE JOB")
            else:
                st.success("✅ Prediction: REAL JOB")

            st.metric("Model Confidence", f"{confidence}%")
            st.metric("Risk Percentage", f"{risk}%")

            st.subheader("🔑 Key Influencing Words")
            for word in result["keywords"]:
                st.write(f"• {word}")

            # GENERATE AND SHOW REPORT
            report_text = generate_report(
                input_type="Text / Image / URL",
                input_text=final_text,
                result=result,
                job_title=job_title if job_title.strip() != "" else "Not Provided"
            )

            st.subheader("📝 Automatic Job Analysis Report")
            st.text(report_text)

            st.download_button(
                label="Download Analysis Report",
                data=report_text,
                file_name="Fake_Job_Analysis_Report.txt",
                mime="text/plain"
            )
