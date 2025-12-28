# Fake Job Posting Detector

This project aims to detect fake job postings using machine learning
and natural language processing techniques.

## Day 1: Problem & Dataset
The problem of fake job postings is analyzed and a suitable dataset
is identified for further development.

### Dataset
The Fake Job Postings dataset from Kaggle is selected for this project:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Due to GitHub file size limitations, the dataset is not included in this
repository. Users should download the dataset manually and place the CSV
file inside the `data/` folder.

## Day 2: Text Preprocessing
Raw job description text was cleaned by:
- Lowercasing
- Removing punctuation
- Removing stopwords

A processed dataset was generated for future machine learning tasks.

## Day 3 – Feature Extraction
TF-IDF is used to convert cleaned job description text into numerical
features for machine learning models.

## Day 4 – Model Training
Trained machine learning models such as Logistic Regression and
Naive Bayes to classify fake and real job postings.

## Day 5 – Explainability & Risk Scoring
Implemented explanation logic to identify influential keywords and
compute a risk score for fake job postings.

## Day 6 – Real-Time Web App
Developed a Streamlit web interface to accept job description text, image URL, or image upload and display instant predictions with model confidence, risk percentage, and key influencing words.

## Day 7 – Automatic Reporting System

- Implemented automatic job analysis report generation
- Report includes prediction result, fake risk percentage, and model confidence
- Supports text, image, and image URL inputs
- Added downloadable report feature
