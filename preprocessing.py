import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
df = pd.read_csv(
    r"C:/Users/aashi/OneDrive/Desktop/fake job detector/fake-job-posting-detector-main/data/fake_job_postings.csv"
)
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text
df['clean_text'] = df['description'].apply(clean_text)
df.to_csv( r"C:/Users/aashi/OneDrive/Desktop/fake job detector/fake-job-posting-detector-main/data/cleaned_data.csv", index=False)


