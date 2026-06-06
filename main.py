import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

st.title("Job Portal Resume Ranking System")
st.write("Using TF-IDF Vector Space Model")

# -------------------------------
# Function to extract text from PDF
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# -------------------------------
# Job Description Input
# -------------------------------
job_description = st.text_area("Enter Job Description")

# -------------------------------
# Upload Resumes
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type="pdf",
    accept_multiple_files=True
)

# -------------------------------
# Ranking Logic
# -------------------------------
if st.button("Rank Resumes"):

    if job_description == "" or uploaded_files is None:
        st.warning("Please enter job description and upload resumes.")
    
    else:

        resumes = []
        resume_names = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)
            resume_names.append(file.name)

        documents = [job_description] + resumes

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)

        job_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]

        similarity = cosine_similarity(job_vector, resume_vectors)

        scores = similarity.flatten()

        results = pd.DataFrame({
            "Resume": resume_names,
            "Similarity Score": scores
        })

        results = results.sort_values(by="Similarity Score", ascending=False)

        st.subheader("Resume Ranking Results")
        st.dataframe(results)