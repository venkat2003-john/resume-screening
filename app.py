import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Combine job description with resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    resume_names = []

    # Extract text from all resumes
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
        resume_names.append(file.name)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Create a DataFrame with resume names and scores
    results = pd.DataFrame({"Resume": resume_names, "Score": scores})

    # Sort results by score in descending order
    results = results.sort_values(by="Score", ascending=False)

    # Display results
    st.write(results)
