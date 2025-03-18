# cv-resume-checker

Overview

The CV/Resume Checker is a Streamlit-based web application that evaluates resumes against job descriptions. It parses resume files (PDF/DOCX), extracts key information, and scores resumes based on cv quality, experience relevance, and location alignment.

Features

Resume Parsing: Extracts text from PDF and DOCX files.

Keyword Matching: Uses TF-IDF and cosine similarity to compare resumes with job descriptions.

Experience Analysis: Extracts work experience duration from resumes.

Formatting Check: Identifies missing key sections like education, work experience, and contact information.

Location Scoring: Compares candidate location with job location.

User Interface: Built using Streamlit for easy interaction.

Installation

Prerequisites

Ensure you have the following installed:

Python 3.7+

pip (Python package manager)

Setup

Clone the repository or download the project files.

git clone https://github.com/cv-resume-checker.git
cd cv-resume-checker

Install the required dependencies:

pip install -r requirements.txt

Usage

Run the Streamlit application:

streamlit run app.py

Upload a resume (PDF/DOCX) and provide a job description.

Configure scoring weights for different evaluation criteria.

View candidate scores and analysis.


Dependencies

This project uses:

Streamlit for UI

NLTK & spaCy for NLP processing

pdfplumber & python-docx for text extraction

scikit-learn for TF-IDF and cosine similarity

Tips for Resume/CV Optimization

Use keywords from job descriptions.

Maintain simple and clear formatting.

Include a skills section with relevant terms.

Avoid using images, tables, or headers/footers.

Author

Developed by Rena Sebastian
