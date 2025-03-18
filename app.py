import os
import re
import pandas as pd
import streamlit as st
import pdfplumber
import docx
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from datetime import datetime

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load NLP model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Resume ATS Checker", layout="wide")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_keyword_match(resume_text, job_description):
    resume_text = preprocess_text(resume_text)
    job_description = preprocess_text(job_description)

    vectorizer = TfidfVectorizer(stop_words='english')
    count_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(count_matrix)[0][1]

    job_words = set(job_description.split()) - set(stopwords.words('english'))
    resume_words = set(resume_text.split())

    matching_keywords = job_words.intersection(resume_words)
    missing_keywords = job_words - resume_words

    return similarity, matching_keywords, missing_keywords

def check_format_issues(resume_text):
    issues = []
    if isinstance(resume_text, bytes):
        resume_text = resume_text.decode('utf-8')
    if isinstance(resume_text, str):  # Ensure resume_text is a string
        if len(resume_text) < 300:
            issues.append("Resume is too short. Consider adding more content.")
    else:
        issues.append("Invalid resume text format. Expected a string.")

    # Check for contact information
    if not re.search(r'[\w\.-]+@[\w\.-]+', resume_text):
        issues.append("Email address might be missing")

    if not re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', resume_text):
        issues.append("Phone number might be missing")

    # Check for education section
    edu_keywords = ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'phd']
    if not any(keyword in resume_text.lower() for keyword in edu_keywords):
        issues.append("Education section might be missing")

    # Check for experience section
    exp_keywords = ['experience', 'work', 'job', 'employment', 'career']
    if not any(keyword in resume_text.lower() for keyword in exp_keywords):
        issues.append("Work experience section might be missing")

    return issues

# CVProcessor class
class CVProcessor:
    def __init__(self):
        self.location_pattern = re.compile(r"(?i)(address|location):?\s*(.+)")
        self.date_pattern = re.compile(r'(\b[A-Za-z]+\s\d{4}\b)\s*-\s*(\b[A-Za-z]+\s\d{4}\b)')
        print("CV Processor initialized")

    def parse_cv(self, file_path):
        print(f"\nParsing CV: {file_path}")
        try:
            if file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                raise ValueError("Unsupported file format")
            return self._clean_text(text)
        except Exception as e:
            print(f"Error parsing {file_path}: {str(e)}")
            return ""

    def _clean_text(self, text):
        cleaned = ' '.join(text.replace('\n', ' ').replace('\t', ' ').split())
        print(f"Cleaned text length: {len(cleaned)} characters")
        return cleaned

    def extract_experience(self, text):
        print("\nExtracting experience...")
        duration_pattern = re.compile(r'(\d+)\s*(weeks?|months?)', re.IGNORECASE)
        date_ranges = re.findall(self.date_pattern, text)

        total_months = 0
        duration_matches = duration_pattern.findall(text)
        for duration, unit in duration_matches:
            if 'week' in unit.lower():
                total_months += int(duration) / 4
            elif 'month' in unit.lower():
                total_months += int(duration)

        for start_str, end_str in date_ranges:
            start_str = start_str if start_str else ""
            end_str = end_str if end_str else ""

            if start_str and end_str:
                try:
                    start_date = datetime.strptime(start_str, "%B %Y")
                except ValueError:
                    start_date = datetime.strptime(start_str, "%b %Y")

                try:
                    end_date = datetime.strptime(end_str, "%B %Y")
                except ValueError:
                    end_date = datetime.strptime(end_str, "%b %Y")

                months_difference = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                total_months += months_difference

                print(f"Found date range: {start_str} to {end_str}, Duration: {months_difference} months")

        experience_years = total_months / 12
        print(f"Total calculated experience: {experience_years:.2f} years")
        return experience_years

    def extract_location(self, text):
        print("\nExtracting location...")
        doc = nlp(text)
        locations = []

        for ent in doc.ents:
            if ent.label_ == "GPE":
                locations.append(ent.text)

        non_location_terms = {"AI", "React", "SaaS", "platform", "team", "features", "models"}
        locations = [loc for loc in locations if loc not in non_location_terms]

        if locations:
            location = locations[0] if len(locations) == 1 else ', '.join(locations)
            print(f"Location found: {location}")
            return location
        else:
            print("No location found")
            return "Unknown"

# CandidateScorer class
class CandidateScorer:
    def __init__(self, job_description):
        print("\nInitializing Candidate Scorer...")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.job_vector = self.vectorizer.fit_transform([job_description])
        self.required_keywords = self.extract_keywords(job_description)
        print("Job description vector created")

    def extract_keywords(self, job_description):
        keywords = []
        lines = job_description.splitlines()
        for line in lines:
            if "Required Skills:" in line or "Preferred Skills:" in line:
                continue
            keywords.extend(re.findall(r'\b\w+\b', line))
        return set(keywords)

    def score_cv_quality(self, text):
        print("\nScoring CV quality...")
        try:
            keyword_matches = self.match_keywords(text)
            keyword_score = min(len(keyword_matches) * 10, 100)
            print(f"Keyword Match Score: {keyword_score}/100")

            formatting_score = self.analyze_formatting(text)
            print(f"Formatting Score: {formatting_score}/100")

            total_score = min(keyword_score + formatting_score, 100)
            print(f"Total CV Quality Score: {total_score}/100")
            return total_score
        except Exception as e:
            print(f"Error in CV quality scoring: {str(e)}")
            return 0

    def match_keywords(self, text):
        found_keywords = [keyword for keyword in self.required_keywords if keyword.lower() in text.lower()]
        print(f"Matched Keywords: {found_keywords}")
        return found_keywords

    def analyze_formatting(self, text):
        sections = ['Education', 'Experience', 'Skills']
        found_sections = [section for section in sections if section.lower() in text.lower()]
        formatting_score = len(found_sections) * 33
        print(f"Found Sections: {found_sections}")
        return min(formatting_score, 100)

    def score_experience_match(self, text):
        print("\nScoring experience match...")
        try:
            cv_vector = self.vectorizer.transform([text])
            score = cosine_similarity(self.job_vector, cv_vector)[0][0] * 100
            print(f"Experience Match Score: {score:.2f}/100")
            return score
        except Exception as e:
            print(f"Error in experience matching: {str(e)}")
            return 0

    def score_location(self, candidate_loc, job_loc):
        print("\nScoring location...")
        score = 100 if job_loc.lower() in candidate_loc.lower() else 0
        print(f"Location Score: {score}/100 ({candidate_loc} vs {job_loc})")
        return score

class RecruitmentSystem:
    def __init__(self, job_location, required_experience, job_description):
        print("\nInitializing Recruitment System...")
        self.job_location = job_location
        self.required_experience = required_experience
        self.job_description = job_description
        self.cv_processor = CVProcessor()
        self.scorer = CandidateScorer(job_description)
        self.candidates = pd.DataFrame()
        print(f"System ready for {job_location} with {required_experience} years requirement")

    def add_candidate(self, file_path):
        print(f"\nAdding candidate: {file_path}")
        try:
            text = self.cv_processor.parse_cv(file_path)
            experience = self.cv_processor.extract_experience(text)
            location = self.cv_processor.extract_location(text)

            new_candidate = {
                'id': os.path.basename(file_path),
                'experience': experience,
                'location': location,
                'cv_quality': self.scorer.score_cv_quality(text),
                'experience_quality': self.scorer.score_experience_match(text),
                'location_score': self.scorer.score_location(location, self.job_location)
            }

            new_df = pd.DataFrame([new_candidate])
            self.candidates = pd.concat([self.candidates, new_df], ignore_index=True)
            print("Candidate added successfully")
            print(f"Current candidates DataFrame:\n{self.candidates}\n")
        except Exception as e:
            print(f"Error adding candidate: {str(e)}")

    def get_top_candidates(self, weights):
        print("\nCalculating final scores...")
        try:
            if self.candidates.empty:
                print("No candidates to calculate scores.")
                return pd.DataFrame()

            total_weight = sum(weights.values())
            norm_weights = {k: v / total_weight for k, v in weights.items()}

            self.candidates['final_score'] = (
                self.candidates['cv_quality'] * norm_weights['cv_quality'] +
                self.candidates['experience_quality'] * norm_weights['experience_quality'] +
                self.candidates['location_score'] * norm_weights['location']
            )

            print("Final candidate scores:")
            print(self.candidates[['id', 'final_score']])
            return self.candidates.sort_values('final_score', ascending=False)
        except Exception as e:
            print(f"Error calculating scores: {str(e)}")
            return pd.DataFrame()

# Streamlit App
def main():
    st.title("CV/Resume ATS Checker")
    st.subheader("Upload your resume and configure the job posting")

    if 'recruitment' not in st.session_state:
        st.session_state.recruitment = None

    job_description = st.text_area("Paste Job Description Here", height=300)
    job_location = st.text_input("Enter job location (City, Country):", "New York, USA")
    required_experience = st.number_input("Required experience (years):", min_value=0.0, value=2.0, step=0.5)

    st.subheader("Enter scoring weights (must sum to 100):")
    cv_quality_weight = st.slider("CV Quality weight:", 0, 100, 30)
    experience_quality_weight = st.slider("Experience Relevance weight:", 0, 100, 40)
    location_weight = st.slider("Location weight:", 0, 100, 30)

    total_weight = cv_quality_weight + experience_quality_weight + location_weight
    if total_weight != 100:
        st.warning(f"Weights sum to {total_weight}. Adjusting weights to sum to 100.")
        cv_quality_weight = int((cv_quality_weight / total_weight) * 100)
        experience_quality_weight = int((experience_quality_weight / total_weight) * 100)
        location_weight = int((location_weight / total_weight) * 100)

    weights = {
        'cv_quality': cv_quality_weight,
        'experience_quality': experience_quality_weight,
        'location': location_weight
    }

    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=['pdf', 'docx'])

    if uploaded_file is not None and job_description:
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.session_state.recruitment is None:
            st.session_state.recruitment = RecruitmentSystem(job_location, required_experience, job_description)

        st.session_state.recruitment.add_candidate(temp_file_path)
        os.remove(temp_file_path)

    if st.session_state.recruitment is not None:
        results = st.session_state.recruitment.get_top_candidates(weights)
        if not results.empty:
            st.subheader("Top Candidates")
            st.dataframe(results[['id', 'final_score']])
    
# Add explanations and tips
st.sidebar.title("How ATS Works")
st.sidebar.write("""
An Applicant Tracking System (ATS) is software used by employers to manage job applications. Here's how it works:

1. Parsing: The ATS extracts information from your resume
2. Keyword Matching: It looks for relevant keywords from the job description
3. Ranking: Applications are scored and ranked based on matches
4. Filtering: Low-scoring applications may be automatically rejected

This tool helps you optimize your resume for ATS by:
- Comparing your resume against a specific job description
- Identifying missing keywords
- Highlighting potential formatting issues
- Providing an estimated compatibility score
""")

st.sidebar.title("Tips for ATS Success")
st.sidebar.write("""
- Tailor your resume for each job application
- Use keywords from the job description naturally
- Keep formatting simple and consistent
- Use standard section headings
- Include a skills section with relevant keywords
- Avoid using images, tables, or headers/footers
- Save your resume as a .docx or .pdf file
""")

if __name__ == "__main__":
    main()
