from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pickle
import io
import os
import re
import nltk
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

nltk.download('stopwords')
nltk.download('punkt')

genai.configure(api_key="AIzaSyDssLo5_lALOwaJ4wZo9zsH3eh9zYG_oB0")

app = FastAPI()

# CORS settings to allow frontend requests
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://sudesha-f6697070b686.herokuapp.com/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

spacy_model = 'en_core_web_sm'
try:
    nlp = spacy.load(spacy_model)
except OSError:
    spacy.cli.download(spacy_model)
    nlp = spacy.load(spacy_model)

clf = pickle.load(open(r'clf.pkl', 'rb'))
tfidf = pickle.load(open(r'tfidf.pkl', 'rb'))

input_prompt = """
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field,software engineering,data science ,data analyst
and big data engineering and all the core fields such as mechanical , electrical, electronics and civil. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
best assistance for improving thr resumes. Assign the percentage Matching based 
on Jd and
the missing keywords with high accuracy. The response should be minimum 500 words and detaliled.
resume:{text}
description:{jd}

I want the response in one single string having the structure
{{"JD Match":"%","MissingKeywords:[]","Profile Summary":""}}
"""


def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text


def pdf_reader(file_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(file_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()

    converter.close()
    fake_file_handle.close()

    return text


def clean_resume(resume_text):
    clean_txt = re.sub('http\S+\s*', ' ', resume_text)
    clean_txt = re.sub('@\S+', ' ', clean_txt)
    clean_txt = re.sub('[0-9]+', ' ', clean_txt)
    clean_txt = re.sub('#\S+\s', ' ', clean_txt)
    clean_txt = re.sub('RT|cc', ' ', clean_txt)
    clean_txt = re.sub('[!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', clean_txt)
    clean_txt = re.sub('\s+', ' ', clean_txt)
    clean_txt = re.sub('\n', ' ', clean_txt)
    clean_txt = re.sub('\w*\d\w*', ' ', clean_txt)
    clean_txt = re.sub(' +', ' ', clean_txt)
    clean_txt = clean_txt.lower()
    clean_txt = clean_txt.strip()
    return clean_txt


def extract_skills(resume_text):
    skill_keywords = {
        'HR': ['Recruitment', 'Employee Relations', 'HR Management'],
        'DESIGNER': ['Adobe Creative Suite', 'UI/UX Design', 'Creativity'],
        'INFORMATION-TECHNOLOGY': ['Software Development', 'Network Security', 'IT Support'],
        'TEACHER': ['Curriculum Development', 'Classroom Management', 'Lesson Planning'],
        'ADVOCATE': ['Legal Research', 'Client Counseling', 'Litigation'],
        'BUSINESS-DEVELOPMENT': ['Sales Strategy', 'Market Research', 'Client Acquisition'],
        'HEALTHCARE': ['Patient Care', 'Medical Terminology', 'Healthcare Management'],
        'FITNESS': ['Personal Training', 'Nutrition Planning', 'Exercise Science'],
        'AGRICULTURE': ['Crop Management', 'Soil Science', 'Agricultural Machinery'],
        'BPO': ['Customer Service', 'Call Center Operations', 'Telemarketing'],
        'SALES': ['Sales Techniques', 'Customer Relationship Management (CRM)', 'Negotiation'],
        'CONSULTANT': ['Strategic Planning', 'Problem Solving', 'Project Management'],
        'DIGITAL-MEDIA': ['Content Creation', 'Social Media Marketing', 'SEO'],
        'AUTOMOBILE': ['Vehicle Maintenance', 'Automotive Engineering', 'Mechanics'],
        'CHEF': ['Culinary Arts', 'Menu Planning', 'Kitchen Management'],
        'FINANCE': ['Financial Analysis', 'Budgeting', 'Accounting'],
        'APPAREL': ['Fashion Design', 'Textile Knowledge', 'Merchandising'],
        'ENGINEERING': ['Project Management', 'Technical Drawing', 'Engineering Design'],
        'ACCOUNTANT': ['Bookkeeping', 'Tax Preparation', 'Financial Reporting'],
        'CONSTRUCTION': ['Building Codes', 'Construction Management', 'Blueprint Reading'],
        'PUBLIC-RELATIONS': ['Media Relations', 'Crisis Management', 'Press Releases'],
        'BANKING': ['Financial Services', 'Risk Management', 'Customer Service'],
        'ARTS': ['Art History', 'Creative Techniques', 'Artistic Skills'],
        'AVIATION': ['Flight Operations', 'Aircraft Maintenance', 'Pilot Training'],
        'Data Science': ['Machine Learning', 'Data Analysis', 'Statistical Modeling'],
        'Advocate': ['Legal Research', 'Client Counseling', 'Litigation'],
        'Arts': ['Art History', 'Creative Techniques', 'Artistic Skills'],
        'Web Designing': ['HTML/CSS', 'Responsive Design', 'JavaScript'],
        'Mechanical Engineer': ['Mechanical Design', 'CAD Software', 'Thermodynamics'],
        'Sales': ['Sales Techniques', 'Customer Relationship Management (CRM)', 'Negotiation'],
        'Health and fitness': ['Personal Training', 'Nutrition Planning', 'Exercise Science'],
        'Civil Engineer': ['Structural Analysis', 'Construction Management', 'Civil Engineering Design'],
        'Java Developer': ['Java Programming', 'Spring Framework', 'Object-Oriented Design'],
        'Business Analyst': ['Business Process Modeling', 'Data Analysis', 'Requirements Gathering'],
        'SAP Developer': ['SAP ABAP', 'SAP Modules', 'System Configuration'],
        'Automation Testing': ['Test Automation', 'Selenium', 'Quality Assurance'],
        'Electrical Engineering': ['Circuit Design', 'Electrical Systems', 'Power Distribution'],
        'Operations Manager': ['Operational Efficiency', 'Supply Chain Management', 'Leadership'],
        'Python Developer': ['Python Programming', 'Django', 'Data Analysis'],
        'DevOps Engineer': ['CI/CD', 'Docker', 'Cloud Computing'],
        'Network Security Engineer': ['Cybersecurity', 'Network Infrastructure', 'Firewalls'],
        'PMO': ['Project Management', 'Resource Allocation', 'Program Governance'],
        'Database': ['SQL', 'Database Management', 'Data Modeling'],
        'Hadoop': ['Big Data', 'Hadoop Ecosystem', 'MapReduce'],
        'ETL Developer': ['Data Warehousing', 'ETL Tools', 'SQL'],
        'DotNet Developer': ['C# Programming', '.NET Framework', 'ASP.NET'],
        'Blockchain': ['Blockchain Technology', 'Smart Contracts', 'Cryptography'],
        'Testing': ['Software Testing', 'Manual Testing', 'Test Planning']
    }

    tokens = nltk.word_tokenize(resume_text.lower())
    found_skills = set()

    for category, skills in skill_keywords.items():
        for skill in skills:
            if skill.lower() in tokens:
                print(f"Skill found: {skill}")
                found_skills.add(skill)

    return list(found_skills)


def extract_info(resume_text):
    nlp_text = nlp(resume_text)
    names = [ent.text.replace(' ', '') for ent in nlp_text.ents if ent.label_ == 'PERSON']
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    phones = re.findall(r'\b\d{10}\b', resume_text)
    skills = extract_skills(resume_text)

    return {
        "names": names,
        "emails": emails,
        "phones": phones,
        "skills": skills
    }


@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume Evaluator API"}


@app.post("/extract-info/")
async def extract_resume_info(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    resume_text = pdf_reader(temp_file_path)
    resume_text = clean_resume(resume_text)
    return extract_info(resume_text)


@app.post("/evaluate-resume/")
async def evaluate_resume(resume: UploadFile, job_description: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await resume.read())
        temp_file_path = temp_file.name
    resume_text = pdf_reader(temp_file_path)
    resume_text = clean_resume(resume_text)
    prompt = input_prompt.format(text=resume_text, jd=job_description)
    return {"evaluation": get_gemini_response(prompt)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
