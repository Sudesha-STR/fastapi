from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import pickle
import io
import os
import re
import nltk
import tempfile
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import google.generativeai as genai
from dotenv import load_dotenv
from fpdf import FPDF

app = FastAPI()

load_dotenv()

nltk.download('stopwords')
nltk.download('punkt')

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

spacy_model = 'en_core_web_sm'
try:
    nlp = spacy.load(spacy_model)
except OSError:
    spacy.cli.download(spacy_model)
    nlp = spacy.load(spacy_model)

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

input_prompt = """
Hey Act Like a skilled or very experienced ATS (Application Tracking System)
with a deep understanding of tech field, software engineering, data science, data analyst,
and big data engineering, as well as core fields such as mechanical, electrical, electronics, and civil engineering. Your task is to evaluate the resume based on the given job description and also provide feedback on grammatical correctness and overall quality. You must consider the job market is very competitive and provide the best assistance for improving resumes. Assign a percentage matching based on JD, missing keywords, and overall score out of 100 with detailed reasons.

resume:{text}
description:{jd}

I want the response in one single string having the structure
{{"JD Match":"%","MissingKeywords":[],"Profile Summary":"","Grammar Suggestions":[],"Overall Score":"", "Reasons":""}}
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
                found_skills.add(skill)

    return list(found_skills)

def extract_info(resume_text):
    nlp_text = nlp(resume_text)
    names = [ent.text.replace(' ', '') for ent in nlp_text.ents if ent.label_ == 'PERSON']
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    phones = re.findall(r'\b\d{10}\b', resume_text)
    skills = extract_skills(resume_text)

    return {
        'name': names[0] if names else 'N/A',
        'email': emails[0] if emails else 'N/A',
        'mobile_number': phones[0] if phones else 'N/A',
        'skills': skills
    }

category_mapping = {
    'HR': 31,
    'DESIGNER': 28,
    'INFORMATION-TECHNOLOGY': 22,
    'TEACHER': 19,
    'ADVOCATE': 7,
    'BUSINESS-DEVELOPMENT': 9,
    'HEALTHCARE': 8,
    'FITNESS': 6,
    'AGRICULTURE': 0,
    'BPO': 4,
    'SALES': 23,
    'CONSULTANT': 10,
    'DIGITAL-MEDIA': 25,
    'AUTOMOBILE': 2,
    'CHEF': 5,
    'FINANCE': 13,
    'APPAREL': 1,
    'ENGINEERING': 11,
    'ACCOUNTANT': 26,
    'CONSTRUCTION': 12,
    'PUBLIC-RELATIONS': 20,
    'BANKING': 3,
    'ARTS': 24,
    'AVIATION': 18,
    'Data Science': 15,
    'Advocate': 7,
    'Arts': 24,
    'Web Designing': 28,
    'Mechanical Engineer': 14,
    'Sales': 23,
    'Health and fitness': 6,
    'Civil Engineer': 14,
    'Java Developer': 22,
    'Business Analyst': 9,
    'SAP Developer': 22,
    'Automation Testing': 17,
    'Electrical Engineering': 11,
    'Operations Manager': 21,
    'Python Developer': 22,
    'DevOps Engineer': 22,
    'Network Security Engineer': 22,
    'PMO': 21,
    'Database': 22,
    'Hadoop': 15,
    'ETL Developer': 22,
    'DotNet Developer': 22,
    'Blockchain': 22,
    'Testing': 17
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume Evaluator API"}

@app.post("/parse_pdf")
async def parse_pdf(file: UploadFile = File(...)):
    if file.filename.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
            text = pdf_reader(tmp_file_path)
            os.remove(tmp_file_path)
        return JSONResponse(content={"text": text})
    else:
        return JSONResponse(content={"error": "Invalid file type. Only PDFs are supported."}, status_code=400)

@app.post("/parse_text")
async def parse_text(text: str = Form(...)):
    text = clean_resume(text)
    return JSONResponse(content={"text": text})

@app.post("/extract_skills")
async def extract_skills_from_text(text: str = Form(...)):
    text = clean_resume(text)
    info = extract_info(text)
    return JSONResponse(content=info)

@app.post("/classify_resume")
async def classify_resume(text: str = Form(...)):
    text = clean_resume(text)
    tfidf_text = tfidf.transform([text])
    prediction = clf.predict(tfidf_text)
    category = list(category_mapping.keys())[list(category_mapping.values()).index(prediction[0])]
    return JSONResponse(content={"category": category})

@app.post("/generate_feedback")
async def generate_feedback(resume_text: str = Form(...), jd: str = Form(...)):
    resume_text = clean_resume(resume_text)
    jd = clean_resume(jd)
    input_text = input_prompt.format(text=resume_text, jd=jd)
    feedback = get_gemini_response(input_text)
    feedback_json = json.loads(feedback)
    return JSONResponse(content=feedback_json)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
