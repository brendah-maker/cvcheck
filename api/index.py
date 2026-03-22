from flask import Flask, request, jsonify, render_template
import os
import io
import PyPDF2
from openai import OpenAI

app = Flask(__name__, template_folder='../templates')

# Initialize OpenAI (Set your API Key in Vercel Environment Variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Get Data
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            cv_text = extract_text_from_pdf(file)

    # 2. Call OpenAI for Score
    prompt = f"""
    Analyze this CV against this Job Description.
    CV: {cv_text[:2000]}
    JD: {jd_text[:2000]}
    Return ONLY a JSON object: 
    {{"score": 85, "missing_count": 4, "errors": 2, "verdict": "Short explanation"}}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    
    return response.choices[0].message.content

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    jd = data.get('jd')
    cv = data.get('cv')

    # Here you would typically verify payment via M-Pesa API before proceeding
    
    prompt = f"""
    Based on this CV and JD, generate:
    1. A list of 5 keywords to add.
    2. A 3-sentence professional summary.
    3. A full cover letter.
    Return JSON: {{"keywords": [], "summary": "", "cover_letter": ""}}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    
    return response.choices[0].message.content
