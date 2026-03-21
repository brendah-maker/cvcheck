import os
import io
import requests
import pdfplumber
from flask import Flask, request, jsonify, render_template, send_file
from groq import Groq
from bs4 import BeautifulSoup
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# CONFIGURATION - Set these in Render Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def extract_pdf_text(file):
    try:
        with pdfplumber.open(file) as pdf:
            return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except Exception:
        return ""

def scrape_job_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for s in soup(["script", "style", "nav", "footer"]): s.decompose()
        return soup.get_text()[:4000]
    except Exception:
        return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Handle CV
    if 'cv_file' in request.files and request.files['cv_file'].filename != '':
        cv_text = extract_pdf_text(request.files['cv_file'])
    else:
        cv_text = request.form.get('cv_text', '')

    # Handle Job Description
    job_url = request.form.get('job_url', '')
    if job_url.startswith('http'):
        jd_text = scrape_job_text(job_url)
    else:
        jd_text = request.form.get('jd_text', '')

    if not cv_text or not jd_text:
        return jsonify({"error": "Please provide both a CV and a Job Description"}), 400

    # Llama 3 8B - Fast Analysis
    prompt = f"""Analyze CV vs JD. Return JSON ONLY: 
    {{"score": 0-100, "missing_count": int, "errors": int, "verdict": "short sentence"}}
    JD: {jd_text[:1500]}
    CV: {cv_text[:1500]}"""

    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            response_format={"type": "json_object"}
        )
        return chat.choices[0].message.content
    except Exception as e:
        return jsonify({"error": "AI Service busy. Try again."}), 500

@app.route('/generate-docs', methods=['POST'])
def generate():
    data = request.json
    jd_text = data.get('jd', '')
    cv_text = data.get('cv', '')

    # Llama 3 70B - Smart Rewrite
    prompt = f"""As an HR expert, optimize this CV for this JD. Return JSON ONLY:
    {{ "keywords": ["7 missing skills"], "summary": "3-sentence professional summary", "cover_letter": "full text" }}
    JD: {jd_text[:2000]}
    CV: {cv_text[:2000]}"""

    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            response_format={"type": "json_object"}
        )
        return chat.choices[0].message.content
    except Exception as e:
        return jsonify({"error": "Generation failed."}), 500

if __name__ == '__main__':
    app.run(debug=True)
