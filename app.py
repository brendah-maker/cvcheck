import os
import io
import requests
import pdfplumber
from flask import Flask, request, jsonify, render_template, send_file
from groq import Groq
from bs4 import BeautifulSoup
from fpdf import FPDF

app = Flask(__name__)

# CONFIGURATION
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
client = Groq(api_key=GROQ_API_KEY)

def extract_pdf_text(file):
    try:
        with pdfplumber.open(file) as pdf:
            return "".join([page.extract_text() for page in pdf.pages])
    except:
        return ""

def scrape_job_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for s in soup(["script", "style"]): s.decompose()
        return soup.get_text()[:3000] # Limit to 3k chars
    except:
        return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Get CV Text
    cv_text = ""
    if 'cv_file' in request.files and request.files['cv_file'].filename != '':
        cv_text = extract_pdf_text(request.files['cv_file'])
    else:
        cv_text = request.form.get('cv_text', '')

    # 2. Get Job Description
    jd_text = ""
    job_url = request.form.get('job_url', '')
    if job_url.startswith('http'):
        jd_text = scrape_job_text(job_url)
    if not jd_text:
        jd_text = request.form.get('jd_text', '')

    if not cv_text or not jd_text:
        return jsonify({"error": "Missing CV or Job Description"}), 400

    # 3. AI Teaser Analysis (Llama 3 8B)
    prompt = f"""Analyze this CV against this Job Description. 
    Return ONLY a JSON object: {{"score": 0-100, "missing_count": int, "errors": int}}
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
        return jsonify({"error": str(e)}), 500

@app.route('/generate-docs', methods=['POST'])
def generate():
    # In production, check M-Pesa payment confirmation here
    data = request.json
    jd_text = data.get('jd', '')
    cv_text = data.get('cv', '')

    # AI Deep Dive (Llama 3 70B)
    prompt = f"""As an HR expert, provide JSON:
    {{ "keywords": ["skill1", "skill2"], "summary": "3-sentence CV summary", "cover_letter": "full letter" }}
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
