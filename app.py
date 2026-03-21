import os
import io
import requests
import pdfplumber
from flask import Flask, request, jsonify, render_template, send_file
from groq import Groq
from bs4 import BeautifulSoup
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv() # Loads variables from .env file

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_pdf_text(file):
    try:
        with pdfplumber.open(file) as pdf:
            return "".join([page.extract_text() for page in pdf.pages])
    except: return ""

def scrape_job_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        for s in soup(["script", "style"]): s.decompose()
        return soup.get_text()[:3000]
    except: return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    cv_text = extract_pdf_text(request.files['cv_file']) if 'cv_file' in request.files else request.form.get('cv_text', '')
    jd_text = scrape_job_text(request.form.get('job_url', '')) or request.form.get('jd_text', '')

    prompt = f"Analyze CV vs JD. Return JSON: {{'score': 0-100, 'missing_count': int, 'errors': int}}. JD: {jd_text[:1500]} CV: {cv_text[:1500]}"
    chat = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama3-8b-8192", response_format={"type": "json_object"})
    return chat.choices[0].message.content

@app.route('/generate-docs', methods=['POST'])
def generate():
    data = request.json
    prompt = f"As HR expert, provide JSON: {{'keywords': [], 'summary': '', 'cover_letter': ''}}. JD: {data.get('jd')[:2000]} CV: {data.get('cv')[:2000]}"
    chat = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama3-70b-8192", response_format={"type": "json_object"})
    return chat.choices[0].message.content

if __name__ == '__main__':
    app.run(debug=True)
