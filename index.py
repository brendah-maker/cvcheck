import os
import json
import PyPDF2
import requests
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

# Keys from Render Environment Variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
INTASEND_PUBLISHABLE_KEY = os.environ.get("INTASEND_PUBLISHABLE_KEY")
INTASEND_SECRET_KEY = os.environ.get("INTASEND_SECRET_KEY")

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except: return ""

@app.route('/')
def index():
    return render_template('index.html')

# 1. Provide the Public Key to Frontend only when requested
@app.route('/get-payment-config')
def get_config():
    return jsonify({"public_key": INTASEND_PUBLISHABLE_KEY})

# 2. Free Analysis Logic
@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            cv_text = extract_text_from_pdf(file)

    prompt = f"Analyze CV: {cv_text[:2000]} vs JD: {jd_text[:1500]}. Return ONLY JSON: {{'score': 45, 'missing_count': 5, 'errors': 2, 'verdict': 'string', 'visibility': 'LOW', 'gap_teaser': 'string', 'format_teaser': 'string'}}"
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": "You are an ATS. Output ONLY JSON."},
                      {"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"score": 0, "verdict": "AI Analysis Failed"}), 500

# 3. Paid Document Generation
@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    prompt = f"Create Keywords, Summary, and Cover Letter for JD: {data.get('jd')[:1000]} CV: {data.get('cv')[:1500]}"
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": "Return ONLY JSON: {'keywords': [], 'summary': '', 'cover_letter': ''}"},
                      {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
