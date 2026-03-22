import os
import json
import PyPDF2
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

# Initialize Groq (Make sure GROQ_API_KEY is in Render Environment Variables)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        print(f"PDF Error: {e}")
        return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    
    # Priority to PDF file if uploaded
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            extracted = extract_text_from_pdf(file)
            if extracted:
                cv_text = extracted

    if not jd_text or not cv_text:
        return jsonify({"error": "Missing input"}), 400

    prompt = f"""
    You are an expert ATS (Applicant Tracking System). Analyze this CV against the JD.
    CV: {cv_text[:3000]}
    JD: {jd_text[:2000]}
    
    Return ONLY a JSON object with:
    "score": (0-100 number),
    "missing_count": (number of keywords missing),
    "errors": (number of formatting issues),
    "verdict": (1-sentence punchy advice)
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a recruitment bot. Respond ONLY in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"score": 0, "missing_count": 0, "errors": 0, "verdict": "Error connecting to AI"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    jd = data.get('jd', '')
    cv = data.get('cv', '')

    prompt = f"""
    Create career documents.
    CV: {cv[:3000]}
    JD: {jd[:2000]}
    
    Return ONLY a JSON object with:
    "keywords": [list of 5 missing skills],
    "summary": "a 3-sentence powerful summary",
    "cover_letter": "a high-conversion professional cover letter"
    """

    try:
        response = client.chat.completify.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"error": "Failed to generate documents"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
