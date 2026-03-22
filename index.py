import os
import json
import PyPDF2
import requests
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

# Keys from Render Environment Variables (Secure)
# Set these in Render Dashboard -> Environment
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
INTASEND_PUBLISHABLE_KEY = os.environ.get("INTASEND_PUBLISHABLE_KEY")

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

# Send the Public Key to the UI securely
@app.route('/get-payment-config')
def get_config():
    return jsonify({"public_key": INTASEND_PUBLISHABLE_KEY})

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            pdf_text = extract_text_from_pdf(file)
            if pdf_text: cv_text = pdf_text

    try:
        prompt = f"""
        Analyze match between CV and JD.
        CV: {cv_text[:2000]}
        JD: {jd_text[:1500]}
        Return ONLY a JSON object:
        {{
            "score": 45,
            "missing_count": 5,
            "errors": 2,
            "verdict": "High risk of rejection.",
            "visibility": "LOW",
            "gap_teaser": "Missing technical keywords in this industry.",
            "format_teaser": "Header layout is confusing the ATS robot."
        }}
        """
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an ATS. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        print(f"Llama Error: {e}")
        return jsonify({"score": 0, "verdict": "AI Analysis Failed"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        prompt = f"Create Keywords, Summary, and Cover Letter for JD: {data.get('jd')[:1000]} CV: {data.get('cv')[:1500]}"
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Return ONLY JSON: {'keywords': [], 'summary': '', 'cover_letter': ''}"},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        print(f"Llama Docs Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
