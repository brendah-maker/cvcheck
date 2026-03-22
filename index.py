import os
import json
import PyPDF2
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

# Initialize Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content
        return text
    except Exception as e:
        return ""

@app.route('/')
def index():
    return render_template('index.html')

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
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an ATS. Return ONLY JSON: {\"score\": number, \"missing_count\": number, \"errors\": number, \"verdict\": \"string\"}"},
                {"role": "user", "content": f"Analyze: CV: {cv_text[:2000]} JD: {jd_text[:2000]}"}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"score": 0, "missing_count": 0, "errors": 0, "verdict": "AI Analysis failed"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Return ONLY JSON: {\"keywords\": [], \"summary\": \"\", \"cover_letter\": \"\"}"},
                {"role": "user", "content": f"Optimize CV for JD: {data.get('jd')[:1500]} CV: {data.get('cv')[:2000]}"}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
