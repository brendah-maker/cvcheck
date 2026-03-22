import os
import json
import PyPDF2
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# Ensure this variable name matches exactly what you typed in Render
INTASEND_KEY = os.environ.get("INTASEND_PUBLISHABLE_KEY")

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content
        return text
    except: return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-payment-config')
def get_config():
    # Sending the key to the frontend
    return jsonify({"public_key": INTASEND_KEY})

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            cv_text = extract_text_from_pdf(file)

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an ATS. Output ONLY JSON."},
                {"role": "user", "content": f"Analyze: CV: {cv_text[:2000]} JD: {jd_text[:1500]}. Return JSON with keys: score, missing_count, errors, verdict, visibility, gap_teaser, format_teaser"}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"score": 0, "verdict": "AI Analysis Failed"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Return ONLY JSON: {'keywords': [], 'summary': '', 'cover_letter': ''}"},
                {"role": "user", "content": f"Optimize CV for JD: {data.get('jd')[:1000]} CV: {data.get('cv')[:1500]}"}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"error": "Failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
