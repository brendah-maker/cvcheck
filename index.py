import os
import json
import PyPDF2
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

# Initialize Groq - Ensure GROQ_API_KEY is in your Render/Vercel Env Variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content
        return text
    except Exception:
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
        # We ask Llama for a "Diagnostic" teaser to build value
        prompt = f"""
        Analyze match between CV and JD.
        CV: {cv_text[:2500]}
        JD: {jd_text[:2000]}
        
        Return ONLY a JSON object:
        {{
            "score": 45,
            "missing_count": 8,
            "errors": 2,
            "verdict": "High risk of rejection.",
            "visibility": "INVISIBLE",
            "gap_teaser": "You are missing 5 technical keywords related to this industry.",
            "format_teaser": "Your header layout is confusing the ATS robot."
        }}
        """
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an ATS Recruiter. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"score": 0, "verdict": "AI Analysis failed"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a Career Expert. Output ONLY JSON: {\"keywords\": [], \"summary\": \"\", \"cover_letter\": \"\"}"},
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
