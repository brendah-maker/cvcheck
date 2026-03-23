import os
import json
import pdfplumber
import io
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-payment-config')
def get_config():
    # Fetch from Render Env Vars
    key = os.environ.get("INTASEND_PUBLISHABLE_KEY")
    return jsonify({"public_key": key})

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    
    # Improved PDF Extraction using pdfplumber logic
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            try:
                with pdfplumber.open(file) as pdf:
                    cv_text = ""
                    for page in pdf.pages:
                        cv_text += page.extract_text() or ""
            except Exception as e:
                return jsonify({"error": "PDF Corrupt"}), 400

    try:
        # CONCRETE PROMPT: Forces AI to act like a logic-based ATS
        sys_prompt = (
            "You are an ATS (Applicant Tracking System) Analyzer. "
            "Return ONLY a JSON object with these keys: "
            "score (0-100), verdict (1 sentence), visibility (HIGH/MED/LOW), "
            "missing_count (int), errors (int), gap_teaser (string), format_teaser (string). "
            "Be strict. If keywords are missing, score must be below 60."
        )
        
        user_prompt = f"JD: {jd_text[:1500]}\n\nCV: {cv_text[:2000]}"
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"score": 0, "verdict": "AI Analysis Timeout. Try again."}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    jd = data.get('jd', '')
    cv = data.get('cv', '')

    try:
        # HIGH-QUALITY GENERATION PROMPT
        sys_prompt = (
            "You are a Senior Career Coach. Return ONLY a JSON object with: "
            "keywords (list of 8 missing skills), summary (3 sentences), "
            "cover_letter (professional, tailored letter)."
        )
        
        user_prompt = f"Create a tactical report for this JD: {jd[:1000]} based on this CV: {cv[:1500]}. " \
                      f"The cover letter should focus on achievements, not just tasks."

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile", # Using the larger model for better writing
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"error": "Generation failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
