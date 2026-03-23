import os
import json
import pdfplumber
import io
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-payment-config')
def get_config():
    key = os.environ.get("INTASEND_PUBLISHABLE_KEY")
    return jsonify({"public_key": key})

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            try:
                with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                    cv_text = "".join([page.extract_text() or "" for page in pdf.pages])
            except:
                return jsonify({"error": "PDF read error"}), 400

    try:
        # CONCRETE & STRICT PROMPT FOR CONSISTENCY
        sys_prompt = (
            "You are a strict ATS (Applicant Tracking System). "
            "Analyze the CV vs JD and return ONLY JSON. "
            "Calculation Rules:\n"
            "1. Score: Must be a WHOLE NUMBER (Integer) between 0 and 100 based on keyword match density.\n"
            "2. Visibility: Must be 'HIGH', 'MEDIUM', or 'LOW'.\n"
            "3. Missing Count: Integer count of hard skills found in JD but not in CV.\n"
            "4. Error Count: Integer count (number of layout/grammar issues).\n"
            "5. Verdict: One short sentence.\n"
            "6. Error Text: A short description of the most critical error."
        )
        
        user_prompt = f"JD: {jd_text[:1200]}\n\nCV: {cv_text[:1800]}"
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0, # Zero temperature ensures consistency
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return jsonify(result)
    except:
        return jsonify({"score": 0, "verdict": "Check inputs"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        sys_prompt = "Return ONLY JSON with keys: keywords (list), summary (text), cover_letter (text)."
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"JD: {data.get('jd')[:1000]} CV: {data.get('cv')[:1000]}"}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"error": "Failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
