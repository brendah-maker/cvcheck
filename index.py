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
                return jsonify({"error": "PDF parse error"}), 400

    try:
        sys_prompt = (
            "You are a strict ATS logic engine. Analyze CV vs JD. "
            "Return ONLY JSON with these exact keys: "
            "score (Integer 0-100, NO strings, NO %), "
            "visibility ('HIGH', 'MEDIUM', or 'LOW'), "
            "missing_count (Integer), "
            "verdict (Short string), "
            "error_text (Short string describing the main issue)."
        )
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"JD: {jd_text[:1200]}\nCV: {cv_text[:1800]}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0, 
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # --- ROBUST SCORE PARSING (Prevents NaN) ---
        raw_score = result.get('score', 0)
        try:
            if isinstance(raw_score, str):
                # Remove any non-digits (like % or text) and convert to int
                clean_score = "".join(filter(str.isdigit, raw_score))
                result['score'] = int(clean_score) if clean_score else 0
            else:
                result['score'] = int(raw_score)
        except:
            result['score'] = 0
            
        return jsonify(result)
    except:
        return jsonify({"score": 0, "verdict": "Internal error. Try again."}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        sys_prompt = "Return ONLY JSON: {keywords:[], summary:'', cover_letter:''}"
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"JD: {data.get('jd')[:1000]} CV: {data.get('cv')[:1000]}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"error": "Failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
