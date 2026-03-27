import os
import json
import pdfplumber
import io
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    
    # Handle PDF Upload
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            try:
                with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                    cv_text = " ".join([page.extract_text() or "" for page in pdf.pages])
            except Exception:
                return jsonify({"error": "Failed to parse PDF"}), 400

    if not jd_text or not cv_text:
        return jsonify({"error": "Missing input data"}), 400

    try:
        # Strict instructions for consistency
        sys_prompt = """
        You are an elite Executive Recruiter and ATS Expert. 
        Perform a professional gap analysis between the Resume and Job Description.
        
        SCORING RUBRIC:
        - Technical Alignment (40%): Hard skills and specific tools.
        - Experience Depth (40%): Industry relevance and level of responsibility.
        - Communication Quality (20%): Clarity, education, and professional formatting.

        Return ONLY JSON:
        {
            "score": int,
            "visibility": "High" | "Medium" | "Low",
            "verdict": "A sophisticated, executive-level summary of the match.",
            "missing_keywords": ["skill1", "skill2"],
            "strategy": "Three professional high-impact suggestions for the CV.",
            "cover_letter": "A high-end, 3-paragraph tailored cover letter."
        }
        """
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": f"JD: {jd_text[:1500]}\nCV: {cv_text[:2000]}"}],
            model="llama-3.3-70b-versatile",
            temperature=0, 
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "AI Analysis failed"}), 500

if __name__ == "__main__":
    # Use port from environment for Render/Heroku
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
