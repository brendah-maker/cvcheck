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
                return jsonify({"error": "Failed to read PDF file"}), 400

    if not jd_text or not cv_text:
        return jsonify({"error": "Missing input data"}), 400

    try:
        # SYSTEM PROMPT: Forces consistency and strict JSON output
        sys_prompt = """
        You are an expert ATS (Applicant Tracking System) Analyzer. 
        Perform a deep-dive comparison between the Resume and Job Description.
        
        SCORING RULES (Strict Consistency):
        1. Keywords Match (40%): Hard skills, tools, and technical terms.
        2. Experience Match (40%): Relevant titles, years of experience, and industry.
        3. Quality & Formatting (20%): Education, clarity, and certifications.
        
        Return ONLY valid JSON:
        {
            "score": int,
            "visibility": "High" | "Medium" | "Low",
            "verdict": "A 1-sentence blunt truth about the match.",
            "missing_keywords": ["list", "of", "missing", "skills"],
            "strategy": "3 sentences on exactly what to change in the CV.",
            "cover_letter": "A professional 3-paragraph tailored cover letter."
        }
        """

        user_content = f"JOB DESCRIPTION:\n{jd_text[:1500]}\n\nRESUME:\n{cv_text[:2000]}"

        # Temperature 0 ensures the same input generates the same output
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0, 
            response_format={"type": "json_object"}
        )
        
        return jsonify(json.loads(response.choices[0].message.content))

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "AI Analysis failed"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
