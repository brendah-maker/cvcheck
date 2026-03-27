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
                return jsonify({"error_text": "Could not read the PDF file."}), 400

    if not jd_text or not cv_text:
        return jsonify({"error_text": "Please provide both a Job Description and a CV."}), 400

    try:
        # SYSTEM PROMPT: Strict instructions to ensure consistency
        sys_prompt = """
        You are a professional ATS (Applicant Tracking System) and Career Coach. 
        Analyze the match between the provided Resume and Job Description.
        
        SCORING RUBRIC (BE CONSISTENT):
        - 40 points: Keyword matching (Hard skills).
        - 40 points: Experience relevance (Industry & Years).
        - 20 points: Formatting, Education, and Certifications.
        
        Return ONLY a JSON object with this exact structure:
        {
            "score": int, 
            "visibility": "High" | "Medium" | "Low", 
            "missing_count": int, 
            "verdict": "short description", 
            "keywords": ["list", "of", "missing", "keywords"],
            "summary": "3-sentence professional improvement summary",
            "cover_letter": "A short, high-impact tailored cover letter draft"
        }
        """

        user_content = f"JOB DESCRIPTION:\n{jd_text[:2000]}\n\nRESUME:\n{cv_text[:3000]}"

        # Using the 70B model for higher accuracy and temperature 0 for consistency
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0, # ZERO temperature ensures the same input gives the same output
            response_format={"type": "json_object"}
        )
        
        analysis_results = json.loads(response.choices[0].message.content)
        return jsonify(analysis_results)

    except Exception as e:
        print(f"AI Error: {str(e)}")
        return jsonify({"error_text": "The AI is currently busy. Please try again in a moment."}), 500

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
