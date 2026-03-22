import os
import json
import PyPDF2
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

# Initialize Groq with the API key from environment
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
        print(f"PDF Error: {e}")
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

    if not jd_text or not cv_text:
        return jsonify({"score": 0, "verdict": "Please provide both CV and JD"}), 400

    try:
        # UPDATED TO LLAMA 3.1
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a recruitment bot. Respond ONLY with a JSON object."},
                {"role": "user", "content": f"Analyze match between CV: {cv_text[:2500]} and JD: {jd_text[:2000]}. Return JSON: {{'score': 70, 'missing_count': 2, 'errors': 1, 'verdict': 'short punchy summary'}}"}
            ],
            model="llama-3.1-8b-instant", 
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        print(f"Llama Analysis Error: {str(e)}")
        return jsonify({"score": 0, "missing_count": 0, "errors": 0, "verdict": "AI Match Failed"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        # UPDATED TO LLAMA 3.1
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": f"Provide keywords, summary, and cover_letter for this role based on JD: {data.get('jd')[:1500]} and CV: {data.get('cv')[:2000]}"}
            ],
            model="llama-3.1-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        print(f"Llama Docs Error: {str(e)}")
        return jsonify({"error": "Failed to generate docs"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
