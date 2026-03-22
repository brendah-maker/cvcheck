import os
import json
import PyPDF2
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__, template_folder='templates')

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-payment-config')
def get_config():
    # Make sure this name is EXACTLY what is in Render
    key = os.environ.get("INTASEND_PUBLISHABLE_KEY")
    return jsonify({"public_key": key})

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            reader = PyPDF2.PdfReader(file)
            cv_text = "".join([page.extract_text() for page in reader.pages])

    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": "Return ONLY JSON."},
                      {"role": "user", "content": f"Analyze CV match: {cv_text[:2000]} vs JD: {jd_text[:1500]}"}],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"score": 0, "verdict": "Error"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": "Return ONLY JSON."},
                      {"role": "user", "content": f"Create documents for CV: {data.get('cv')[:1500]} and JD: {data.get('jd')[:1000]}"}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"error": "Failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
