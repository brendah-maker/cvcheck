import os
import json
import pdfplumber
import io
import uuid # For generating unique transaction IDs
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates')

from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-payment-config')
def get_config():
    key = os.environ.get("INTASEND_PUBLISHABLE_KEY")
    return jsonify({"public_key": key})

# NEW WEBHOOK CALLBACK ROUTE
@app.route('/api/callback', methods=['POST'])
def payment_callback():
    data = request.json
    print(f"Payment Webhook Received: {data}")
    # Here you would typically save the transaction to a database
    return jsonify({"status": "received"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            try:
                # FIXED: Using io.BytesIO to read file from memory
                with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                    cv_text = ""
                    for page in pdf.pages:
                        cv_text += page.extract_text() or ""
            except Exception as e:
                return jsonify({"error": "PDF read error"}), 400

    try:
        sys_prompt = "You are an ATS. Return ONLY JSON: {score, verdict, visibility, missing_count, errors, gap_teaser, format_teaser}."
        user_prompt = f"JD: {jd_text[:1000]}\nCV: {cv_text[:1500]}"
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt},{"role": "user", "content": user_prompt}],
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
        sys_prompt = "Senior Career Coach. Return ONLY JSON: {keywords:[], summary:'', cover_letter:''}"
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt},{"role": "user", "content": f"JD: {data.get('jd')[:1000]} CV: {data.get('cv')[:1000]}"}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"error": "Fail"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
