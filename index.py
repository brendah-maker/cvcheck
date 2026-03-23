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
    # This must be your LIVE Public Key from IntaSend for a real pop-up
    return jsonify({"public_key": os.environ.get("INTASEND_PUBLISHABLE_KEY")})

# WEBHOOK ENDPOINT
@app.route('/api/callback', methods=['POST'])
def payment_callback():
    data = request.json
    # This logs the payment details to your Render logs
    print(f"WEBHOOK RECEIVED: {json.dumps(data)}")
    return jsonify({"status": "ok"}), 200

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
            "You are a strict ATS logic engine. Return ONLY JSON: "
            "score (Integer 0-100), visibility ('HIGH','MEDIUM','LOW'), "
            "missing_count (Integer), verdict (Short string), error_text (Short string)."
        )
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"JD: {jd_text[:1200]}\nCV: {cv_text[:1800]}"}],
            model="llama-3.1-8b-instant",
            temperature=0, 
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        # Ensure score is an integer for NaN protection
        try:
            score_val = str(result.get('score', 0))
            result['score'] = int("".join(filter(str.isdigit, score_val)))
        except: result['score'] = 0
        return jsonify(result)
    except:
        return jsonify({"score": 0, "verdict": "Try again"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        sys_prompt = "Return ONLY JSON: {keywords:[], summary:'', cover_letter:''}"
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"JD: {data.get('jd')[:1000]} CV: {data.get('cv')[:1000]}"}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"error": "Failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
