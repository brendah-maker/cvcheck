import os
import json
import pdfplumber
import io
import requests
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from groq import Groq

app = Flask(__name__, template_folder='templates')

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///payments.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Payment(db.Model):
    id = db.Column(db.String(100), primary_key=True)
    status = db.Column(db.String(20), default="pending")

with app.app_context():
    db.create_all()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# MAKE SURE THESE NAMES MATCH YOUR RENDER ENV VARS
INTASEND_PUBLISHABLE_KEY = os.environ.get("INTASEND_PUBLISHABLE_KEY")
INTASEND_SECRET_KEY = os.environ.get("INTASEND_SECRET_KEY")
IS_LIVE = os.environ.get("IS_LIVE", "False").lower() == "true"
API_BASE = "https://api.intasend.com/api/v1" if IS_LIVE else "https://sandbox.intasend.com/api/v1"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-payment-config')
def get_config():
    return jsonify({"public_key": INTASEND_PUBLISHABLE_KEY})

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_text = request.form.get('jd_text', '')
    cv_text = request.form.get('cv_text', '')
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                cv_text = "".join([page.extract_text() or "" for page in pdf.pages])
    try:
        sys_prompt = "Return ONLY JSON: {score: int, visibility: str, missing_count: int, verdict: str, error_text: str}"
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"JD: {jd_text[:1000]} CV: {cv_text[:1500]}"}],
            model="llama-3.1-8b-instant",
            temperature=0,
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"score": 0, "verdict": "AI Error"}), 500

@app.route('/stkpush', methods=['POST'])
def stk_push():
    try:
        data = request.get_json()
        phone = data.get("phone", "").strip()
        amount = data.get("amount", 20)
        
        # Phone Formatting
        if phone.startswith("0"): phone = "254" + phone[1:]
        elif not phone.startswith("254"): phone = "254" + phone

        payload = {
            "public_key": INTASEND_PUBLISHABLE_KEY,
            "amount": amount,
            "phone_number": phone,
            "api_ref": "CVCheck"
        }
        headers = {
            "Authorization": f"Bearer {INTASEND_SECRET_KEY}",
            "Content-Type": "application/json"
        }
        
        # Log request for debugging
        print(f"DEBUG: Sending to {API_BASE} with phone {phone}")
        
        res = requests.post(f"{API_BASE}/payment/mpesa-stk-push/", json=payload, headers=headers)
        res_data = res.json()
        
        # Log exact error from IntaSend to Render Logs
        if res.status_code != 200:
            print(f"INTASEND ERROR: {res_data}")
            return jsonify({"error": "Rejected", "details": res_data}), 400

        inv_id = res_data.get("invoice", {}).get("invoice_id")
        if inv_id:
            db.session.add(Payment(id=inv_id, status="pending"))
            db.session.commit()
            return jsonify({"checkout_id": inv_id})
        
        return jsonify({"error": "No Invoice ID"}), 400
    except Exception as e:
        print(f"SERVER ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/check-payment/<id>')
def check_payment(id):
    headers = {"Authorization": f"Bearer {INTASEND_SECRET_KEY}"}
    try:
        res = requests.get(f"{API_BASE}/payment/status/{id}/", headers=headers)
        state = res.json().get("invoice", {}).get("state")
        if state == "COMPLETE":
            return jsonify({"status": "paid"})
    except: pass
    return jsonify({"status": "pending"})

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": "Return ONLY JSON: {keywords:[], summary:'', cover_letter:''}"}, 
                      {"role": "user", "content": f"JD: {data.get('jd')[:1000]} CV: {data.get('cv')[:1000]}"}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except: return jsonify({"error": "failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
