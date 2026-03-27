import os
import json
import pdfplumber
import io
import requests
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from groq import Groq

app = Flask(__name__, template_folder='templates')

# --- Database Setup ---
# Note: On Render, SQLite will reset on every deploy unless using a Persistent Disk.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///payments.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Payment(db.Model):
    id = db.Column(db.String(100), primary_key=True) # IntaSend Invoice ID
    status = db.Column(db.String(20), default="pending")

with app.app_context():
    db.create_all()

# --- Configuration & Env Vars ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
INTASEND_PUBLISHABLE_KEY = os.environ.get("INTASEND_PUBLISHABLE_KEY")
INTASEND_SECRET_KEY = os.environ.get("INTASEND_SECRET_KEY")
# Set IS_LIVE to "True" in Render Env Vars for production
IS_LIVE = os.environ.get("IS_LIVE", "False").lower() == "true"

API_BASE = "https://api.intasend.com/api/v1" if IS_LIVE else "https://sandbox.intasend.com/api/v1"

client = Groq(api_key=GROQ_API_KEY)

# --- Helper Functions ---
def format_phone(phone):
    """Formats phone to 254XXXXXXXXX"""
    phone = phone.strip().replace("+", "")
    if phone.startswith("0"):
        return "254" + phone[1:]
    elif phone.startswith("7") or phone.startswith("1"):
        return "254" + phone
    return phone

# --- Routes ---
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
    
    # Handle PDF Upload
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            try:
                with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                    cv_text = " ".join([page.extract_text() or "" for page in pdf.pages])
            except Exception as e:
                return jsonify({"score": 0, "verdict": "PDF Read Error"}), 400

    try:
        sys_prompt = "You are an ATS Parser. Return ONLY JSON. Format: {score: int, visibility: str, missing_count: int, verdict: str, error_text: str}"
        user_content = f"Job Description: {jd_text[:1500]}\n\nResume: {cv_text[:2000]}"
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt}, 
                {"role": "user", "content": user_content}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        print(f"GROQ ERROR: {str(e)}")
        return jsonify({"score": 0, "verdict": "AI Analysis Failed"}), 500

@app.route('/stkpush', methods=['POST'])
def stk_push():
    try:
        data = request.get_json()
        raw_phone = data.get("phone", "").strip()
        phone = format_phone(raw_phone)
        amount = data.get("amount", 20)
        
        payload = {
            "public_key": INTASEND_PUBLISHABLE_KEY,
            "amount": amount,
            "phone_number": phone,
            "api_ref": "CVCheck_Diagnostic"
        }
        
        headers = {
            "Authorization": f"Bearer {INTASEND_SECRET_KEY}",
            "Content-Type": "application/json"
        }
        
        print(f"DEBUG: Initiating STK Push to {API_BASE} for {phone}")
        
        res = requests.post(f"{API_BASE}/payment/mpesa-stk-push/", json=payload, headers=headers)
        res_data = res.json()
        
        if res.status_code not in [200, 201]:
            print(f"INTASEND AUTH ERROR: {res_data}")
            return jsonify({"error": "Payment Failed", "details": res_data}), 400

        inv_id = res_data.get("invoice", {}).get("invoice_id")
        if inv_id:
            db.session.add(Payment(id=inv_id, status="pending"))
            db.session.commit()
            return jsonify({"checkout_id": inv_id})
        
        return jsonify({"error": "No Invoice ID returned"}), 400

    except Exception as e:
        print(f"SERVER ERROR: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/check-payment/<id>')
def check_payment(id):
    headers = {"Authorization": f"Bearer {INTASEND_SECRET_KEY}"}
    try:
        # 1. Check IntaSend Status
        res = requests.get(f"{API_BASE}/payment/status/{id}/", headers=headers)
        data = res.json()
        state = data.get("invoice", {}).get("state") # COMPLETE, PROCESSING, or FAILED
        
        # 2. Update Database if paid
        if state == "COMPLETE":
            payment = Payment.query.get(id)
            if payment:
                payment.status = "paid"
                db.session.commit()
            return jsonify({"status": "paid"})
        
        return jsonify({"status": "pending", "state": state})
    except Exception as e:
        print(f"CHECK ERROR: {str(e)}")
        return jsonify({"status": "error"}), 500

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Return ONLY JSON: {keywords:[], summary:'', cover_letter:''}"}, 
                {"role": "user", "content": f"Optimize for JD: {data.get('jd')[:1000]} CV: {data.get('cv')[:1000]}"}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"error": "Doc generation failed"}), 500

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
