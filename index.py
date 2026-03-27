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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///payments.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Payment(db.Model):
    id = db.Column(db.String(100), primary_key=True) # Invoice ID
    status = db.Column(db.String(20), default="pending")

with app.app_context():
    db.create_all()

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
INTASEND_PUBLISHABLE_KEY = os.environ.get("INTASEND_PUBLISHABLE_KEY")
INTASEND_SECRET_KEY = os.environ.get("INTASEND_SECRET_KEY")
IS_LIVE = os.environ.get("IS_LIVE", "False").lower() == "true"

API_BASE = "https://api.intasend.com/api/v1" if IS_LIVE else "https://sandbox.intasend.com/api/v1"

client = Groq(api_key=GROQ_API_KEY)

def format_phone(phone):
    phone = phone.strip().replace("+", "")
    if phone.startswith("0"): return "254" + phone[1:]
    elif phone.startswith("7") or phone.startswith("1"): return "254" + phone
    return phone

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
            try:
                with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                    cv_text = " ".join([page.extract_text() or "" for page in pdf.pages])
            except: pass

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Return ONLY JSON: {score:int, visibility:str, missing_count:int, verdict:str, error_text:str}"},
                {"role": "user", "content": f"JD: {jd_text[:1000]} CV: {cv_text[:1500]}"}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except:
        return jsonify({"score": 0, "verdict": "AI Error"}), 500

@app.route('/stkpush', methods=['POST'])
def stk_push():
    try:
        data = request.get_json()
        phone = format_phone(data.get("phone", ""))
        
        payload = {
            "public_key": INTASEND_PUBLISHABLE_KEY,
            "amount": data.get("amount", 20),
            "phone_number": phone,
            "api_ref": "CVCheck"
        }
        headers = {"Authorization": f"Bearer {INTASEND_SECRET_KEY}", "Content-Type": "application/json"}
        
        res = requests.post(f"{API_BASE}/payment/mpesa-stk-push/", json=payload, headers=headers)
        res_data = res.json()
        
        if res.status_code not in [200, 201]:
            return jsonify({"error": "Failed", "details": res_data}), 400

        inv_id = res_data.get("invoice", {}).get("invoice_id")
        if inv_id:
            db.session.add(Payment(id=inv_id, status="pending"))
            db.session.commit()
            return jsonify({"checkout_id": inv_id})
        
        return jsonify({"error": "No Invoice ID"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NEW: WEBHOOK CALLBACK HANDLER ---
@app.route('/api/callback', methods=['POST'])
def payment_callback():
    data = request.get_json()
    invoice_id = data.get("invoice_id")
    state = data.get("state") # COMPLETE, FAILED, etc.

    if invoice_id and state == "COMPLETE":
        payment = Payment.query.get(invoice_id)
        if payment:
            payment.status = "paid"
            db.session.commit()
            print(f"WEBHOOK: Invoice {invoice_id} marked as PAID")
            
    return jsonify({"status": "ok"}), 200

@app.route('/check-payment/<id>')
def check_payment(id):
    # 1. Check our database first (updated by webhook)
    payment = Payment.query.get(id)
    if payment and payment.status == "paid":
        return jsonify({"status": "paid"})

    # 2. If not marked paid yet, check IntaSend API (Polling)
    headers = {"Authorization": f"Bearer {INTASEND_SECRET_KEY}"}
    try:
        res = requests.get(f"{API_BASE}/payment/status/{id}/", headers=headers)
        api_state = res.json().get("invoice", {}).get("state")
        
        if api_state == "COMPLETE":
            if payment:
                payment.status = "paid"
                db.session.commit()
            return jsonify({"status": "paid"})
            
        return jsonify({"status": api_state.lower()})
    except:
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
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
