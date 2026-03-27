import os
import json
import pdfplumber
import io
import requests
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from groq import Groq

app = Flask(__name__, template_folder='templates')

# --- Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///payments.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
INTASEND_PUBLISHABLE_KEY = os.environ.get("INTASEND_PUBLISHABLE_KEY")
INTASEND_SECRET_KEY = os.environ.get("INTASEND_SECRET_KEY")
IS_LIVE = os.environ.get("IS_LIVE", "False").lower() == "true"
API_BASE = "https://api.intasend.com/api/v1" if IS_LIVE else "https://sandbox.intasend.com/api/v1"

client = Groq(api_key=GROQ_API_KEY)

# --- Database Model ---
class Payment(db.Model):
    id = db.Column(db.String(100), primary_key=True) # IntaSend Invoice ID
    status = db.Column(db.String(20), default="pending")
    amount = db.Column(db.Float, nullable=True)

with app.app_context():
    db.create_all()

# --- Helper Functions ---
def format_phone(phone):
    """Formats phone number to 254XXXXXXXXX format"""
    phone = phone.strip().replace("+", "")
    if phone.startswith("0"):
        return "254" + phone[1:]
    if phone.startswith("7") or phone.startswith("1"):
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
    
    if 'cv_file' in request.files:
        file = request.files['cv_file']
        if file.filename != '':
            try:
                with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                    cv_text = " ".join([page.extract_text() or "" for page in pdf.pages])
            except Exception as e:
                return jsonify({"error": "Failed to read PDF"}), 400

    if not jd_text or not cv_text:
        return jsonify({"error": "Missing JD or CV content"}), 400

    try:
        # We increase character limits slightly as Llama 3 handles context well
        sys_prompt = "You are an ATS System. Return ONLY valid JSON."
        user_prompt = f"""
        Compare CV and JD. Return JSON: 
        {{ "score": int, "visibility": "High/Med/Low", "missing_count": int, "verdict": "string", "error_text": "" }}
        JD: {jd_text[:2000]}
        CV: {cv_text[:3000]}
        """
        
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({"score": 0, "verdict": "AI Analysis failed"}), 500

@app.route('/stkpush', methods=['POST'])
def stk_push():
    try:
        data = request.get_json()
        phone = format_phone(data.get("phone", ""))
        amount = data.get("amount", 20)

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
        
        res = requests.post(f"{API_BASE}/payment/mpesa-stk-push/", json=payload, headers=headers)
        res_data = res.json()
        
        if res.status_code not in [200, 201]:
            return jsonify({"error": "Payment initialization failed", "details": res_data}), 400

        inv_id = res_data.get("invoice", {}).get("invoice_id")
        if inv_id:
            new_pay = Payment(id=inv_id, status="pending", amount=amount)
            db.session.add(new_pay)
            db.session.commit()
            return jsonify({"checkout_id": inv_id})
        
        return jsonify({"error": "No Invoice ID received"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/check-payment/<id>')
def check_payment(id):
    # Check local DB first to save API calls
    payment = Payment.query.get(id)
    if payment and payment.status == "COMPLETE":
        return jsonify({"status": "paid"})

    headers = {"Authorization": f"Bearer {INTASEND_SECRET_KEY}"}
    try:
        res = requests.get(f"{API_BASE}/payment/status/{id}/", headers=headers)
        data = res.json()
        state = data.get("invoice", {}).get("state") # COMPLETE, FAILED, PENDING
        
        if state == "COMPLETE":
            if payment:
                payment.status = "COMPLETE"
                db.session.commit()
            return jsonify({"status": "paid"})
        
        return jsonify({"status": state.lower()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional career coach. Return ONLY JSON."},
                {"role": "user", "content": f"Create optimized content. JD: {data.get('jd')[:1500]} CV: {data.get('cv')[:2000]}. Return JSON: {{'keywords':[], 'summary':'', 'cover_letter':''}}"}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"error": "Document generation failed"}), 500

if __name__ == "__main__":
    # Use port from environment (Render requirement)
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
