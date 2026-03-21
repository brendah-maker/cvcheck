import os
from flask import Flask, request, jsonify, render_template, send_file
from groq import Groq
from fpdf import FPDF
import io

app = Flask(__name__)

# Get your free key at https://console.groq.com/
client = Groq(api_key="YOUR_GROQ_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_cv():
    data = request.json
    jd, cv_text = data.get('jd'), data.get('cv')

    # Llama 3 8B: Lightning fast for the "Teaser"
    prompt = f"Analyze this CV against this JD. Return JSON ONLY with: 'score' (0-100), 'missing_count' (int), 'errors' (int). JD: {jd} CV: {cv_text}"
    
    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        response_format={"type": "json_object"}
    )
    return chat.choices[0].message.content

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    # M-Pesa Verification Logic should go here
    data = request.json
    jd, cv_text = data.get('jd'), data.get('cv')

    # Llama 3 70B: High Intelligence for the Rewrite
    prompt = f"""
    System: Expert HR. Return JSON ONLY.
    1. 'keywords': [7 skills missing]
    2. 'summary': '3-sentence professional summary'
    3. 'cover_letter': 'Professional letter'
    JD: {jd} CV: {cv_text}
    """

    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        response_format={"type": "json_object"}
    )
    return chat.choices[0].message.content

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    data = request.json
    content = data.get('content') # The text from the AI

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Simple PDF formatting
    pdf.multi_cell(0, 10, txt=content)
    
    # Save PDF to memory and send to user
    pdf_output = io.BytesIO()
    pdf_string = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_string)
    pdf_output.seek(0)

    return send_file(pdf_output, attachment_filename="CV_Check_Optimized.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
