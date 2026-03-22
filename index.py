import os
from flask import Flask, request, jsonify, render_template
from groq import Groq
import PyPDF2

app = Flask(__name__, template_folder='../templates')

# Initialize Groq Client (Set GROQ_API_KEY in Vercel)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return ""

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
            pdf_text = extract_text_from_pdf(file)
            if pdf_text:
                cv_text = pdf_text

    # Llama 3 Prompt for Analysis
    prompt = f"""
    Analyze the following CV against the Job Description.
    CV: {cv_text[:3000]}
    JD: {jd_text[:2000]}
    
    Return a JSON object with:
    1. "score": (0-100 integer)
    2. "missing_count": (number of missing keywords)
    3. "errors": (number of formatting issues)
    4. "verdict": (A short 1-sentence punchy verdict)
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert ATS recruitment bot. Return ONLY JSON."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192", # Or llama3-70b-8192 for better quality
        response_format={"type": "json_object"}
    )
    
    return chat_completion.choices[0].message.content

@app.route('/generate-docs', methods=['POST'])
def generate_docs():
    data = request.json
    jd = data.get('jd')
    cv = data.get('cv')

    # Llama 3 Prompt for Paid Content
    prompt = f"""
    Based on this CV and Job Description, provide:
    1. keywords: A list of 5 high-impact keywords missing from the CV.
    2. summary: A 3-line powerful professional summary.
    3. cover_letter: A high-conversion cover letter.
    
    CV: {cv[:3000]}
    JD: {jd[:2000]}
    
    Return ONLY a JSON object.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a professional career coach. Return ONLY JSON."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192", # Using the larger Llama model for better writing
        response_format={"type": "json_object"}
    )
    
    return chat_completion.choices[0].message.content


from flask import Flask, request, jsonify, render_template
# ... (rest of your imports and functions) ...

app = Flask(__name__, template_folder='../templates')

# ... (your @app.route functions) ...

# IMPORTANT: Remove app.run(). Vercel handles the execution.


import os
from flask import Flask, request, jsonify, render_template
# ... (your other imports)

# Tell Flask where to find the HTML file
app = Flask(__name__, template_folder='../templates')

# ... (your routes: @app.route('/') and @app.route('/analyze') etc)

# Add this at the very bottom so Render can start it
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
