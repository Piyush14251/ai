import os
from flask import Flask, request, render_template, flash
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import openai
import shutil
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set the Tesseract path and OpenAI API key from environment variables
tesseract_path = shutil.which('tesseract')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    raise EnvironmentError("Tesseract is not installed or it's not in your PATH. See README file for more information.")

openai.api_key = os.getenv('OPENAI_API_KEY')

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'

# Verify Tesseract installation
try:
    tesseract_version = subprocess.run(['tesseract', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    app.logger.info(f'Tesseract version: {tesseract_version.stdout}')
except Exception as e:
    app.logger.error(f'Error verifying Tesseract installation: {e}')
    raise EnvironmentError("Tesseract is not installed or it's not in your PATH. See README file for more information.")

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return images

def ocr_images(images):
    ocr_text = ""
    for image in images:
        ocr_text += pytesseract.image_to_string(image)
    return ocr_text

def split_text(text, chunk_size=8000):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_test_cases(text, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\n{text}"}
        ],
        max_tokens=1500
    )
    return response['choices'][0]['message']['content'].strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        pdf_file = request.files['pdf']
        
        if pdf_file:
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
            pdf_file.save(pdf_path)
            
            try:
                pdf_text = extract_text_from_pdf(pdf_path)
                images = extract_images_from_pdf(pdf_path)
                ocr_text = ocr_images(images)
                combined_text = pdf_text + ocr_text
                
                text_chunks = split_text(combined_text)
                test_cases = []
                for chunk in text_chunks:
                    test_cases.append(generate_test_cases(chunk, prompt))
                
                final_test_cases = "\n\n".join(test_cases)
                return render_template('index.html', prompt=prompt, test_cases=final_test_cases)
            except Exception as e:
                app.logger.error(f"Error processing the PDF: {e}")
                flash(f"Error processing the PDF: {e}")
    
    return render_template('index.html', prompt='', test_cases='')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
