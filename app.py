import os
from flask import Flask, request, render_template, flash
import fitz  # PyMuPDF
from PIL import Image
import io
import openai
import shutil
import logging
from dotenv import load_dotenv
import pytesseract

# Load environment variables
load_dotenv()

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = os.getenv('TESSDATA_PREFIX')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.urandom(24)

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set the Tesseract path
tesseract_path = shutil.which('tesseract')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    logging.error("Tesseract is not installed or it's not in your PATH.")
    print("Tesseract is not installed or it's not in your PATH.")

# Print environment and PATH for debugging
print("Environment PATH:", os.environ['PATH'])
print("Tesseract path:", tesseract_path)
print("TESSDATA_PREFIX:", os.getenv('TESSDATA_PREFIX'))

openai.api_key = os.getenv('OPENAI_API_KEY')

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
                flash('Test cases generated successfully!', 'success')
                return render_template('index.html', prompt=prompt, test_cases=final_test_cases)

            except Exception as e:
                logging.error(f"Error processing the PDF: {e}")
                flash(f"An error occurred: {e}", 'danger')

    return render_template('index.html', prompt='', test_cases='')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
