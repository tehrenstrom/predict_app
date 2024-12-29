import os
import csv
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util
import spacy
from fuzzywuzzy import fuzz, process
import openai
import re

# Initialize Flask app
app = Flask(__name__)

# Directories and Configurations
UPLOAD_FOLDER_SUPPLIERS = 'uploads/suppliers'
UPLOAD_FOLDER_ADDRESSES = 'uploads/addresses'
STATIC_FOLDER_DOCUMENTS = 'static/documents'
RESULTS_FILE = 'results.json'

os.makedirs(UPLOAD_FOLDER_SUPPLIERS, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_ADDRESSES, exist_ok=True)
os.makedirs(STATIC_FOLDER_DOCUMENTS, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'pdf', 'png', 'jpg', 'jpeg'}

document_results = {}
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_lg")
print("Loaded model:", nlp.meta["name"])

openai.api_key = 'your-openai-api-key-here'

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def save_results():
    with open(RESULTS_FILE, 'w') as file:
        json.dump(document_results, file, indent=4)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    log(f"Extracting text from PDF: {filepath}")
    try:
        images = convert_from_path(filepath)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image, config='--psm 6') + "\n"
        return text
    except Exception as e:
        log(f"Error extracting text from PDF: {e}", level="ERROR")
        return ""

from fuzzywuzzy import fuzz, process
from sentence_transformers import util
import spacy

# Assuming embedding_model and nlp are already initialized

def fuzzy_match(text, items, threshold=80):
    """
    Fuzzy matches the text against a list of items.
    Normalizes the input and applies a consistent threshold.
    """
    if not items:
        log("No items provided for fuzzy matching", level="WARNING")
        return []
    
    # Normalize input
    text = text.strip().lower()
    items = [item.strip().lower() for item in items]
    
    matches = process.extract(text, items, scorer=fuzz.partial_ratio)
    
    # Filter by a consistent threshold
    results = [
        {"item": match[0], "score": match[1]} 
        for match in matches if match[1] >= threshold
    ]
    return sorted(results, key=lambda x: x['score'], reverse=True)

def sentence_transformer_match(text, items, similarity_threshold=0.75):
    """
    Uses SentenceTransformer for semantic matching.
    Processes input in batches and normalizes scores.
    Includes robust input validation and logging for debugging.
    """
    if not items:
        log("No items provided for SentenceTransformer matching.", level="WARNING")
        return []

    if not text.strip():
        log("Input text is empty or invalid for SentenceTransformer matching.", level="WARNING")
        return []

    # Normalize input
    text = text.strip()
    items = [item.strip() for item in items if item.strip()]

    if not items:
        log("Filtered items list is empty after normalization.", level="WARNING")
        return []

    try:
        # Encode text and items
        log(f"Encoding text: {text}")
        text_embedding = embedding_model.encode(text, convert_to_tensor=True)
        log(f"Encoding {len(items)} items for matching.")
        item_embeddings = embedding_model.encode(items, convert_to_tensor=True, batch_size=16)

        # Compute cosine similarity
        log("Computing cosine similarity.")
        scores = util.cos_sim(text_embedding, item_embeddings)[0]

        results = [
            {"item": items[i], "score": float(scores[i])}
            for i in range(len(items)) if scores[i] >= similarity_threshold
        ]

        log(f"SentenceTransformer match results: {results}")
        return sorted(results, key=lambda x: x['score'], reverse=True)
    except Exception as e:
        log(f"Error during SentenceTransformer matching: {e}", level="ERROR")
        return []

def spacy_match(text, items, similarity_threshold=0.75):
    """
    Matches text against items using spaCy similarity.
    Processes items in a batch for efficiency and handles edge cases.
    """
    if not items:
        log("No items provided for spaCy matching.", level="WARNING")
        return []

    if not text.strip():
        log("Input text is empty or invalid for spaCy similarity.", level="WARNING")
        return []

    # Create the spaCy Doc object for the input text
    doc = nlp(text.lower())
    if doc.vector_norm == 0:
        log("Input text has empty vectors; skipping spaCy similarity.", level="WARNING")
        return []

    # Preprocess and batch-create spaCy Doc objects for items
    preprocessed_items = [item.lower() for item in items]
    item_docs = list(nlp.pipe(preprocessed_items))

    results = []
    for item, item_doc in zip(preprocessed_items, item_docs):
        if item_doc.vector_norm == 0:
            log(f"Skipping item '{item}' due to empty vectors.", level="WARNING")
            continue

        similarity = doc.similarity(item_doc)
        if similarity >= similarity_threshold:
            results.append({"item": item, "score": similarity * 100})

    # Sort results by score in descending order
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    if not results:
        log("No spaCy matches exceeded the similarity threshold.", level="INFO")

    log(f"spaCy match results: {results}")
    return results

def sanitize_text(text):
    """
    Converts text to lowercase and removes special characters except currency symbols.
    """
    return re.sub(r'[^a-zA-Z0-9\s$€£]', '', text.lower())

def load_suppliers():
    filepath = os.path.join(UPLOAD_FOLDER_SUPPLIERS, 'suppliers.csv')
    if not os.path.exists(filepath):
        return []
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        return [
            {
                "id": row["Supplier_ID"],
                "name": sanitize_text(row["Supplier_Name"])
            }
            for row in reader
        ]

def load_addresses():
    filepath = os.path.join(UPLOAD_FOLDER_ADDRESSES, 'addresses.csv')
    if not os.path.exists(filepath):
        return []
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        return [sanitize_text(row["Formatted_Address"]) for row in reader]

def get_aggregated_supplier(spacy_results, fuzzy_results, sentence_results, suppliers):
    """
    Aggregates results from spaCy, fuzzy matching, and SentenceTransformer.
    Returns the best-matched supplier name and its corresponding ID.
    """
    all_results = spacy_results + fuzzy_results + sentence_results
    if not all_results:
        return "No result", "No ID"

    # Normalize and aggregate scores by supplier name
    scores = {}
    for res in all_results:
        item = res.get("item").strip().lower()  # Normalize item name
        score = res.get("score", 0)
        scores[item] = scores.get(item, 0) + score

    # Find the best match
    best_supplier = max(scores, key=scores.get) if scores else "No result"

    # Map to Supplier ID
    supplier_id = next(
        (supplier["id"] for supplier in suppliers if supplier["name"].strip().lower() == best_supplier),
        "No ID"  # Default if no match is found
    )

    return best_supplier.capitalize(), supplier_id

def get_aggregated_address(spacy_results, fuzzy_results, sentence_results, addresses):
    """
    Aggregates results from spaCy, fuzzy matching, and SentenceTransformer.
    Returns the best-matched address.
    """
    all_results = spacy_results + fuzzy_results + sentence_results
    if not all_results:
        return "No result"

    # Aggregate scores by address
    scores = {}
    for res in all_results:
        item = res.get("item")
        score = res.get("score", 0)
        scores[item] = scores.get(item, 0) + score

    # Find the best match
    best_address = max(scores, key=scores.get) if scores else "No result"

    return best_address

@app.route('/')
def index():
    document_files = os.listdir(STATIC_FOLDER_DOCUMENTS)
    document_summaries = []

    uploaded_supplier_file = next((f for f in os.listdir(UPLOAD_FOLDER_SUPPLIERS) if f.endswith('.csv')), "No file uploaded")
    uploaded_address_file = next((f for f in os.listdir(UPLOAD_FOLDER_ADDRESSES) if f.endswith('.csv')), "No file uploaded")

    suppliers = load_suppliers()

    for filename in document_files:
        result = document_results.get(filename, {})
        document_summaries.append({
            "filename": filename,
            "has_results": bool(result),
            "aggregated_supplier": result.get("aggregated_supplier", "No result"),
            "supplier_id": result.get("supplier_id", "No ID"),  # Add supplier_id
            "aggregated_address": result.get("aggregated_address", "No result"),
            "address_supplier_id": result.get("address_supplier_id", "No ID")  # Add address_supplier_id
        })

    return render_template('index.html',
                           document_summaries=document_summaries,
                           uploaded_supplier_file=uploaded_supplier_file,
                           uploaded_address_file=uploaded_address_file)

@app.route('/upload_files', methods=['POST'])
def upload_files():
    supplier_file = request.files.get('supplier_file')
    address_file = request.files.get('address_file')
    documents = request.files.getlist('documents[]')

    if supplier_file and allowed_file(supplier_file.filename):
        supplier_file.save(os.path.join(UPLOAD_FOLDER_SUPPLIERS, secure_filename(supplier_file.filename)))
    if address_file and allowed_file(address_file.filename):
        address_file.save(os.path.join(UPLOAD_FOLDER_ADDRESSES, secure_filename(address_file.filename)))
    for document in documents:
        if document and allowed_file(document.filename):
            document.save(os.path.join(STATIC_FOLDER_DOCUMENTS, secure_filename(document.filename)))

    return redirect(url_for('index'))

@app.route('/delete_document/<filename>', methods=['POST'])
def delete_document(filename):
    file_path = os.path.join(STATIC_FOLDER_DOCUMENTS, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        document_results.pop(filename, None)
        save_results()
    return redirect(url_for('index'))

@app.route('/view_document/<filename>')
def view_document(filename):
    # Ensure the file exists in the STATIC_FOLDER_DOCUMENTS
    file_path = os.path.join(STATIC_FOLDER_DOCUMENTS, filename)
    if not os.path.exists(file_path):
        return f"Document {filename} not found", 404

    # Fetch the document results if they exist
    result = document_results.get(filename, {})
    ocr_text = result.get("ocr_text", "")

    # Collect logs
    logs = []  # You can define how you want to collect logs here
    if "logs" in result:
        logs = result["logs"]  # If logs are stored in the result

    # Render the document.html template with the relevant data
    return render_template(
        'document.html',
        filename=filename,
        result=result,
        ocr_text=ocr_text,
        logs=logs  # Pass logs to the template
    )

@app.route('/delete_all', methods=['POST'])
def delete_all():
    for filename in os.listdir(STATIC_FOLDER_DOCUMENTS):
        file_path = os.path.join(STATIC_FOLDER_DOCUMENTS, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            document_results.pop(filename, None)
    save_results()
    return redirect(url_for('index'))

@app.route('/delete_supplier_csv', methods=['POST'])
def delete_supplier_csv():
    supplier_filepath = os.path.join(UPLOAD_FOLDER_SUPPLIERS, 'suppliers.csv')
    if os.path.exists(supplier_filepath):
        os.remove(supplier_filepath)
    return redirect(url_for('index'))

@app.route('/delete_address_csv', methods=['POST'])
def delete_address_csv():
    address_filepath = os.path.join(UPLOAD_FOLDER_ADDRESSES, 'addresses.csv')
    if os.path.exists(address_filepath):
        os.remove(address_filepath)
    return redirect(url_for('index'))

@app.route('/scan_documents', methods=['POST'])
def scan_documents():
    document_files = os.listdir(STATIC_FOLDER_DOCUMENTS)
    suppliers = load_suppliers()
    supplier_names = [supplier['name'] for supplier in suppliers]
    addresses = load_addresses()

    for filename in document_files:
        file_path = os.path.join(STATIC_FOLDER_DOCUMENTS, filename)
        
        log(f"Starting OCR for file: {filename}")
        ocr_text = extract_text_from_pdf(file_path)
        log(f"OCR completed for {filename}: {ocr_text[:100]}...")  # Log the first 100 characters of OCR

        # Perform matching for suppliers
        log(f"Starting supplier matching for {filename}")
        supplier_spacy_results = spacy_match(ocr_text, supplier_names)
        supplier_fuzzy_results = fuzzy_match(ocr_text, supplier_names)
        supplier_sentence_results = sentence_transformer_match(ocr_text, supplier_names)

        # Perform matching for addresses
        log(f"Starting address matching for {filename}")
        address_spacy_results = spacy_match(ocr_text, addresses)
        address_fuzzy_results = fuzzy_match(ocr_text, addresses)
        address_sentence_results = sentence_transformer_match(ocr_text, addresses)

        # Aggregate supplier results
        aggregated_supplier, supplier_id = get_aggregated_supplier(
            supplier_spacy_results, supplier_fuzzy_results, supplier_sentence_results, suppliers
        )

        # Aggregate address results
        aggregated_address = get_aggregated_address(
            address_spacy_results, address_fuzzy_results, address_sentence_results, addresses
        )

        # Save results
        log(f"Saving results for {filename}")
        document_results[filename] = {
            "ocr_text": ocr_text,
            "aggregated_supplier": aggregated_supplier,
            "supplier_id": supplier_id,
            "aggregated_address": aggregated_address,
            "spacy_results": supplier_spacy_results,
            "fuzzy_results": supplier_fuzzy_results,
            "sentence_results": supplier_sentence_results,
        }

    save_results()
    log("Document scanning and processing completed.")
    return redirect(url_for('index'))

@app.route('/view_supplier_csv')
def view_supplier_csv():
    filepath = os.path.join(UPLOAD_FOLDER_SUPPLIERS, 'suppliers.csv')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = f.read().splitlines()
        return render_template('view_csv.html', title="Supplier CSV", data=data)
    return redirect(url_for('index'))

@app.route('/view_address_csv')
def view_address_csv():
    filepath = os.path.join(UPLOAD_FOLDER_ADDRESSES, 'addresses.csv')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = f.read().splitlines()
        return render_template('view_csv.html', title="Address CSV", data=data)
    return redirect(url_for('index'))

@app.route('/predict_suppliers', methods=['POST'])
def predict_suppliers():
    """
    Placeholder route for OpenAI-based supplier prediction.
    """
    predictions = {}

    for filename, result in document_results.items():
        ocr_text = result.get("ocr_text", "")

        # Simulated OpenAI interaction
        openai_prompt = f"""
        The following text was extracted from a document:
        {ocr_text}

        Predict the supplier name based on the text and existing supplier database.
        """
        try:
            # Replace this with the actual OpenAI API call when ready
            response = openai.Completion.create(
                model="gpt-4",
                prompt=openai_prompt,
                max_tokens=150,
                temperature=0.5,
            )
            predicted_supplier = response.choices[0].text.strip()
        except Exception as e:
            predicted_supplier = f"Error: {e}"

        predictions[filename] = predicted_supplier
        # Update document results
        document_results[filename]["predicted_supplier"] = predicted_supplier

    save_results()
    return render_template('index.html', predictions=predictions, document_summaries=document_results)

if __name__ == "__main__":
    app.run(debug=True)