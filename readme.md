Here’s the formatted README.md file you can copy and paste directly:

# Predict App

**Predict App** is a Flask-based application designed for processing and analyzing uploaded documents. It uses OCR, machine learning, and natural language processing to extract and predict supplier information and remit-to addresses from PDFs and other document formats.

## Features

- **Document Upload**: Upload PDFs and other document formats for processing.
- **OCR Processing**: Extract text from uploaded documents using Tesseract.
- **Supplier Prediction**: Utilize machine learning (SentenceTransformer) and NLP (spaCy, Fuzzy Matching) to predict suppliers and addresses.
- **Interactive Viewer**: View processed documents directly within the app.
- **Search and Match**: Match text from documents with existing supplier and address databases.
- **CSV Uploads**: Import and manage supplier and address databases.
- **Prediction Feedback**: Leverage GPT-based feedback for prediction accuracy.

---

## Installation

### Prerequisites
- Python 3.11 or later
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- Virtual environment tools (e.g., `venv` or `virtualenv`)

### Steps
1. Clone the repository:
   ```bash
   git clone git@github.com:tehrenstrom/predict_app.git
   cd predict_app

   2. Create a virtual environment and activate it:

python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate


   3. Install dependencies:

pip install -r requirements.txt


   4. Configure OpenAI API Key:
   •  Update the openai.api_key in app.py with your API key.
   5. Run the application:

python app.py


   6. Access the app at:
   •  http://127.0.0.1:5000/

Usage
   1. Upload Files:
   •  Use the interface to upload documents and CSV files.
   2. Process Documents:
   •  Click the “Scan Documents” button to process uploaded files.
   3. View Results:
   •  View aggregated results, supplier matches, and OCR text.
   4. Predict Suppliers:
   •  Leverage AI to predict suppliers for uploaded documents.

Contributing

Feel free to submit issues and pull requests for bug fixes or improvements.
   1. Fork the repository.
   2. Create a feature branch: git checkout -b feature-name.
   3. Commit changes: git commit -m 'Add new feature'.
   4. Push to the branch: git push origin feature-name.
   5. Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Author

Developed by Travis Ehrenstrom.
For inquiries, contact: travisehrenstrom@gmail.com.

Save this content as `README.md` in your project directory. Then, commit and push it to GitHub:

```bash
git add README.md
git commit -m "Add README.md"
git push origin main