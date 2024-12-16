# PDF and Unemployment Rate Extraction Tool

This project includes a Python script that extracts text and tables from PDFs, performs text embeddings using Sentence Transformers, and performs similarity searches. It is designed to retrieve specific unemployment data based on different degree types and extract tabular data from PDFs.

## Features

1. **Extract Text from PDF**: 
   - Downloads a PDF from a given URL.
   - Extracts text from specified pages of the PDF.
   
2. **Text Chunking & Embeddings**:
   - Splits the extracted text into smaller chunks.
   - Generates embeddings for the chunks using the `all-MiniLM-L6-v2` Sentence Transformer model.

3. **Similarity Search**:
   - Searches for relevant chunks of text based on a user input query using FAISS.

4. **Unemployment Data Extraction**:
   - Provides unemployment rates for different degrees (Bachelor's, Master's, etc.).
   
5. **Table Extraction**:
   - Extracts tables from specified pages in the PDF using `pdfplumber`.

---

## Prerequisites

Make sure to install the following dependencies to run the script:

- **Python 3.x**
- The required libraries listed in the `requirements.txt`.

### Installation

1. Clone the repository or download the code.
   
2. Navigate to the project directory:
   ```bash
   cd /path/to/your/project
