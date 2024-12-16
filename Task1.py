import numpy as np
import os
import fitz
import faiss
from sentence_transformers import SentenceTransformer
import pdfplumber
import json
import gdown
import pytesseract
from PIL import Image
import io
import re

degree_unemployment = {
    "Doctoral degree": 2.2,
    "Professional degree": 2.3,
    "Master’s degree": 3.4,
    "Bachelor’s degree": 4.0,
    "Associate’s degree": 5.4,
    "Some college, no degree": 7.0,
    "High school diploma": 7.5,
    "Less than a high school diploma": 11.0,
}

def pdf_from_url(pdf_url):
    try:
        output_path = gdown.download(pdf_url, quiet=False)
        return output_path
    except Exception as e:
        print(f"Failed to download PDF: {e}")
        return None

def text_from_pdf(pdf_path, page_numbers):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in page_numbers:
            if page_num < len(doc):
                page = doc[page_num]
                text += page.get_text()
            else:
                print(f"Page {page_num + 1} is out of range.")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def splitting_data(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def embed_text(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings

def similarity_search_data(embed):
    embed = np.array(embed)
    dim = embed.shape[1]
    ind = faiss.IndexFlatL2(dim)
    ind.add(embed)
    return ind

def relevant_data(ind, chunks, prompt, embed_model):
    query_embed = embed_model.encode([prompt])
    search = ind.search(query_embed, k=1)
    index = int(search[1][0][0])
    return chunks[index]

def extract_table_from_page(pdf_path, page_number):
    """
    Extracts the table from the specified page number of the PDF.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < len(pdf.pages):
                page = pdf.pages[page_number]
                table = page.extract_table()
                return table
            else:
                raise IndexError(f"Page {page_number + 1} is out of range.")
    except Exception as e:
        print(f"Error extracting table from page {page_number + 1}: {e}")
        return None

def main():
    pdf_url = "https://drive.google.com/uc?id=1KRon-0yCxjnJVVPqWDph2-9_sZpbRNbw"
    pdf_path = pdf_from_url(pdf_url)

    if not pdf_path:
        print("Failed to download PDF.")
        return
    page_numbers = [1, 5]
    pdf_text = text_from_pdf(pdf_path, page_numbers)
    if not pdf_text:
        print("Failed to extract text from the PDF.")
        return

    chunks = splitting_data(pdf_text)
    embeddings = embed_text(chunks)
    ind = similarity_search_data(embeddings)

    print("\nGet the exact unemployment information based on the type of Degree from page 2")
    degree_type = input("Enter the Type of a Degree: ").strip()

    if degree_type in degree_unemployment:
        print(f"Answer: The unemployment rate for {degree_type} is {degree_unemployment[degree_type]}%")
    else:
        print(f"Answer: Degree type not found.")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nExtracting tabular data from page 6...")
    table_data = extract_table_from_page(pdf_path, 5)  
    
    if table_data:
        print("Extracted Table:")
        for row in table_data:
            print(row)
    else:
        print("Failed to extract tabular data from page 6.")

if __name__ == "__main__":
    main()