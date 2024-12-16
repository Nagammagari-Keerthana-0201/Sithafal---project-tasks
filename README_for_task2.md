# Web Scraping and Semantic Similarity Search Tool

This script provides an end-to-end solution for extracting web data, embedding textual content, performing similarity searches, and generating answers using a question-answering model.

## Features

1. **Web Scraping**:
   - Extracts metadata and text from a list of provided URLs using `BeautifulSoup`.
   - Tokenizes and preprocesses sentences using `NLTK`.

2. **Vector Embeddings**:
   - Generates semantic embeddings for sentences using `SentenceTransformers`.

3. **Similarity Search**:
   - Indexes the embeddings using FAISS for efficient nearest-neighbor searches.
   - Retrieves the top-k similar sentences for a user query.

4. **Question Answering**:
   - Uses a pre-trained DistilBERT model to answer questions based on retrieved content.

## Requirements

To use the script, you need the following dependencies installed:

- `requests`: For fetching webpage content.
- `beautifulsoup4`: For HTML parsing.
- `pandas`: For data handling.
- `numpy`: For numerical computations.
- `nltk`: For sentence tokenization.
- `sentence-transformers`: For generating embeddings.
- `faiss-cpu`: For similarity search.
- `transformers`: For question-answering capabilities.

Install all dependencies using:
```bash
pip install -r requirements_code2.txt
