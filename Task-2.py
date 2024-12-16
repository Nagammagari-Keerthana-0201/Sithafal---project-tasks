import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import pipeline
import pickle

nltk.download('punkt')
nltk.download('punkt_tab')

model = SentenceTransformer("all-MiniLM-L6-v2")
qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
}

def extract_data(url):
    print(f"\nExtracting data from: {url}")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        metadata = {}
        for meta_tag in soup.find_all('meta'):
            name = meta_tag.get('name') or meta_tag.get('property')
            content = meta_tag.get('content')
            if name and content:
                metadata[name] = content

        paragraphs = [p.get_text() for p in soup.find_all('p')]
        sentences = nltk.sent_tokenize(" ".join(paragraphs))

        return {
            'url': url,
            'metadata': metadata,
            'sentences': sentences
        }
    else:
        print(f"Failed to fetch webpage {url}: {response.status_code}")
        return None

def scrape_urls(urls):
    scraped_data = []
    for url in urls:
        data = extract_data(url)
        if data:
            scraped_data.append(data)
    return scraped_data

def data_to_chunks(scraped_data):
    sentences = []
    for entry in scraped_data:
        sentences.extend(entry.get('sentences', []))
    return sentences

def vector_embedding(chunks):
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def similarity_search_data(embeddings, metadata):
    embed = np.array(embeddings).astype(np.float32)  
    dim = embed.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embed)

    with open("metadata.pkl", "wb") as f:  
        pickle.dump(metadata, f)

    return index

def finding_similarities(query, index, sentences, metadata):
    query_vector = model.encode([query]).astype(np.float32)  
    k = 5
    D, I = index.search(query_vector, k)
    top_sentences = [sentences[i] for i in I[0]]
    top_metadata = [metadata[i] for i in I[0]]
    return top_sentences, top_metadata

# LLM Integration
def response(query, top_sentences, top_metadata):
    context = "\n".join([f"{meta}: {sent}" for meta, sent in zip(top_metadata, top_sentences)])
    result = qa_model(question=query, context=context)
    return result['answer']

def main():
    urls = [
        "https://www.uchicago.edu/",
        "https://www.washington.edu/",
        "https://www.stanford.edu/",
        "https://und.edu/"
    ]

    scraped_data = scrape_urls(urls)

    sentences = data_to_chunks(scraped_data)

    embeddings = vector_embedding(sentences)

    metadata = []
    for item in scraped_data:
        metadata.extend([item['metadata']] * len(item['sentences']))

    index = similarity_search_data(embeddings, metadata)

    query = input("Enter your Query: \n")
    
    top_sentences, top_metadata = finding_similarities(query, index, sentences, metadata)

    for i, sentence in enumerate(top_sentences):
        print(f"{i + 1}. {sentence}")

    result = response(query, top_sentences, top_metadata)
    print(f"Relevant Answer: {result}")

if __name__ == "__main__":
    main()
