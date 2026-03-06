# Script: Set up example RAG - IPCC vs. NIPCC reports
# This script processes IPCC and NIPCC report text files, generates embeddings, and saves them

# Import libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
with open('../keys/your_key_txt_file.txt', 'r') as key_file:
    os.environ['OPENAI_API_KEY'] = key_file.read().strip()
from langchain_core.documents import Document
import zipfile


# Process input text data from txt files generated from IPCC and NIPCC reports
def read_text_from_txt_or_zip(path: str) -> str:
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            txt_files = [name for name in zf.namelist() if name.lower().endswith(".txt")]
            if not txt_files:
                raise FileNotFoundError(f"No .txt file found inside zip: {path}")
            with zf.open(txt_files[0]) as f:
                return f.read().decode("utf-8")

    raise ValueError(f"Unsupported file type: {path}")


ipcc_zip = "../data/ipcc_nipcc/IPCC_consolidated.zip"
ipcc_txt = "../data/ipcc_nipcc/IPCC_consolidated.txt"
nipcc_zip = "../data/ipcc_nipcc/NIPCC_consolidated.zip"
nipcc_txt = "../data/ipcc_nipcc/NIPCC_consolidated.txt"

ipcc_path = ipcc_zip if os.path.exists(ipcc_zip) else ipcc_txt
nipcc_path = nipcc_zip if os.path.exists(nipcc_zip) else nipcc_txt

ipcc = read_text_from_txt_or_zip(ipcc_path)
nipcc = read_text_from_txt_or_zip(nipcc_path)
        

text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)


# Split into chunks for efficient retrieval
def create_documents(chunks, source):
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = {"source": source, "chunk_id": i}
        documents.append(Document(page_content=chunk, metadata=metadata))
    return documents 

chunks_ipcc = create_documents(text_splitter.split_text(ipcc), "ipcc")
chunks_nipcc = create_documents(text_splitter.split_text(nipcc), "nipcc")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'),
                                model="text-embedding-3-small")


# Generate embeddings with a progress bar
embedded_vectors = []
texts = []

for doc in tqdm(chunks_ipcc, desc="Generating embeddings"):
    if doc.page_content:  # Ensure page_content is not empty
        vector = embeddings.embed_query(doc.page_content)  # Compute embedding
        if vector:  # Ensure vector is not empty
            embedded_vectors.append(vector)
            texts.append(doc.page_content)
        else:
            print(f"Warning: Empty vector generated for document with content: {doc.page_content[:50]}...")
    else:
        print("Warning: Document with empty page content encountered and skipped.")


# Save embedded_vectors and texts objects
# Save objects
np.save("../Data/ipcc_nipcc/embedded_vectors_ipcc_temp.npy", np.array(embedded_vectors))
with open("../Data/ipcc_nipcc/texts_ipcc_temp.txt", "w", encoding="utf-8") as f:
    f.writelines(text + "\n" for text in texts)

# Save chunks_ipcc as a serialized object
with open("../Data/ipcc_nipcc/chunks_ipcc_temp.pkl", "wb") as f:
    pickle.dump(chunks_ipcc, f)

metadata = [doc.metadata for doc in chunks_ipcc]  # Extract metadata - assumes metadata is a dictionary with relevant info for each chunk, adjust as needed


# Convert to FAISS format
vector_db_case1 = FAISS.from_embeddings(
    list(zip(texts, embedded_vectors)),  # Combine texts and embeddings correctly
    embeddings,  # The embedding model instance (required)
    metadatas=metadata  # Optional metadata
)

# Save FAISS index
vector_db_case1.save_local("../Data/ipcc_nipcc/vector_db_ipcc.faiss")
print("FAISS vector database saved successfully!")


# Generate embeddings with a progress bar
embedded_vectors2 = []
texts2 = []

for doc in tqdm(chunks_nipcc, desc="Generating embeddings"):
    vector = embeddings.embed_query(doc.page_content)  # Compute embedding
    embedded_vectors2.append(vector)
    texts2.append(doc.page_content)
    
metadata2 = [doc.metadata for doc in chunks_nipcc]  # Extract metadata


vector_db_case2 = FAISS.from_embeddings(
    list(zip(texts2, embedded_vectors2)),  # Combine texts and embeddings correctly
    embeddings,  # The embedding model instance (required)
    metadatas=metadata2  # Optional metadata
)


# Save FAISS index
vector_db_case2.save_local("../Data/ipcc_nipcc/vector_db_nipcc.faiss")
print("FAISS vector database saved successfully!")



