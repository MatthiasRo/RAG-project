# From RAGs to Riches

### An Efficient Pipeline for Exploratory Text Analysis

This repository implements the **Comparative Retrieval-Augmented
Generation (RAG) pipeline** proposed in the working paper:

> Roesti, Matthias, From RAGs to (feature) Riches - 
> An Efficient Pipeline for Exploratory Text Analysis (June 29, 2025). 
> Available at SSRN: http://dx.doi.org/10.2139/ssrn.5331899 

The project demonstrates how modern **retrieval-augmented language
models** can be used to efficiently explore, compare, and extract
structured insights from **large text corpora**.

The core idea is to adapt RAG architectures---commonly used for
chatbots---into a **research pipeline for comparative text analysis**,
allowing researchers to systematically query multiple corpora and
compare their responses.

------------------------------------------------------------------------

# Repository Structure

    RAG_project/
    │
    ├── README.md
    ├── codes/
    │   ├── RAG_example_ipcc_nipcc.py
    │   └── climatebot.py
    │
    ├── data/
    │   └── ipcc_nipcc/
    │       ├── IPCC_consolidated.zip
    │       └── NIPCC_consolidated.zip
    │
    └── keys/
        └── your_key_txt_file.txt

### Folder descriptions

| Folder | Purpose |
| :--- | :--- |
| `codes/` | Implementation scripts for the RAG pipeline |
| `data/ipcc_nipcc/` | Example corpus used in the paper |
| `keys/` | Location for API keys (not committed to version control) |
  
------------------------------------------------------------------------

# Conceptual Overview

The pipeline implements a **comparative RAG architecture** that enables
systematic comparison across multiple corpora.

### Step 1 --- Pre-processing and Embedding

-   Clean and preprocess text
-   Split text into overlapping chunks
-   Convert chunks into vector embeddings
-   Store embeddings in a vector database

### Step 2 --- Retrieval

-   Embed the user query
-   Retrieve the most similar text chunks from the vector database

### Step 3 --- Response Generation

-   Pass retrieved chunks to an LLM
-   Instruct the model to **use only the retrieved context**
-   Produce grounded answers

------------------------------------------------------------------------

# Implementation Example

## Comparative Climate Narratives: IPCC vs. NIPCC

The repository includes a working example comparing two large text
corpora:

-   **IPCC reports** (climate science consensus)
-   **NIPCC reports** (climate skepticism)

The goal is to demonstrate how the pipeline can extract **contrasting
viewpoints** from different text collections using identical queries.

------------------------------------------------------------------------

# Installation

## 1. Clone the repository

``` bash
cd "project_folder"
git clone https://github.com/MatthiasRo/RAG-project.git
```

## 2. Create a Python environment

``` bash
conda create -n rag_pipeline python=3.10
conda activate rag_pipeline
```

## 3. Install dependencies

``` bash
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install faiss-cpu
pip install pandas
pip install numpy
pip install tqdm
pip install streamlit
```

## 4. Add your OpenAI API key (if using OpenAI embeddings as in the example code)

Create:

    keys/your_key_txt_file.txt

with contents:

    sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

------------------------------------------------------------------------

# Running the Pipeline

The implementation consists of **two scripts**.

------------------------------------------------------------------------

# Step 1 --- Build the Vector Databases

Script:

    codes/RAG_example_ipcc_nipcc.py

Purpose:

-   Reads the IPCC and NIPCC report corpora
-   Splits them into chunks
-   Generates embeddings
-   Stores them in FAISS vector databases

Run:

``` bash
python codes/RAG_example_ipcc_nipcc.py
```

This generates:

    data/ipcc_nipcc/vector_db_ipcc.faiss
    data/ipcc_nipcc/vector_db_nipcc.faiss

------------------------------------------------------------------------

# Step 2 --- Launch the Comparative RAG Interface

Script:

    codes/climatebot.py

Run:

``` bash
streamlit run codes/climatebot.py
```

The Streamlit app allows interactive queries comparing IPCC vs NIPCC
responses.

Example question:

    Is climate change primarily caused by human activity?

The system retrieves relevant passages separately from each corpus and
generates grounded responses.

------------------------------------------------------------------------

# Exporting Results

The interface allows exporting answers as:

-   **LaTeX tables** for academic papers
-   **CSV files** for downstream analysis

------------------------------------------------------------------------

# Extending the Pipeline

The architecture is modular and can easily be adapted.

### Alternative Embedding Models

-   text-embedding-3-small
-   text-embedding-3-large
-   Qwen embeddings
-   sentence-transformers

### Alternative Vector Databases

-   FAISS
-   Milvus
-   Pinecone
-   Weaviate

### Alternative LLMs

-   GPT models
-   Claude
-   Gemini
-   Llama

------------------------------------------------------------------------

# Potential Research Applications

The pipeline is designed for **comparative text analysis at scale**,
including:

-   Political discourse comparison
-   Media framing analysis
-   Corporate communication analysis
-   Policy document comparison
-   Scientific literature synthesis

------------------------------------------------------------------------

# Citation

If you use this codebase in research, please cite:

    Roesti, Matthias, From RAGs to (feature) Riches - An Efficient Pipeline for Exploratory Text Analysis (June 29, 2025). 
    Available at SSRN: http://dx.doi.org/10.2139/ssrn.5331899 

------------------------------------------------------------------------

# License

This project is licensed under the MIT License – see the LICENSE file for details.

------------------------------------------------------------------------

# Author

Matthias Roesti - [matthiasroesti.net](https://www.matthiasroesti.net/)


