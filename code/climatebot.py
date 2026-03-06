# Script: Test deployment of climate chatbot

# To run app - Step 1: conda activate your_RAG_env_name
# To run app - Step 2: cd "directory_where_climatebot.py_is_located"
# To run app - Step 3: streamlit run climatebot.py

# Import libraries
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os

with open('../keys/your_key_txt_file.txt', 'r') as key_file:
    os.environ['OPENAI_API_KEY'] = key_file.read().strip()

# Set up Streamlit app
st.title("IPCC (climate scientists) vs. NIPCC (climate skeptics) Comparative RAG")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'),
                                model="text-embedding-3-small")

# Load FAISS vector databases for IPCC and NIPCC - created by the 'RAG_example_ipcc_nipcc.py' script
vector_db_ipcc = FAISS.load_local("../Data/ipcc_nipcc/vector_db_ipcc.faiss", embeddings, allow_dangerous_deserialization = True)
vector_db_nipcc = FAISS.load_local("../Data/ipcc_nipcc/vector_db_nipcc.faiss", embeddings, allow_dangerous_deserialization = True)

# Create a document retriever
retriever_ipcc = vector_db_ipcc.as_retriever()
retriever_nipcc = vector_db_nipcc.as_retriever()
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key= os.environ['OPENAI_API_KEY'])

# Create a system prompt that instructs the model to only use the provided context and not to introduce any new information. This is crucial for ensuring that the model's responses are based solely on the retrieved documents and do not include any hallucinated facts or opinions. The prompt should also explicitly state that if the model cannot answer the question based on the provided context, it should say it doesn't know. This helps to set clear expectations for the model's behavior and encourages it to be honest about its limitations.
system_prompt = (
    "You are a helpful interpreter of a given set of information. Use ONLY the given context to answer the question."
    "It is crucial that you do not introduce any new information."
    "If you cannot answer the question solely based on the provided context, say you don't know."
    "{context}"
)

# Create a prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a document and retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

chain_ipcc = create_retrieval_chain(retriever_ipcc, question_answer_chain)
chain_nipcc = create_retrieval_chain(retriever_nipcc, question_answer_chain)

# Streamlit input for question
question = st.text_input("Ask a question about climate science, and I will provide the facts, as well as the alternative facts:")
if question:
    st.write("Based on the IPCC reports, my answer is:")
    # Answer
    response1 = chain_ipcc.invoke({"input": question})['answer']
    st.write(response1)
    
    st.write("Based on the NIPCC reports, my answer is:")
    # Answer
    response2 = chain_nipcc.invoke({"input": question})['answer']
    st.write(response2)
    
# Add a button to export the responses to a latex table
# Create a LaTeX table with the responses
latex_table = r"""
\begin{table}[ht]
    \centering
    \begin{tabular}{|l|l|l|}
        \hline
        Query & Response: IPCC & Response: NIPCC \\
        \hline
        """ + question + r""" & """ + response1 + r""" & """ + response2 + r""" \\
        \hline
    \end{tabular}
    \caption{Responses to Climate Questions}
    \label{tab:climate_responses}
\end{table}
""" 
# Provide an option to download the LaTeX table
st.download_button(
    label="Download LaTeX Table",
    data=latex_table,
    file_name="climate_responses.tex",
    mime="text/plain"
)

# Create a CSV file with the responses
csv_data = f"Source,Question,Answer\n"
csv_data += f"IPCC,{question},{response1}\n"
csv_data += f"NIPCC,{question},{response2}\n"

# Provide an option to download the CSV file
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="climate_responses.csv",
    mime="text/csv"
)   