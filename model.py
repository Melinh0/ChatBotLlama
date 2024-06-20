import os
import chromadb
import logging
import sys
import pandas as pd

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Document)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global query_engine
query_engine = None

def init_llm():
    llm = Ollama(model="qwen:1.8B", request_timeout=300.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model

def read_parquet_files(input_dir):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".parquet"):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_parquet(file_path)
            for _, row in df.iterrows():
                document_content = ' '.join(map(str, row.values))
                yield Document(text=document_content)

def init_index(embed_model):
    reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
    text_documents = reader.load_data()

    documents = text_documents + list(read_parquet_files(input_dir="./docs"))
    logging.info("index creating with `%d` documents", len(documents))

    chroma_client = chromadb.EphemeralClient()

    try:
        chroma_collection = chroma_client.create_collection("iollama")
    except chromadb.db.base.UniqueConstraintError:
        chroma_collection = chroma_client.get_collection("iollama")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    return index

def init_query_engine(index):
    global query_engine

    # Custom prompt template
    template = (
        "Imagine you are an advanced AI expert in cyber security laws, with access to all current and relevant legal documents, "
        "case studies, and expert analyses. Your goal is to provide insightful, accurate, and concise answers to questions in this domain.\n\n"
        "Here is some context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry with detailed references to applicable laws, "
        "precedents, or principles where appropriate:\n\n"
        "Question: {query_str}\n\n"
        "Answer succinctly, starting with the phrase 'According to cyber security law,' and ensure your response is understandable to someone without a legal background."
    )
    qa_template = PromptTemplate(template)

    # Build query engine with custom template
    # text_qa_template specifies custom template
    # similarity_top_k configures the retriever to return the top 3 most similar documents,
    # the default value of similarity_top_k is 2
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

    return query_engine

def chat(input_question, user):
    global query_engine

    response = query_engine.query(input_question)
    logging.info("got response from llm - %s", response)

    return response.response

# Initialize the LLM and embeddings
init_llm()

# Initialize the index with documents and CSV data
index = init_index(Settings.embed_model)

# Initialize the query engine
query_engine = init_query_engine(index)
