import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

import chromadb
import logging
import sys
import tensorflow as tf
import warnings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

warnings.filterwarnings("ignore", category=FutureWarning)

def compute_loss(labels, logits):
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global query_engine
query_engine = None

def init_llm():
    llm = Ollama(model="llama3", request_timeout=3600.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model

def init_index(embed_model):
    reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
    documents = reader.load_data()

    logging.info("Indexing %d documents", len(documents))

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("iollama")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    return index

def init_query_engine(index):
    global query_engine

    template = (
        "You are an expert in operating systems, with access to all relevant documentation and study materials. "
        "Your goal is to provide clear, detailed answers to questions about operating systems.\n\n"
        "Here is the context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the information above, respond to the following query:\n\n"
        "Question: {query_str}\n\n"
        "Provide a concise and clear answer using your technical knowledge of operating systems."
    )
    qa_template = PromptTemplate(template)

    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

    return query_engine

def chat(input_question, user):
    global query_engine

    if query_engine is None:
        logging.error("Query engine is not initialized")
        raise RuntimeError("Query engine is not initialized")

    response = query_engine.query(input_question)
    logging.info("Got response from LLM - %s", response)

    return response.response

def chat_cmd():
    global query_engine

    while True:
        input_question = input("Enter your question (or 'exit' to quit): ")
        if input_question.lower() == 'exit':
            break

        if query_engine is None:
            logging.error("Query engine is not initialized")
            print("Query engine is not initialized")
            continue

        response = query_engine.query(input_question)
        logging.info("Got response from LLM - %s", response)
        print(response.response)

if __name__ == '__main__':
    try:
        init_llm()
        index = init_index(Settings.embed_model)
        init_query_engine(index)
    except Exception as e:
        logging.error("Initialization error: %s", str(e))
        sys.exit(1)

    chat_cmd()
