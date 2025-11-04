from langchain_chroma import Chroma
from langchain_core.documents import Document
from models import Word
from langchain_ollama import OllamaEmbeddings
import os
from config import Config
from rich import print

from database import get_all_words


def _create_vectorstore():
    """Creates a Chroma vector store from the SQLite database using Ollama embeddings."""
    words = get_all_words()
    if not words:
        print("No words found in the database to create a vector store.")
        return None

    documents = [
        Document(
            page_content=word.embed_format(),
            metadata=word.model_dump(),
        )
        for word in words
    ]
    print(f"Created {len(documents)} documents from the database.")
    embeddings = OllamaEmbeddings(model=Config.OLLAMA_EMBEDDING_MODEL)
    print(f"Creating Chroma vector store at {Config.CHROMA_DB_PATH}...")
    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=Config.CHROMA_DB_PATH
    )
    print("Chroma vector store created and persisted.")
    return vectorstore

def _get_vectorstore():
    """Loads the Chroma vector store from disk."""
    embeddings = OllamaEmbeddings(model=Config.OLLAMA_EMBEDDING_MODEL)
    return Chroma(persist_directory=Config.CHROMA_DB_PATH, embedding_function=embeddings)

def get_or_create_vectorstore():
    """Gets the Chroma vector store if it exists, otherwise creates it."""
    if not os.path.exists(Config.CHROMA_DB_PATH) or not os.listdir(Config.CHROMA_DB_PATH):
        print("Vector store not found. Creating a new one...")
        return _create_vectorstore()
    else:
        # print("Vector store found. Loading existing store...")
        return _get_vectorstore()

def add_word_to_vectorstore(word: Word):
    """Add a new word to the Chroma vector store."""
    vectorstore = _get_vectorstore()
    embeddings = OllamaEmbeddings(model=Config.OLLAMA_EMBEDDING_MODEL)
    
    # Create a document for the new word
    document = Document(
        page_content=word.embed_format(),
        metadata=word.model_dump()
    )
    
    # Add to vectorstore
    vectorstore.add_documents([document])
    print(f"Added '{word.keyword}' to vector store.")

if __name__ == "__main__":
    vectorstore = get_or_create_vectorstore()