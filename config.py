class Config:
    DICTIONARY_JSON_PATH = "Dictionary.json"
    DB_PATH = "dictionary.db"
    CHROMA_DB_PATH = "./chroma_db"
    OLLAMA_EMBEDDING_MODEL = "embeddinggemma:300m"  # You can change this to another Ollama embedding model
    OLLAMA_LLM_MODEL = "gemma3n:e4b"  # You can change this to another Ollama LLM model (e.g., "mistral")
