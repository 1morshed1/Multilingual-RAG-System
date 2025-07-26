import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class for the RAG (Retrieval-Augmented Generation) system.
    This class centralizes all tunable parameters for clarity and easy modification.
    """

    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    
    GEMINI_MODEL = "gemini-1.5-flash"

    
    PDF_PATH = Path("pdfs/hsc_bangla1.pdf")
    # Path to store the ChromaDB vector database. Using pathlib.Path.
    VECTOR_DB_PATH = Path("./chromadb")

    
    # Optimized chunking parameters based on empirical tuning for Bengali text.
    # CHUNK_SIZE: The maximum number of characters per text chunk.
    CHUNK_SIZE = 320
    # OVERLAP: The number of characters to overlap between consecutive chunks to maintain context.
    OVERLAP = 40
    # MIN_CHUNK_SIZE: Minimum characters for a chunk to be considered valid.
    MIN_CHUNK_SIZE = 25
    
    MAX_SENTENCE_LEN_FOR_SPLIT = 300 
    
    OVERLAP_SENTENCE_COUNT = 2 

    
    HEADER_HEIGHT_PERCENTAGE = 0.055
    FOOTER_HEIGHT_PERCENTAGE = 0.055

    
    EMBEDDING_MODEL = "BAAI/bge-m3"
    
    VECTOR_DB_NAME = "hsc_bangla_bge_m3"

    
    TOP_K_RETRIEVAL = 7
    
    SIMILARITY_THRESHOLD = 0.62

    
    MAX_CHAT_HISTORY = 6
    
    CONTEXT_WINDOW = 8192