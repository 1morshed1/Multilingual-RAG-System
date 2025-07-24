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

    # --- API Configuration ---
    # Gemini API Key: Loaded from environment variables.
    # It's crucial to set this in your .env file (e.g., GEMINI_API_KEY="your_api_key_here").
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Gemini Model: Using the faster and more cost-effective Gemini 1.5 Flash.
    # It has a large context window (1 million tokens) suitable for RAG.
    GEMINI_MODEL = "gemini-1.5-flash"

    # --- File Paths ---
    # Path to the PDF document to be processed. Using pathlib.Path for robustness.
    PDF_PATH = Path("pdfs/hsc_bangla1.pdf")
    # Path to store the ChromaDB vector database. Using pathlib.Path.
    VECTOR_DB_PATH = Path("./chromadb")

    # --- Text Processing ---
    # Optimized chunking parameters based on empirical tuning for Bengali text.
    # CHUNK_SIZE: The maximum number of characters per text chunk.
    CHUNK_SIZE = 320
    # OVERLAP: The number of characters to overlap between consecutive chunks to maintain context.
    OVERLAP = 40
    # MIN_CHUNK_SIZE: Minimum characters for a chunk to be considered valid.
    MIN_CHUNK_SIZE = 25
    # New: Maximum sentence length for splitting. Sentences longer than this will be
    # heuristically split at conjunctions/discourse markers.
    MAX_SENTENCE_LEN_FOR_SPLIT = 300 # Added this line
    # New: Number of sentences to overlap between chunks for semantic chunking.
    OVERLAP_SENTENCE_COUNT = 2 # Added this line

    # Header and Footer Removal: Percentage of page height to consider as header/footer.
    # These values are slightly increased, likely to better capture running titles or page numbers.
    HEADER_HEIGHT_PERCENTAGE = 0.055
    FOOTER_HEIGHT_PERCENTAGE = 0.055

    # --- Vector Database ---
    # Embedding Model: BAAI/bge-m3 is a state-of-the-art multilingual embedding model
    # known for strong performance in retrieval tasks across many languages, including Bengali.
    EMBEDDING_MODEL = "BAAI/bge-m3"
    # Name of the collection within the vector database, reflecting content and embedding model.
    VECTOR_DB_NAME = "hsc_bangla_bge_m3"

    # --- Retrieval ---
    # TOP_K_RETRIEVAL: Number of most relevant documents to retrieve from the vector database.
    # Tuned to 7, indicating it might benefit from slightly more context than the initial 5.
    TOP_K_RETRIEVAL = 7
    # SIMILARITY_THRESHOLD: Minimum similarity score to consider a retrieved chunk relevant.
    # Tuned to 0.62, suggesting a slightly stricter relevance criterion.
    SIMILARITY_THRESHOLD = 0.62

    # --- Memory Management (for conversational agents) ---
    # MAX_CHAT_HISTORY: Maximum number of previous conversation turns to keep in memory.
    # Keeping 6 turns balances context and token usage efficiently.
    MAX_CHAT_HISTORY = 6
    # CONTEXT_WINDOW: The maximum number of tokens the LLM can process in a single request.
    # Gemini 1.5 Flash has a massive 1,000,000 token context window.
    # We set a practical limit here, much larger than the default 4000.
    CONTEXT_WINDOW = 8192