import logging
import os
from typing import Dict, Any
from preprocess import TextPreprocessor
from vector_store import OptimizedVectorStore
from rag_pipeline import RAGPipeline
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualRAGSystem:
    def __init__(self):
        self.config = Config()
        self.preprocessor = TextPreprocessor()
        self.vector_store = OptimizedVectorStore()
        self.rag_pipeline = RAGPipeline()
        self.is_initialized = False

    def initialize(self, force_rebuild: bool = False):
        """Initialize the system with document processing"""
        if self.is_initialized and not force_rebuild:
            logger.info("System already initialized")
            return

        logger.info("Initializing RAG system...")

        # Check if PDF exists
        if not os.path.exists(self.config.PDF_PATH):
            raise FileNotFoundError(f"PDF not found: {self.config.PDF_PATH}")

        # Check if vector store already exists and skip processing if not force_rebuild
        if not force_rebuild and self.vector_store.collection.count() > 0:
            logger.info("Vector store already contains documents, skipping processing")
            self.is_initialized = True
            return

        # Extract and process text
        pages_data = self.preprocessor.extract_text_from_pdf(self.config.PDF_PATH)
        chunks = self.preprocessor.smart_chunk_text(pages_data)

        # Add to vector store
        self.vector_store.add_documents(chunks)

        self.is_initialized = True
        logger.info("RAG system initialized successfully")

    def query(self, user_query: str) -> Dict[str, Any]:
        """Process user query and return answer"""
        if not self.is_initialized:
            self.initialize()

        # Retrieve relevant contexts
        contexts = self.vector_store.retrieve(user_query)

        # Debug logging
        logger.info(f"Retrieved {len(contexts)} chunks for query: '{user_query}'")
        for i, context in enumerate(contexts):
            logger.info(f" Chunk {i+1} (Page {context['metadata']['page_number']}): {context['text'][:200]}...")

        # Generate answer (synchronous call)
        result = self.rag_pipeline.generate_answer(
            user_query,
            contexts,
            self.rag_pipeline.chat_history
        )

        # Add to conversation history
        self.rag_pipeline.add_to_history(user_query, result['answer'])

        return {
            'query': user_query,
            'answer': result['answer'],
            'language': result['language'],
            'confidence': result['confidence'],
            'sources': result['sources'],
            'context_count': len(contexts)
        }

if __name__ == "__main__":
    # Initialize system
    rag_system = MultilingualRAGSystem()
    rag_system.initialize()

    print("\nMultilingual RAG System Ready!")
    print("Type your query in English or Bengali. Type 'exit' to quit.")

    while True:
        user_input = input("\nYour Query: ")
        if user_input.lower() == 'exit':
            print("Exiting RAG system. Goodbye!")
            break

        result = rag_system.query(user_input)
        print(f"\n❓ Query: {result['query']}")
        print(f"✅ Answer: {result['answer']}")
        print(f"🔍 Confidence: {result['confidence']:.2f}")
        print(f"📚 Sources: {result['context_count']} chunks")
        
        