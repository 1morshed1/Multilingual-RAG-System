import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
from config import Config
import logging

logger = logging.getLogger(__name__)

class OptimizedVectorStore:
    def __init__(self):
        self.config = Config()
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=self.config.VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=self.config.VECTOR_DB_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List[Dict]):
        """Add documents with batch processing and error handling"""
        try:
            texts = [chunk['text'] for chunk in chunks]

            # Generate embeddings in batches for efficiency
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                # Convert numpy array to list before extending
                all_embeddings.extend(batch_embeddings.tolist())

            # Prepare data for ChromaDB
            ids = [chunk['id'] for chunk in chunks]
            documents = texts
            
            metadatas = [self._prepare_metadata(chunk) for chunk in chunks]

            # Add to collection
            self.collection.add(
                embeddings=all_embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Successfully added {len(chunks)} documents to vector store")

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def _prepare_metadata(self, chunk: Dict) -> Dict:
        """
        Prepare metadata for ChromaDB storage, ensuring all fields are correctly
        accessed from the nested 'metadata' dictionary within the chunk and
        defaulting to non-None values if a key is missing or its value is None.
        """
        chunk_metadata = chunk.get('metadata', {}) 

        return {
            'page_number': chunk_metadata.get('page_number', 0),
            'sentence_count': chunk_metadata.get('sentence_count', 0), 
            'char_count': chunk_metadata.get('char_count', 0), 
            'word_count': chunk_metadata.get('word_count', 0), 
            'has_tables_on_page': chunk_metadata.get('has_tables_on_page', False), 
            'source': chunk_metadata.get('source', ""), 
            'chapter': chunk_metadata.get('chapter', ""), 
            'contains_dialogue': chunk_metadata.get('contains_dialogue', False), 
            'named_entities': ','.join(chunk_metadata.get('named_entities', [])), 
            'bengali_percentage': chunk_metadata.get('bengali_percentage', 0.0), 
            'complexity_score': chunk_metadata.get('complexity_score', 0.0) 
        }

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Enhanced retrieval with re-ranking and filtering"""
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL

        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])

            # Retrieve from vector store
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(top_k * 2, 10)  
            )

            # Re-rank results based on multiple factors
            candidates = []
            if results and results.get('documents') and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    candidate = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0.0,
                        'id': results['ids'][0][i]
                    }
                    candidates.append(candidate)

            # Apply re-ranking logic
            re_ranked = self._rerank_candidates(query, candidates)

            return re_ranked[:top_k]

        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []

    def _rerank_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Re-rank candidates based on multiple signals"""
        query_lower = query.lower()

        for candidate in candidates:
            score = 1.0 - candidate['distance']  
            text_lower = candidate['text'].lower()

            # Boost score for exact keyword matches
            if any(word in text_lower for word in query_lower.split()):
                score += 0.1

            
            entities_str = candidate['metadata'].get('named_entities', '')
            entities = [e.strip() for e in entities_str.split(',') if e.strip()]
            if any(entity.lower() in query_lower for entity in entities):
                score += 0.15

            
            if candidate['metadata'].get('contains_dialogue', False) and any(word in query_lower for word in ['বলে', 'বলল', 'বললেন']): # Default to False
                score += 0.05

            candidate['final_score'] = score

        
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)