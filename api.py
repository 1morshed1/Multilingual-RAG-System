import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


from main import MultilingualRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI application
app = FastAPI(
    title="Multilingual RAG System API",
    description="A lightweight REST API to interact with the Multilingual RAG System for HSC Bangla 1st Paper.",
    version="1.0.0"
)

# Initialize the RAG system globally
# This ensures the system is initialized only once when the API starts
rag_system: Optional[MultilingualRAGSystem] = None

@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG system when the FastAPI application starts up.
    This ensures the PDF is processed and the vector store is ready before
    any API requests are served.
    """
    global rag_system
    rag_system = MultilingualRAGSystem()
    logger.info("Attempting to initialize MultilingualRAGSystem...")
    try:
        
        rag_system.initialize(force_rebuild=False)
        logger.info("MultilingualRAGSystem initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize MultilingualRAGSystem: {e}", exc_info=True)
        

# Pydantic model for conversation turns
class ConversationTurn(BaseModel):
    role: str = Field(..., description="Role of the speaker (e.g., 'user', 'assistant').")
    content: str = Field(..., description="Content of the message.")

# Pydantic model for the API request body
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query in Bengali or English.")
    
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, description="Previous conversation turns to provide context for the current query."
    )

# Pydantic model for the API response body
class QueryResponse(BaseModel):
    query: str = Field(..., description="The original query.")
    answer: str = Field(..., description="The model-generated answer.")
    language: str = Field(..., description="Detected language of the query (e.g., 'bn', 'en').")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the answer (0.0 to 1.0).")
    sources: List[Dict[str, Any]] = Field(..., description="List of source metadata (page_number, snippet, chapter).")
    context_count: int = Field(..., description="Number of context chunks retrieved.")
    error: Optional[str] = Field(None, description="Error message if an issue occurred.")


@app.get("/")
async def read_root():
    """
    Returns a welcome message to confirm the API is running.
    You can visit http://127.0.0.1:8000/docs for interactive API documentation.
    """
    return {"message": "Welcome to the Multilingual RAG System API! Visit /docs for API documentation."}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Processes a user query using the RAG system and returns a model-generated answer.
    """
    global rag_system
    if rag_system is None or not rag_system.is_initialized:
        logger.error("RAG system is not initialized. Cannot process query.")
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready. Please wait for initialization or check server logs."
        )

    logger.info(f"Received query: '{request.query}'")
    
    
    formatted_history = []
    if request.conversation_history:
        for turn in request.conversation_history:
            formatted_history.append({"role": turn.role, "content": turn.content})

    try:
        
        response_data = rag_system.query(request.query)

        

        return QueryResponse(
            query=response_data['query'],
            answer=response_data['answer'],
            language=response_data['language'],
            confidence=response_data['confidence'],
            sources=response_data['sources'],
            context_count=response_data['context_count']
        )
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
