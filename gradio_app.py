import sys
sys.modules["pydub"] = __import__("types").SimpleNamespace()
sys.modules["pydub"].AudioSegment = lambda *args, **kwargs: None

import gradio as gr
from main import MultilingualRAGSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG system
rag_system = MultilingualRAGSystem()

def initialize_system():
    """Initialize the system on startup"""
    try:
        rag_system.initialize()
        return "✅ System ready!"
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        return f"❌ Initialization failed: {str(e)}"

def qa_interface(message, history):
    """Enhanced chat interface with better error handling"""
    try:
        if not rag_system.is_initialized:
            return "⚠️ System not initialized. Please wait..."
        
        # Call the synchronous query method
        result = rag_system.query(message)
        
        # Format response with additional info
        response = result['answer']
        if result['confidence'] < 0.3:
            response += f"\n\n⚠️ কম নিশ্চিততা (Low confidence: {result['confidence']:.2f})"
        
        return response
    
    except Exception as e:
        logger.error(f"Error in QA interface: {str(e)}")
        return f"❌ দুঃখিত, একটি ত্রুটি হয়েছে: {str(e)}"

# Create Gradio interface
def create_interface():
    # Initialize system
    init_status = initialize_system()
    
    with gr.Blocks(
        title="🇧🇩 বাংলা RAG QA System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Noto Sans Bengali', 'SolaimanLipi', sans-serif !important;
        }
        """
    ) as interface:
        
        gr.Markdown("# 📚 বাংলা + English RAG QA System")
        gr.Markdown("**HSC Bangla 1st Paper** থেকে প্রশ্নের উত্তর পান")
        
        # Status indicator
        status = gr.Textbox(value=init_status, label="System Status", interactive=False)
        
        # Chat interface
        chatbot = gr.ChatInterface(
            fn=qa_interface,  
            examples=[
                "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
                "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "What is the main theme of this text?"
            ],
            title="Ask your question in Bengali or English"
        )
        
        gr.Markdown("""
        ### 💡 Tips:
        - আপনি বাংলা অথবা ইংরেজিতে প্রশ্ন করতে পারেন
        - প্রশ্ন স্পষ্ট এবং সুনির্দিষ্ট হলে ভাল উত্তর পাবেন
        - You can ask questions in both Bengali and English
        """)
    
    return interface

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )