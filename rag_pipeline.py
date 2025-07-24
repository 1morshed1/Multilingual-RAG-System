import logging
from typing import List, Dict, Optional, Any
from config import Config
from langdetect import detect
import json
import requests # Import the requests library

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.config = Config()
        self.chat_history = []

    def detect_language(self, text: str) -> str:
        """Detect query language"""
        try:
            lang = detect(text)
            return 'bn' if lang == 'bn' else 'en'
        except:
            # Fallback: check for Bengali Unicode characters
            bengali_chars = len([c for c in text if '\u0980' <= c <= '\u09FF'])
            return 'bn' if bengali_chars > 0 else 'en'

    # Changed from async def to def
    def generate_answer(self, query: str, contexts: List[Dict],
                              conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Enhanced answer generation with better context handling using Gemini API"""

        # Detect language
        query_language = self.detect_language(query)

        # Prepare context
        context_text = self._prepare_context(contexts)

        # Create system prompt based on language
        system_prompt = self._create_system_prompt(query_language)

        # Prepare conversation context for Gemini
        messages = [{"role": "user", "parts": [{"text": system_prompt}]}]

        # Add conversation history if provided
        if conversation_history:
            for turn in conversation_history[-self.config.MAX_CHAT_HISTORY:]:
                if turn["role"] == "user":
                    messages.append({"role": "user", "parts": [{"text": turn["content"]}]})
                elif turn["role"] == "assistant":
                    messages.append({"role": "model", "parts": [{"text": turn["content"]}]})

        # Create user prompt
        user_prompt = self._create_user_prompt(query, context_text, query_language)
        messages.append({"role": "user", "parts": [{"text": user_prompt}]})

        # Gemini API payload
        payload = {
            "contents": messages,
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500
            }
        }

        # API Key and URL for Gemini
        apiKey = self.config.GEMINI_API_KEY
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

        try:
            # Use requests.post for synchronous HTTP request
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # --- DEBUGGING: Log raw response text ---
            response_text = response.text
            logger.error(f"Raw API response text: {response_text}")
            # --- END DEBUGGING ---

            result = response.json() # Parse JSON directly

            answer = ""
            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and \
               result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                logger.warning(f"Unexpected Gemini API response structure: {result}")
                answer = 'দুঃখিত, উত্তর তৈরি করা যায়নি। Sorry, could not generate an answer.'

            return {
                'answer': answer,
                'language': query_language,
                'sources': [ctx['metadata'] for ctx in contexts],
                'confidence': self._estimate_confidence(answer, contexts)
            }

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during Gemini API call: {e}. Response: {response_text}")
            return {
                'answer': f'দুঃখিত, API থেকে একটি HTTP ত্রুটি হয়েছে: {e.response.status_code}। Sorry, an HTTP error occurred from API: {e.response.status_code}.',
                'language': query_language,
                'sources': [],
                'confidence': 0.0
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}. Response was not valid JSON. Raw response: {response_text}")
            return {
                'answer': 'দুঃখিত, API থেকে অপ্রত্যাশিত প্রতিক্রিয়া। Sorry, unexpected response from API.',
                'language': query_language,
                'sources': [],
                'confidence': 0.0
            }
        except Exception as e:
            logger.error(f"Error generating answer with Gemini API: {str(e)}")
            return {
                'answer': 'দুঃখিত, একটি ত্রুটি ঘটেছে। Sorry, an error occurred.',
                'language': query_language,
                'sources': [],
                'confidence': 0.0
            }

    def _prepare_context(self, contexts: List[Dict]) -> str:
        """Prepare context text with source attribution"""
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            page_info = f"[পৃষ্ঠা {ctx['metadata']['page_number']}]"
            context_parts.append(f"{page_info} {ctx['text']}")
        return "\n\n".join(context_parts)

    def _create_system_prompt(self, language: str) -> str:
        """Create language-appropriate system prompt"""
        if language == 'bn':
            return """আপনি একটি বাংলা সাহিত্য বিশেষজ্ঞ। দেওয়া প্রসঙ্গ থেকে প্রশ্নের উত্তর দিন। উত্তর সংক্ষিপ্ত এবং সরাসরি হতে হবে। যদি উত্তর প্রসঙ্গে না থাকে, তাহলে "এই তথ্য পাওয়া যায়নি" বলুন।"""
        else:
            return """You are a Bengali literature expert. Answer questions based on the provided context. Keep answers concise and direct. If the answer is not in the context, say "Information not found"."""

    def _create_user_prompt(self, query: str, context: str, language: str) -> str:
        """Create user prompt with context"""
        if language == 'bn':
            return f"""প্রসঙ্গ:{context}প্রশ্ন: {query}উত্তর:"""
        else:
            return f"""Context:{context}Question: {query}Answer:"""

    def _estimate_confidence(self, answer: str, contexts: List[Dict]) -> float:
        """Simple confidence estimation"""
        if not answer or "তথ্য পাওয়া যায়নি" in answer or "not found" in answer.lower():
            return 0.0

        # Check if answer contains specific information from context
        answer_words = set(answer.lower().split())
        context_words = set()
        for ctx in contexts:
            context_words.update(ctx['text'].lower().split())

        overlap = len(answer_words.intersection(context_words))
        total_answer_words = len(answer_words)

        return min(overlap / max(total_answer_words, 1), 1.0)

    def add_to_history(self, query: str, answer: str):
        """Add interaction to conversation history"""
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

        # Maintain history limit
        if len(self.chat_history) > self.config.MAX_CHAT_HISTORY:
            self.chat_history = self.chat_history[-self.config.MAX_CHAT_HISTORY:]