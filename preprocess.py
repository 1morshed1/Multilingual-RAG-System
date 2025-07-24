import re
import unicodedata
import pdfplumber
import logging
from typing import List, Dict, Optional, Iterator, Union
from config import Config # Assuming Config class exists and has HEADER_HEIGHT_PERCENTAGE, FOOTER_HEIGHT_PERCENTAGE, CHUNK_SIZE, MIN_CHUNK_SIZE, OVERLAP_SENTENCE_COUNT, MAX_SENTENCE_LEN_FOR_SPLIT

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Enhanced Bengali text preprocessor with intelligent chunking and OCR correction."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._setup_patterns()

    def _setup_patterns(self):
        """Initialize regex patterns and OCR fixes for efficient reuse."""
        # Common Bengali OCR correction patterns. Expand this based on observed errors.
        self.ocr_fixes = {
            'ক্স': 'ক্ষ',   # Common ligature error (kṣa)
            'ত্র': 'ত্র',    # Example: if 'ত্র' is misread as something else, fix it. (Currently no-op)
            'ৃ': 'ৃ',      # Example: if 'ৃ' is misread. (Currently no-op)
            'ব্': 'ব',      # Example: if 'ব্' (ba with halant) is misread.
            '্য': '্য',     # Example: if '্য' (ya-phala) is misread.
            'ৎ': 'ত্',     # Khanda-ta correction (often misread)
            '৷': '।',      # Bengali section mark sometimes misread as danda
            # Add more as needed, e.g., similar-looking characters:
            # 'ণ': 'ন', 'র': 'ড়', 'ড': 'ড়', 'স': 'শ', 'ষ': 'শ' (context-dependent)
        }

        # Compiled regex patterns for cleaning
        # Keeps Bengali characters, ASCII alphanumeric, and specific punctuation (danda, period, exclamation, question, en/em dash, single/double quotes)
        self.clean_pattern = re.compile(r'[^\u0980-\u09FF\u0020-\u007E\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\u2013-\u2015।\'"]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.hyphen_pattern = re.compile(r'-\s*\n\s*')
        self.punctuation_pattern = re.compile(r'[।\.!?]+') # For splitting sentences

        # Chapter and dialogue patterns
        self.chapter_pattern = re.compile(r'(অধ্যায়|পরিচ্ছেদ)\s*[\d\u09e6-\u09ef]+|\b[\d\u09e6-\u09ef]+\.\s*[ক-হ]\b')
        self.dialogue_pattern = re.compile(r'\b(?:বলল|বলে|বললেন|কহিল|বলিল|জিজ্ঞেস করল|প্রশ্ন করল|উত্তরে বলল|উচ্চারণ করল|চিৎকার করল|কঁাদল)\b|["\'—](?:.*?)[”\']')
        
        # Known entities for HSC Bangla 1st Paper (expand as needed)
        self.known_entities = {
            'অনুপম', 'কল্যাণী', 'শম্ভুনাথ', 'মামা', 'দীপু', 'রবি',
            'হরি', 'গোপাল', 'শ্যাম', 'রাম', 'সীতা', 'গীতা', 'বিনু', 'শরৎচন্দ্র'
        }

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extracts text from PDF pages, removing headers/footers based on vertical position,
        and preserving page-level metadata including tables.
        """
        pages_data = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Reconstruct text from filtered words, preserving some layout with newlines
                    # This method handles header/footer filtering internally
                    text = self._reconstruct_text_from_words(page)

                    tables = self._extract_tables(page)
                    
                    if text and len(text.strip()) > self.config.MIN_CHUNK_SIZE // 2: # Min content for a page to be considered
                        page_data = {
                            'text': text,
                            'tables': tables, # Raw table data
                            'metadata': { # Page-level metadata
                                'page_number': page_num, # MOVED HERE: Ensure page_number is nested
                                'bbox': page.bbox,
                                'word_count': len(text.split()),
                                'char_count': len(text),
                                'has_bengali': self._has_bengali_text(text),
                                'has_tables_on_page': bool(tables) # Indicates if tables were found on this page
                            }
                        }
                        pages_data.append(page_data)

            logger.info(f"Successfully extracted {len(pages_data)} pages from PDF with header/footer removal.")
            return pages_data

        except Exception as e:
            logger.error(f"Error extracting PDF '{pdf_path}': {str(e)}")
            raise

    def _reconstruct_text_from_words(self, page) -> str:
        """
        Extracts words from a PDF page, filters out header/footer regions,
        and reconstructs the text attempting to preserve line breaks.
        """
        page_height = page.height
        header_cutoff = page_height * self.config.HEADER_HEIGHT_PERCENTAGE
        footer_cutoff = page_height * (1 - self.config.FOOTER_HEIGHT_PERCENTAGE)

        content_words = [
            word for word in page.extract_words()
            if word['top'] > header_cutoff and word['bottom'] < footer_cutoff
        ]

        if not content_words:
            return ""

        # Sort words by their y-coordinate (top) and then x-coordinate (left)
        content_words.sort(key=lambda w: (w['top'], w['x0']))

        reconstructed_text = []
        current_line_top = -1
        line_buffer = []
        previous_word = None # Keep track of the previous word
        
        # Estimate average line height for better line break detection
        line_heights = [word['bottom'] - word['top'] for word in content_words]
        avg_line_height = sum(line_heights) / len(line_heights) if line_heights else 10
        line_break_threshold = avg_line_height * 0.7 # A slightly higher threshold for new lines

        for word in content_words:
            # Safely get word dimensions, defaulting to a small value if missing
            word_x0 = word.get('x0', 0)
            word_x1 = word.get('x1', word_x0 + 1) # Default x1 if missing
            word_width = word.get('width', 10) # Default width if missing

            if current_line_top == -1: # First word in content_words
                current_line_top = word['top']
                line_buffer.append(word['text'])
            elif word['top'] - current_line_top > line_break_threshold: # New line detected
                reconstructed_text.append(" ".join(line_buffer))
                line_buffer = [word['text']]
                current_line_top = word['top']
            else: # Same line, check for space
                # Add space if there's a significant horizontal gap from the previous word
                if previous_word:
                    prev_word_x1 = previous_word.get('x1', previous_word.get('x0', 0) + 1)
                    if word_x0 - prev_word_x1 > word_width * 0.5: # Use current word's width for gap comparison
                        line_buffer.append(" ") # Add an explicit space for larger gaps
                line_buffer.append(word['text'])
            previous_word = word # Update previous_word for the next iteration
        
        if line_buffer: # Add the last line
            reconstructed_text.append(" ".join(line_buffer))

        return "\n".join(reconstructed_text)


    def _extract_tables(self, page) -> List[List[List[str]]]:
        """Extract table data if present."""
        try:
            # pdfplumber.extract_tables() returns a list of tables, each table is a list of rows,
            # and each row is a list of cell strings.
            return page.extract_tables() or []
        except Exception as e:
            logger.warning(f"Could not extract tables from page {page.page_number}: {e}")
            return []

    def _has_bengali_text(self, text: str) -> bool:
        """Check if text contains any Bengali characters."""
        return bool(re.search(r'[\u0980-\u09FF]', text))

    def clean_bengali_text(self, text: str) -> str:
        """
        Advanced Bengali text cleaning with proper Unicode handling,
        common OCR error fixes, and whitespace normalization.
        """
        if not text:
            return ""

        # 1. Normalize Unicode to NFC (Canonical Composition)
        text = unicodedata.normalize('NFC', text)

        # 2. Apply OCR error fixes
        for wrong, correct in self.ocr_fixes.items():
            text = text.replace(wrong, correct)

        # 3. De-hyphenate line breaks (e.g., "দেশ-" + "\n" + "প্রেম" -> "দেশপ্রেম")
        text = self.hyphen_pattern.sub('', text)
        
        # 4. Replace multiple newlines/carriage returns with a single space
        text = re.sub(r'[\n\r]+', ' ', text)

        # 5. Normalize all whitespace (tabs, multiple spaces) to single spaces
        text = self.whitespace_pattern.sub(' ', text)

        # 6. Remove unwanted characters based on the pre-compiled pattern
        text = self.clean_pattern.sub(' ', text)

        # 7. Consolidate excessive punctuation (e.g., "।।।" -> "।")
        text = self.punctuation_pattern.sub('।', text) # Use Bengali danda as standard

        return text.strip()

    def smart_chunk_text(self, pages_data: List[Dict]) -> List[Dict]:
        """
        Intelligent chunking that preserves semantic boundaries by prioritizing
        sentence and paragraph breaks, with configurable overlap.
        """
        chunks = []
        chunk_id = 0

        for page_data in pages_data:
            cleaned_text = self.clean_bengali_text(page_data['text'])
            if not cleaned_text or len(cleaned_text) < self.config.MIN_CHUNK_SIZE:
                continue

            # Iterate through semantic chunks (sentences with overlap)
            for chunk_text in self._create_semantic_chunks(cleaned_text):
                chunk_data = self._create_chunk_metadata(
                    chunk_id, chunk_text, page_data
                )
                chunks.append(chunk_data)
                chunk_id += 1
        
        # Post-processing to merge small trailing chunks (re-added for robustness)
        final_chunks = []
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            # If current chunk is too small and there's a next chunk to merge with
            if len(current_chunk['text']) < self.config.MIN_CHUNK_SIZE and i + 1 < len(chunks):
                next_chunk = chunks[i+1]
                merged_text = current_chunk['text'] + " " + next_chunk['text']
                
                # Only merge if the combined chunk doesn't become excessively large
                if len(merged_text) <= self.config.CHUNK_SIZE * 1.5: # Allow up to 150% of target chunk size
                    # Create a new metadata dict for the merged chunk
                    merged_metadata = current_chunk['metadata'].copy()
                    next_metadata = next_chunk['metadata'].copy()

                    # Merge specific metadata fields
                    merged_metadata['named_entities'] = list(set(
                        merged_metadata.get('named_entities', []) + next_metadata.get('named_entities', [])
                    ))
                    merged_metadata['contains_dialogue'] = (
                        merged_metadata.get('contains_dialogue', False) or 
                        next_metadata.get('contains_dialogue', False)
                    )
                    merged_metadata['has_tables_on_page'] = (
                        merged_metadata.get('has_tables_on_page', False) or 
                        next_metadata.get('has_tables_on_page', False)
                    )
                    # Update character and word counts for the merged chunk
                    merged_metadata['char_count'] = len(merged_text)
                    merged_metadata['word_count'] = len(merged_text.split())
                    
                    # For page_number, if merged across pages, you might want a range or the first page.
                    # For simplicity, we'll keep the page number of the first chunk.
                    # merged_metadata['page_number'] = current_chunk['metadata']['page_number'] 
                    # (This is already in merged_metadata from the copy)

                    merged_chunk_data = {
                        'id': current_chunk['id'], # Keep the ID of the first chunk
                        'text': merged_text,
                        'metadata': merged_metadata
                    }
                    final_chunks.append(merged_chunk_data)
                    i += 2 # Skip the next chunk as it's merged
                else:
                    final_chunks.append(current_chunk)
                    i += 1
            else:
                final_chunks.append(current_chunk)
                i += 1

        logger.info(f"Generated {len(final_chunks)} optimized chunks (after merging small ones).")
        return final_chunks

    def _create_semantic_chunks(self, text: str) -> Iterator[str]:
        """
        Creates chunks that respect semantic boundaries (sentences) with overlap.
        Falls back to word-based windowing if sentence splitting is ineffective.
        """
        sentences = self._split_bengali_sentences(text)

        if not sentences:
            # Fallback to word-based windowing if no sentences are found
            logger.warning("No sentences found after splitting. Falling back to word-based chunking.")
            yield from self._create_word_windows(text)
            return

        current_chunk_sentences = []
        current_chunk_char_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # Check if adding this sentence exceeds chunk size
            # and if the current chunk is not empty and already reasonably sized
            if (current_chunk_char_count + sentence_len > self.config.CHUNK_SIZE and
                current_chunk_char_count > self.config.MIN_CHUNK_SIZE): # Ensure current chunk is not too small before breaking
                
                # Yield current chunk
                chunk_text = " ".join(current_chunk_sentences)
                if len(chunk_text) >= self.config.MIN_CHUNK_SIZE:
                    yield chunk_text
                
                # Start new chunk with overlap
                overlap_size = min(len(current_chunk_sentences), self.config.OVERLAP_SENTENCE_COUNT)
                overlap_sentences = current_chunk_sentences[-overlap_size:] if overlap_size > 0 else []
                
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_char_count = len(" ".join(current_chunk_sentences))

            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_char_count += sentence_len + (1 if current_chunk_char_count > 0 else 0) # Add 1 for space if not first sentence

        # Yield final chunk if it meets minimum size
        if current_chunk_sentences:
            final_chunk_text = " ".join(current_chunk_sentences)
            if len(final_chunk_text) >= self.config.MIN_CHUNK_SIZE:
                yield final_chunk_text

    def _create_word_windows(self, text: str) -> Iterator[str]:
        """
        Creates overlapping word-based windows as a fallback chunking strategy.
        Approximates word count based on character chunk size.
        """
        words = text.split()
        # Approximate words per chunk based on average word length (e.g., 5 chars/word)
        chunk_size_words = max(1, self.config.CHUNK_SIZE // 5)
        overlap_words = max(0, self.config.OVERLAP // 5) # Use OVERLAP from config for word-based

        step = max(1, chunk_size_words - overlap_words)

        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size_words]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text) >= self.config.MIN_CHUNK_SIZE:
                yield chunk_text

    def _split_bengali_sentences(self, text: str) -> List[str]:
        """
        Smart sentence splitting for Bengali text, using common punctuation
        and heuristic for very long sentences.
        """
        # Primary split on Bengali danda (।), English periods (.), exclamation marks (!), question marks (?)
        # The regex `(?<=[।\.!?])\s*` splits *after* the punctuation and any whitespace.
        # `|\n\n+` also splits at paragraph breaks (two or more newlines).
        sentences = re.split(r'(?<=[।\.!?])\s*|\n\n+', text)
        
        refined_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Heuristic: If a sentence is extremely long, try to split at common Bengali conjunctions
            # or discourse markers to prevent very large "sentences" that are actually multiple thoughts.
            if len(sentence) > self.config.MAX_SENTENCE_LEN_FOR_SPLIT:
                # Split at common conjunctions/connectors if they are followed by a space
                # and don't seem to be part of an abbreviation or number.
                # Adding word boundaries \b to conjunctions to prevent splitting within words.
                sub_parts = re.split(r'(?<!\d)(?:[,;:]|\bএবং\b|\bকিন্তু\b|\bতবে\b|\bঅথচ\b|\bযদিও\b|\bকারণ\b|\bফলে\b|\bএজন্য\b|\bতাছাড়া\b)\s*', sentence)
                sub_parts = [s.strip() for s in sub_parts if s.strip()]
                if len(sub_parts) > 1: # Only extend if actual splits occurred
                    refined_sentences.extend(sub_parts)
                else:
                    refined_sentences.append(sentence) # No effective split, keep as is
            else:
                refined_sentences.append(sentence)

        return [s.strip() for s in refined_sentences if s.strip()]

    def _create_chunk_metadata(self, chunk_id: int, text: str, page_data: Dict) -> Dict:
        """
        Create chunk with rich metadata, ensuring all metadata fields are nested
        under the 'metadata' key for compatibility with vector databases.
        """
        # Start with a copy of page-level metadata to inherit page_number, bbox, etc.
        chunk_metadata = page_data['metadata'].copy()
        
        # Add/update chunk-specific metadata
        chunk_metadata.update({
            'page_number': page_data['metadata']['page_number'], # CORRECTED: Access page_number from page_data['metadata']
            'sentence_count': len(self._split_bengali_sentences(text)), # Recalculate for the chunk text
            'char_count': len(text),
            'word_count': len(text.split()),
            'chapter': self._extract_chapter_info(text),
            'contains_dialogue': self._contains_dialogue(text),
            'named_entities': self._extract_named_entities(text),
            'bengali_percentage': self._calculate_bengali_percentage(text),
            'complexity_score': self._calculate_complexity_score(text)
        })

        return {
            'id': str(chunk_id),
            'text': text.strip(),
            'metadata': chunk_metadata
        }

    def _extract_chapter_info(self, text: str) -> str:
        """Extract chapter or section information using pre-compiled regex."""
        match = self.chapter_pattern.search(text)
        return match.group().strip() if match else "unknown"

    def _contains_dialogue(self, text: str) -> bool:
        """Check if text contains dialogue markers using pre-compiled regex."""
        return bool(self.dialogue_pattern.search(text))

    def _extract_named_entities(self, text: str) -> List[str]:
        """Extract known named entities from text based on a predefined list."""
        return [entity for entity in self.known_entities if entity in text]

    def _calculate_bengali_percentage(self, text: str) -> float:
        """Calculate percentage of Bengali characters in text (excluding whitespace)."""
        if not text:
            return 0.0
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        total_non_whitespace_chars = len(re.sub(r'\s', '', text))
        return (bengali_chars / total_non_whitespace_chars * 100) if total_non_whitespace_chars > 0 else 0.0

    def _calculate_complexity_score(self, text: str) -> float:
        """
        Calculate text complexity based on sentence length and vocabulary richness.
        Score is normalized between 0 and 1.
        """
        if not text:
            return 0.0

        sentences = self._split_bengali_sentences(text)
        if not sentences:
            return 0.0

        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        words = text.split()
        unique_words = len(set(words))
        total_words = len(words)

        vocabulary_richness = unique_words / total_words if total_words > 0 else 0.0
        
        # Normalize avg_sentence_length (e.g., max expected 25 words per sentence)
        normalized_avg_sentence_length = min(avg_sentence_length / 25.0, 1.0)

        # Combine factors for a simple complexity score
        # You might want to tune weights or use more sophisticated metrics
        complexity_score = (vocabulary_richness * 0.5) + (normalized_avg_sentence_length * 0.5)
        return complexity_score

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Complete PDF processing pipeline: extraction, cleaning, and intelligent chunking."""
        logger.info(f"Starting PDF processing: {pdf_path}")

        # Extract pages with initial cleaning and header/footer removal
        pages_data = self.extract_text_from_pdf(pdf_path)

        # Create semantic chunks with metadata
        chunks = self.smart_chunk_text(pages_data)

        logger.info(f"Processing complete: {len(chunks)} chunks created for {pdf_path}")
        return chunks
