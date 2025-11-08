from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF
import re
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGIndex:
    def __init__(self, pdf_path):
        logger.info("📘 Loading PDF and creating knowledge base...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = self.extract_text(pdf_path)
        
        if not self.docs:
            raise ValueError("❌ No text could be extracted from the PDF")
            
        logger.info(f"📊 Extracted {len(self.docs)} text chunks")
        
        # Print sample chunks for debugging
        for i, doc in enumerate(self.docs[:3]):
            logger.info(f"Sample chunk {i+1}: {doc[:100]}...")
            
        self.embeddings = self.model.encode(self.docs, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        logger.info(f"✅ Indexed {len(self.docs)} text chunks.")

    def extract_text(self, pdf_path) -> List[str]:
        """Extract text from PDF using multiple methods"""
        text_chunks = []
        
        try:
            with fitz.open(pdf_path) as doc:
                logger.info(f"📄 Processing PDF with {len(doc)} pages")
                
                for page_num, page in enumerate(doc):
                    # Method 1: Regular text extraction
                    text = page.get_text("text").strip()
                    
                    if not text:
                        # Method 2: Try with different parameters
                        text = page.get_text("words")
                        text = " ".join([word[4] for word in text]).strip()
                    
                    if not text:
                        # Method 3: Try raw text extraction
                        text = page.get_text("raw").strip()
                    
                    if text:
                        chunks = self.process_text(text, page_num + 1)
                        text_chunks.extend(chunks)
                    else:
                        logger.warning(f"⚠️ No text found on page {page_num + 1}")
            
            # If still no text, try OCR-like approach (basic)
            if not text_chunks:
                text_chunks = self.extract_text_fallback(pdf_path)
                
            return text_chunks
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from PDF: {str(e)}")
            raise

    def process_text(self, text: str, page_num: int) -> List[str]:
        """Process and chunk extracted text"""
        chunks = []
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace
        text = text.strip()
        
        if not text or len(text) < 10:
            return chunks
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would make chunk too long, save current chunk
            if len(current_chunk) + len(sentence) > 500 and current_chunk:
                chunks.append(f"Page {page_num}: {current_chunk.strip()}")
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if any
        if current_chunk:
            chunks.append(f"Page {page_num}: {current_chunk.strip()}.")
        
        # If no chunks created (very short text), use the whole text
        if not chunks and text:
            chunks.append(f"Page {page_num}: {text}")
            
        return chunks

    def extract_text_fallback(self, pdf_path) -> List[str]:
        """Fallback method for difficult PDFs"""
        chunks = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    # Try to get text blocks
                    blocks = page.get_text("dict")
                    
                    for block in blocks["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text = span["text"].strip()
                                    if text and len(text) > 10:
                                        chunks.append(f"Page {page_num + 1}: {text}")
            
            return chunks
        except Exception as e:
            logger.error(f"❌ Fallback extraction failed: {str(e)}")
            return []

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Return top-k relevant paragraphs from PDF"""
        try:
            if not self.docs:
                return ["No documents available for retrieval."]
                
            query_vec = self.model.encode([query], convert_to_numpy=True)
            
            # Search for more results initially
            k_search = min(k * 2, len(self.docs))
            distances, indices = self.index.search(query_vec, k_search)
            
            # Filter and rank results
            results = []
            seen_content = set()
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.docs):
                    doc_content = self.docs[idx]
                    distance = distances[0][i]
                    
                    # Use a more lenient distance threshold
                    if distance < 1.5 and doc_content not in seen_content:
                        results.append(doc_content)
                        seen_content.add(doc_content)
                    
                    # Stop when we have enough unique results
                    if len(results) >= k:
                        break
            
            logger.info(f"🔍 Query: '{query}' -> Found {len(results)} results")
            
            if not results:
                # Return some context anyway if no good matches
                return self.docs[:min(3, len(self.docs))]
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during retrieval: {str(e)}")
            return ["Error retrieving information from the document."]

    def get_document_info(self) -> dict:
        """Get basic information about the loaded document"""
        return {
            "total_chunks": len(self.docs),
            "sample_chunks": self.docs[:2] if self.docs else []
        }