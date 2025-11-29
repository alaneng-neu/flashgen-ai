import pymupdf
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional
from api.models import FlashcardTypeEnum


class GeneratedFlashcard(BaseModel):
    term: str = Field(..., description="The term, question, or concept to be learned")
    definition: str = Field(..., description="The definition, answer, or explanation")
    flashcard_type: FlashcardTypeEnum = Field(..., description="The type of flashcard")

class FlashcardSet(BaseModel):
    flashcards: List[GeneratedFlashcard] = Field(..., description="A list of generated flashcards")

class RAGQuery(BaseModel):
    query: str = Field(..., description="The search query for the vector store")
    keywords: List[str] = Field(..., description="Keywords to enhance the search")

class TextResponse(BaseModel):
    content: str = Field(..., description="The generated text content")


# Utility functions
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pymupdf"""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_file(content: bytes, content_type: str, filename: str) -> str:
    """Route to appropriate extractor based on file type"""
    if filename.lower().endswith('.pdf') or content_type == 'application/pdf':
        return extract_text_from_pdf(content)
    elif filename.lower().endswith('.txt') or content_type == 'text/plain':
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            return content.decode('latin-1')
    else:
        # Try to decode as text for other types, or return empty string
        try:
            return content.decode('utf-8')
        except:
            return ""


class LLMClient:
    def __init__(self, model: str = "qwen3:8b", host: str = "http://localhost:11434/v1"):
        self.model = model
        self.host = host

        self.client = instructor.from_provider(
            f"ollama/{model}",
            base_url=host,
            mode=instructor.Mode.JSON,
        )

    def generate_flashcards(
        self, 
        file_contents: List[dict], 
        system_prompt: Optional[str] = None, 
        rag_context: Optional[str] = None, 
        num_flashcards: int = 10, 
        flashcard_types: Optional[List[FlashcardTypeEnum]] = None
    ) -> FlashcardSet:
        """
        Generate flashcards from file contents and optional RAG context.
        
        file_contents: List of dicts with 'filename', 'content_type', 'content' (bytes)
        """
        
        # Extract text from all files
        full_text = ""
        for file in file_contents:
            text = extract_text_from_file(file['content'], file.get('content_type', ''), file['filename'])
            if text:
                full_text += f"\n--- Content from {file['filename']} ---\n{text}\n"
        
        if rag_context:
            full_text += f"\n--- Additional Context (RAG) ---\n{rag_context}\n"
            
        if not full_text.strip():
            # If no text could be extracted, return empty set or handle error
            return FlashcardSet(flashcards=[])

        # Construct the prompt
        prompt = f"Generate {num_flashcards} flashcards based on the following content."
        
        if flashcard_types:
            types_str = ", ".join([t.value for t in flashcard_types])
            prompt += f" Please use the following flashcard types: {types_str}."
            
        prompt += f"\n\nContent:\n{full_text[:50000]}" # Truncate if too long
        
        base_system_prompt = "You are an expert educational content creator. Your goal is to create high-quality flashcards that help students learn effectively."
        if system_prompt:
            base_system_prompt += f"\n\nSpecific Instructions:\n{system_prompt}"

        try:
            resp = self.client.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": base_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_model=FlashcardSet,
            )
            return resp
        except Exception as e:
            print(f"Error generating flashcards: {e}")
            return FlashcardSet(flashcards=[])

    def generate_rag_query(self, file_contents: List[dict], system_prompt: Optional[str] = None) -> RAGQuery:
        """Generate search query for vector store based on content summary or topic"""
        
        filenames = ", ".join([f['filename'] for f in file_contents])
        prompt = f"I have the following files: {filenames}. "
        if system_prompt:
            prompt += f"Context: {system_prompt}. "
        prompt += "Generate a search query and keywords to find relevant documents in a vector database."

        try:
            resp = self.client.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates search queries."},
                    {"role": "user", "content": prompt}
                ],
                response_model=RAGQuery,
            )
            return resp
        except Exception as e:
            print(f"Error generating RAG query: {e}")
            return RAGQuery(query="", keywords=[])

    def generate_raw(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Simple text generation without structured output"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            resp = self.client.create(
                model=self.model,
                messages=messages,
                response_model=TextResponse,
            )
            return resp.content
        except Exception as e:
            print(f"Error generating raw text: {e}")
            return ""

    def generate_topic_system_prompt(self, file_contents: List[dict]) -> str:
        """Analyzes uploaded educational materials and returns a system prompt"""
        
        # Extract text (limit to first few pages/files to avoid context overflow)
        full_text = ""
        for file in file_contents[:3]: # Analyze first 3 files
            text = extract_text_from_file(file['content'], file.get('content_type', ''), file['filename'])
            if text:
                full_text += f"\n--- Content from {file['filename']} ---\n{text[:5000]}\n" # First 5000 chars per file
        
        if not full_text.strip():
            return "This topic focuses on general concepts."

        prompt = (
            "Analyze the following educational materials. "
            "Identify the question styles, difficulty level, format patterns, and key concepts. "
            "Generate a system prompt that describes how to create flashcards matching this style. "
            "The output should be a concise paragraph instructing an LLM on how to generate flashcards for this specific topic. "
            "Do NOT include specific content from the files in the prompt, but rather describe the TYPES of questions and concepts to focus on. "
            "For example, instead of saying 'Ask about the mitochondria', say 'Ask about cell organelles and their functions'. "
            "Ensure the prompt is general enough to apply to new material within the same subject area."
        )
        
        messages = [
            {"role": "system", "content": "You are an expert curriculum designer."},
            {"role": "user", "content": prompt + "\n\nMaterials:\n" + full_text}
        ]
        
        try:
            resp = self.client.create(
                model=self.model,
                messages=messages,
                response_model=TextResponse,
            )
            return resp.content
        except Exception as e:
            print(f"Error generating topic system prompt: {e}")
            return "Error generating system prompt."

    async def generate_flashcards_from_files(
        self, 
        file_contents: List[dict], 
        system_prompt: Optional[str] = None,
        num_flashcards: int = 10
    ) -> List[dict]:
        """Async wrapper that returns dicts ready for database insertion"""
        # Since the underlying calls are synchronous (unless we use AsyncOllamaClient), 
        # we might want to run this in a threadpool if it blocks too long.
        # For now, we'll just call it directly.
        
        flashcard_set = self.generate_flashcards(file_contents, system_prompt, num_flashcards=num_flashcards)
        
        return [fc.model_dump() for fc in flashcard_set.flashcards]


# Singleton instance
_client_instance = None

def get_client(model: str = "qwen3:8b", host: str = "http://localhost:11434/v1") -> LLMClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient(model, host)
    return _client_instance
