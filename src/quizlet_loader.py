from typing import Optional
from pathlib import Path
import json
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class QuizletLoader(BaseLoader):
    """
    Load Quizlet flashcard exports in multiple formats.
    
    Supports:
    - Tab-separated text files (classic Quizlet export)
    - JSON files with term/definition objects
    
    JSON format: [{"term": "...", "definition": "..."}, ...]
    Text format: term\tdefinition (one per line)
    """
    
    def __init__(
        self,
        file_path: str,
        file_format: str = "auto",
        delimiter: str = "\t",
        encoding: str = "utf-8",
        combine_cards: bool = False,
        source_name: Optional[str] = None
    ) -> None:
        """
        Initialize the loader.
        
        Args:
            file_path: Path to the Quizlet export file
            file_format: 'auto', 'json', or 'text'. Auto-detects by file extension.
            delimiter: Delimiter for text format (default: tab)
            encoding: File encoding (default: utf-8)
            combine_cards: If True, create one document with all cards
            source_name: Optional name to use in metadata
        """
        self.file_path = Path(file_path)
        self.file_format = file_format
        self.delimiter = delimiter
        self.encoding = encoding
        self.combine_cards = combine_cards
        self.source_name = source_name or self.file_path.name
        
        # Auto-detect format if needed
        if self.file_format == "auto":
            self.file_format = self._detect_format()
    
    def _detect_format(self) -> str:
        """Detect file format based on extension and content."""
        extension = self.file_path.suffix.lower()
        
        if extension == ".json":
            return "json"
        elif extension in [".txt", ".tsv", ".csv"]:
            return "text"
        else:
            # Try to parse as JSON, fall back to text
            try:
                with open(self.file_path, "r", encoding=self.encoding) as f:
                    json.load(f)
                return "json"
            except (json.JSONDecodeError, UnicodeDecodeError):
                return "text"
    
    def lazy_load(self):
        """Lazy load Quizlet flashcards from file."""
        if self.file_format == "json":
            yield from self._load_json()
        else:
            yield from self._load_text()
    
    def _load_json(self):
        """Load flashcards from JSON format."""
        with open(self.file_path, "r", encoding=self.encoding) as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data)}")
        
        if self.combine_cards:
            yield from self._load_json_combined(data)
        else:
            yield from self._load_json_individual(data)
    
    def _load_json_individual(self, data: list):
        """Load each JSON flashcard as a separate document."""
        for idx, card in enumerate(data, start=1):
            if not isinstance(card, dict):
                continue
            
            term = card.get("term", "").strip()
            definition = card.get("definition", "").strip()
            
            if not term and not definition:
                continue
            
            content = f"Term: {term}\nDefinition: {definition}"
            
            yield Document(
                page_content=content,
                metadata={
                    "source": str(self.file_path),
                    "source_name": self.source_name,
                    "card_number": idx,
                    "term": term,
                    "definition": definition,
                    "type": "flashcard",
                    "format": "json"
                }
            )
    
    def _load_json_combined(self, data: list):
        """Load all JSON flashcards into a single document."""
        cards = []
        total_cards = 0
        
        for idx, card in enumerate(data, start=1):
            if not isinstance(card, dict):
                continue
            
            term = card.get("term", "").strip()
            definition = card.get("definition", "").strip()
            
            if not term and not definition:
                continue
            
            cards.append(
                f"Card {idx}:\nTerm: {term}\nDefinition: {definition}"
            )
            total_cards += 1
        
        combined_content = "\n\n".join(cards)
        
        yield Document(
            page_content=combined_content,
            metadata={
                "source": str(self.file_path),
                "source_name": self.source_name,
                "total_cards": total_cards,
                "type": "flashcard_set",
                "format": "json"
            }
        )
    
    def _load_text(self):
        """Load flashcards from text format."""
        if self.combine_cards:
            yield from self._load_text_combined()
        else:
            yield from self._load_text_individual()
    
    def _load_text_individual(self):
        """Load each text flashcard as a separate document."""
        with open(self.file_path, "r", encoding=self.encoding) as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(self.delimiter, maxsplit=1)
                
                if len(parts) == 2:
                    term, definition = parts
                    term = term.strip()
                    definition = definition.strip()
                    
                    content = f"Term: {term}\nDefinition: {definition}"
                    
                    yield Document(
                        page_content=content,
                        metadata={
                            "source": str(self.file_path),
                            "source_name": self.source_name,
                            "card_number": idx,
                            "term": term,
                            "definition": definition,
                            "type": "flashcard",
                            "format": "text"
                        }
                    )
                elif len(parts) == 1:
                    # Handle malformed lines
                    yield Document(
                        page_content=parts[0].strip(),
                        metadata={
                            "source": str(self.file_path),
                            "source_name": self.source_name,
                            "card_number": idx,
                            "type": "flashcard",
                            "format": "text",
                            "warning": "No delimiter found in line"
                        }
                    )
    
    def _load_text_combined(self):
        """Load all text flashcards into a single document."""
        cards = []
        total_cards = 0
        
        with open(self.file_path, "r", encoding=self.encoding) as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(self.delimiter, maxsplit=1)
                
                if len(parts) == 2:
                    term, definition = parts
                    cards.append(
                        f"Card {idx}:\nTerm: {term.strip()}\nDefinition: {definition.strip()}"
                    )
                    total_cards += 1
                elif len(parts) == 1:
                    cards.append(f"Card {idx}: {parts[0].strip()}")
                    total_cards += 1
        
        combined_content = "\n\n".join(cards)
        
        yield Document(
            page_content=combined_content,
            metadata={
                "source": str(self.file_path),
                "source_name": self.source_name,
                "total_cards": total_cards,
                "type": "flashcard_set",
                "format": "text"
            }
        )
