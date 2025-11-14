from typing import Optional, List
from pathlib import Path
import json
try:
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import Document
except Exception:
    # dont need langchain installed to use the loader in isolation
    class BaseLoader:
        pass

    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict

# Try transformers-based zero-shot classifier, or fall back to a lightweight rule-based classifier
# This part optional so the loader works without extra dependencies
try:
    from transformers import pipeline
except Exception:
    pipeline = None


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
        source_name: Optional[str] = None,
        # default False to avoid requiring dependencies
        classify_cards: bool = False,
        # loader will use the HF zero-shot pipeline with 'facebook/bart-large-mnli' by default
        classify_model: Optional[str] = None,
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
        self.classify_cards = classify_cards
        self.classify_model = classify_model or "facebook/bart-large-mnli"
        
        # Auto-detect format if needed
        if self.file_format == "auto":
            self.file_format = self._detect_format()

        # Initialize HF zero-shot pipeline if requested and available
        self._zs_pipeline = None
        if self.classify_cards and pipeline is not None:
            try:
                self._zs_pipeline = pipeline("zero-shot-classification", model=self.classify_model)
            except Exception:
                # If pipeline initialization fails, fallback to rule-based
                self._zs_pipeline = None
    
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
            # Optionally classify the flashcard into a flashcard_type
            flashcard_type = None
            if self.classify_cards:
                flashcard_type = self._classify_card(term, definition)

            metadata = {
                "source": str(self.file_path),
                "source_name": self.source_name,
                "card_number": idx,
                "term": term,
                "definition": definition,
                "type": "flashcard",
                "format": "json",
            }
            if flashcard_type:
                metadata["flashcard_type"] = flashcard_type

            yield Document(page_content=content, metadata=metadata)
    
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
                    # Optionally classify the flashcard into a flashcard_type
                    flashcard_type = None
                    if self.classify_cards:
                        flashcard_type = self._classify_card(term, definition)

                    metadata = {
                        "source": str(self.file_path),
                        "source_name": self.source_name,
                        "card_number": idx,
                        "term": term,
                        "definition": definition,
                        "type": "flashcard",
                        "format": "text",
                    }
                    if flashcard_type:
                        metadata["flashcard_type"] = flashcard_type

                    yield Document(page_content=content, metadata=metadata)
                elif len(parts) == 1:
                    # Handle malformed lines
                    metadata = {
                        "source": str(self.file_path),
                        "source_name": self.source_name,
                        "card_number": idx,
                        "type": "flashcard",
                        "format": "text",
                        "warning": "No delimiter found in line",
                    }
                    # Try to classify even malformed lines if requested
                    if self.classify_cards:
                        flashcard_type = self._classify_card(parts[0].strip(), "")
                        if flashcard_type:
                            metadata["flashcard_type"] = flashcard_type

                    yield Document(page_content=parts[0].strip(), metadata=metadata)
    
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

    def _classify_card(self, term: str, definition: str) -> Optional[str]:
        """Classify a flashcard into one of the known types.

        Returns a string label (e.g. 'term_definition', 'fill_in_blank', ...)
        or None if classification is not possible.
        """
        text = f"Term: {term}\nDefinition: {definition}".strip()

        labels = [
            "term_definition",
            "fill_in_blank",
            "list_stages",
            "question_to_answer",
            "example_to_concept",
            "multiple_choice",
            "true_false",
        ]

        # uses transformers zero-shot pipeline if available
        if self._zs_pipeline is not None:
            try:
                result = self._zs_pipeline(text, candidate_labels=labels)
                # result can be a dict with 'labels' list; pick top label
                if isinstance(result, dict) and result.get("labels"):
                    return result["labels"][0]
            except Exception:
                # go to rule-based
                pass

        # rule-based fallback
        lower_text = text.lower()
        if "___" in lower_text or "fill in the" in lower_text or "fill-in" in lower_text:
            return "fill_in_blank"
        if lower_text.count(";") >= 2 or "first," in lower_text and "then" in lower_text:
            return "list_stages"
        if "a)" in lower_text or "b)" in lower_text or "a." in lower_text  or "b." in lower_text or "multiple choice" in lower_text:
            return "multiple_choice"
        if lower_text.strip().endswith("?") or "what is" in lower_text or "which" in lower_text:
            return "question_to_answer"
        if "for example" in lower_text or "e.g." in lower_text:
            return "example_to_concept"
        if "true" in lower_text or "false" in lower_text or "true/false" in lower_text:
            return "true_false"

        # Default
        return "term_definition"
