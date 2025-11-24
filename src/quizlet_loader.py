from typing import Optional, List, Tuple
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

# Use a small LLM client wrapper for zero-shot classification (lazy init)
from llm import LLMClient, get_shared_zero_shot


class QuizletLoader(BaseLoader):
    """
    Load Quizlet flashcard exports in multiple formats.
    
    Supports:
    - Tab-separated text files (classic Quizlet export)
    - JSON files with term/definition objects
    
    JSON format: [{"term": "...", "definition": "..."}, ...]
    Text format: term\tdefinition (one per line)
    """
    
    # Class-level canonical mappings so callers (and experiments) can reuse them
    LABEL_PHRASES = {
        "term_definition": "a term-definition pair",
        "fill_in_blank": "a fill-in-the-blank question",
        "list_stages": "a list of stages or steps",
        "question_to_answer": "a question expecting an answer",
        "example_to_concept": "an example illustrating a concept",
        "multiple_choice": "a multiple-choice question",
        "true_false": "a true or false question",
    }

    DEFAULT_CUE_MAP = {
        "multiple_choice": ["Multiple choice question:"],
        "fill_in_blank": ["Fill-in-the-blank question:"],
        "list_stages": ["A list of stages or steps:"],
        "question_to_answer": ["A question expecting an answer:"],
        "example_to_concept": ["An example illustrating a concept:"],
        "true_false": ["A true or false question:"],
        "term_definition": ["A term-definition pair:"],
    }

    def __init__(
        self,
        file_path: str,
        file_format: str = "auto",
        delimiter: str = "\t",
        encoding: str = "utf-8",
        combine_cards: bool = False,
        source_name: Optional[str] = None,
        # loader will use the HF zero-shot pipeline with 'facebook/bart-large-mni' by default
        classify_model: Optional[str] = None,
        # optional injected LLM client for dependency injection / testing
        zero_shot_client: Optional[LLMClient] = None,
        # threshold for accepting the transformer's top prediction when rule is not strong
        transformer_confidence_threshold: float = 0.5,
        # when to use cues: 'strong_only' (default) uses cues only for strong rule matches,
        # 'always' will always include a cue when available, 'never' disables cueing
        use_cues_when: str = "strong_only",
        # hypothesis template for HF zero-shot (if used)
        hypothesis_template: str = "This is {}.",
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
        self.classify_model = classify_model or "facebook/bart-large-mnli"
        self.transformer_confidence_threshold = float(transformer_confidence_threshold)
        # optionally injected or shared zero-shot client (lazy init occurs inside client)
        self.zero_shot_client = zero_shot_client
        if self.zero_shot_client is None:
            try:
                self.zero_shot_client = get_shared_zero_shot(self.classify_model)
                # note: actual model weights are loaded lazily by LLMClient.classify
                print("zero-shot client configured (pipeline will init lazily)")
            except Exception:
                self.zero_shot_client = None
                print("use fallback rule-based classifier")

        # cue config (use class-default unless caller overrides attribute later)
        self.cue_map = self.DEFAULT_CUE_MAP.copy()
        self.use_cues_when = use_cues_when
        self.hypothesis_template = hypothesis_template
        
        # Auto-detect format if needed
        if self.file_format == "auto":
            self.file_format = self._detect_format()

        # Initialize HF zero-shot pipeline if requested and available
    
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
            # Classify the flashcard into a flashcard_type
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
                    # Classify the flashcard into a flashcard_type
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
                    # Try to classify even malformed lines
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
        
    def rule_label_and_confidence(self, term: str, definition: str) -> Tuple[Optional[str], bool]:
        """Rule-based detector returning (label, strong_confidence).
        Strong indicates a high-confidence rule match where cues should be used.
        Uses simple heuristics (underscores/blanks, enumerated options, semicolons,
        question forms, example markers, and true/false indicators).
        """
        txt = f"{term} {definition}".lower()

        # Strong signals are run first
        if "___" in term or "fill in the" in txt or "fill-in" in txt:
            return "fill_in_blank", True
        if any(tok in txt for tok in ["a)", "b)", "c)", "(a)", "(b)", "(c)", "1.", "2.", "3.", "a.", "b.", "c."]) and ("?" not in definition):
            return "multiple_choice", True
        if any(word in txt for word in ["stages", "phases", "list"]):
            return "list_stages", True
        if "true or false" in txt or "true/false" in txt or ("true" in txt and "false" in txt):
            return "true_false", True
        if "example" in txt or "instance" in txt or "e.g." in txt:
            return "example_to_concept", True
        if any(word in txt for word in ["what", "which", "how", "why", "when", "where", "?"]):
            return "question_to_answer", True
        if txt.count(";") >= 2 or any(word in txt for word in ["first", "then", "next", "step"]):
            return "list_stages", False

        # Default when both term and definition exist
        if term.strip() and definition.strip():
            return "term_definition", False

        # Nothing matched
        return None, False

    def _classify_card(self, term: str, definition: str) -> Optional[str]:
        # Classify a flashcard into one of the known types, fall back to rule label.
        # Use rule detector to create optional cue, then call the LLM client (if available),
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

        rule_label, strong = self.rule_label_and_confidence(term, definition)
        # ensure we always assign a type, default to term_definition when unknown
        if rule_label is None:
            rule_label = "term_definition"
            strong = False

        # If no LLM client, just return the rule_label
        if self.zero_shot_client is None:
            return rule_label

        # prepare candidate phrases (natural language) mapping to short labels
        label_phrases = self.LABEL_PHRASES
        candidate_phrases = list(label_phrases.values())

        # decide whether to include a cue (always prepend)
        cue = ""
        if self.cue_map and rule_label in self.cue_map:
            if self.use_cues_when == "always":
                cue_val = self.cue_map[rule_label]
            elif self.use_cues_when == "strong_only" and strong:
                cue_val = self.cue_map[rule_label]
            else:
                cue_val = None

            if cue_val:
                cue = cue_val[0] if isinstance(cue_val, (list, tuple)) else cue_val

        input_text = (cue + " " + text).strip()

        try:
            result = self.zero_shot_client.classify(input_text, candidate_labels=candidate_phrases, hypothesis_template=self.hypothesis_template)
            if isinstance(result, dict) and result.get("labels"):
                labels_out = result.get("labels", [])
                scores_out = result.get("scores", [])
                top_phrase = labels_out[0] if labels_out else None
                top_score = scores_out[0] if scores_out else None
                inverse_map = {v: k for k, v in label_phrases.items()}
                short = inverse_map.get(top_phrase)
                if short:
                    # prefer rule when rule detector is strong
                    if strong:
                        return rule_label
                    # otherwise accept transformer's prediction only if above threshold
                    if top_score is not None and top_score >= self.transformer_confidence_threshold:
                        return short
                    return rule_label
        except Exception:
            # fall back!
            return rule_label

        # if transformer didn't return a mapped short label, return the rule label
        return rule_label
