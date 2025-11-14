Install dependencies: pip install -r requirements.txt

Preprocessing Overview

- Preprocesses Quizlet flashcard exports to prepare them for classification and integration as a study tool.
- Preprocessing pipeline implemented in quizlet_loader.py and supports both JSON and text-based flashcard formats

quizlet_loader.py : Loads, parses, and preprocesses Quizlet flashcard exports in different formats, and classifies flashcards by type

- File Format Detection: the input file is JSON or text (tab-separated or CSV)?
- Parsing: extracting the term and definition
- Cleaning: Strips whitespace and handles malformed or incomplete lines
- Classification: Classifies each flashcard into types with either a transformers-based zero-shot classifier or a rule-based fallback.
- Metadata: attached to each flashcard, including its type, source, and position in the dataset
- Document Creation: Wraps each processed flashcard set in standardized Document object

poc.py : proof of concept of LLM connection, example of flashcard classification

quizlet_rag.py : pipeline for Quizlet flashcards, enabling efficient search and retrieval using vector embeddings

- QuizletLoader: Loads and preprocesses Quizlet flashcards from JSON or text exports
- Document Chunking: Chunking strategies (no split, by term/definition, or recursive splitting) to optimize retrieval
- Embedding Creation: Converts flashcards into vector embeddings using Hugging Face models
- Vector Store Management: Creates, loads, and updates a persistent vector database (we use Chroma) for storing and retrieving embedded flashcards
- Querying: similarity search over the flashcard database, can filter by metadata

preprocess_flashcards.ipynb: Automates preprocessing of Quizlet flashcard exports for downstream retrieval and study applications

- Connects to quizlet_loader.py: Handles parsing, cleaning, and metadata enrichment of flashcards
- Connects to quizlet_rag.py: Manages chunking, embedding, and vector database operations

query_notebook.ipynb: Provides an interactive interface to query the embedded Quizlet flashcards

- Connects to the vector database from preprocessing
- search across all flashcards using natural language queries
- can filter results by source file (subject)
- show detailed metadata for each result (including flashcard type)
- batch queries and an query loop for explorations
