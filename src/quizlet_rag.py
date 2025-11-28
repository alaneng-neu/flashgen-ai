from typing import List, Optional
import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.vectorstores.base import VectorStore

from langchain_huggingface import HuggingFaceEmbeddings

from quizlet_loader import QuizletLoader


class QuizletRAGPipeline:
    """
    Complete pipeline for loading Quizlet flashcards, chunking, embedding, 
    and storing in a vector database for RAG applications.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str = "./chroma_db"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Model name for embeddings.
            vector_store_path: Path to store the vector database.
        """
        self.vector_store_path = vector_store_path
        self.embeddings = self._setup_embeddings(embedding_model)
        self.vectorstore = None
        
    def _setup_embeddings(self, model_name: str):
        """Setup embedding model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 32}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    def load_flashcards(
        self,
        file_paths: List[str],
        delimiter: str = "\t",
        chunk_strategy: str = "individual"
    ) -> List[Document]:
        """
        Load flashcards from one or more Quizlet export files.
        
        Args:
            file_paths: List of paths to Quizlet export files
            delimiter: Delimiter used in export (tab or comma)
            chunk_strategy: 'individual' (one card per doc) or 'combined' (all cards)
        
        Returns:
            List of Document objects
        """
        all_docs = []
        
        for file_path in file_paths:
            combine = (chunk_strategy == "combined")
            loader = QuizletLoader(
                file_path=file_path,
                delimiter=delimiter,
                combine_cards=combine
            )
            docs = loader.load()
            all_docs.extend(docs)
            
        print(f"✓ Loaded {len(all_docs)} documents from {len(file_paths)} file(s)")
        return all_docs
    
    def chunk_documents(
        self,
        documents: List[Document],
        strategy: str = "no_split"
    ) -> List[Document]:
        """
        Chunk documents based on strategy.
        
        For flashcards, typically no splitting needed since they're atomic.
        
        Args:
            documents: List of documents to chunk
            strategy: 'no_split', 'by_term', or 'recursive'
        
        Returns:
            List of chunked documents
        """
        if strategy == "no_split":
            # Flashcards are already atomic - no splitting needed
            print(f"✓ Using {len(documents)} flashcards as-is (no splitting)")
            return documents
        
        elif strategy == "by_term":
            # Create separate chunks for term and definition
            # Useful for better retrieval on specific queries
            chunks = []
            for doc in documents:
                if doc.metadata.get("type") == "flashcard":
                    term = doc.metadata.get("term", "")
                    definition = doc.metadata.get("definition", "")
                    
                    # Term chunk
                    chunks.append(Document(
                        page_content=f"Term: {term}",
                        metadata={**doc.metadata, "chunk_type": "term"}
                    ))
                    
                    # Definition chunk
                    chunks.append(Document(
                        page_content=f"Definition: {definition}",
                        metadata={**doc.metadata, "chunk_type": "definition"}
                    ))
                else:
                    chunks.append(doc)
            
            print(f"✓ Split into {len(chunks)} chunks (terms + definitions)")
            return chunks
        
        elif strategy == "recursive":
            # Use recursive splitter for long combined documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(documents)
            print(f"✓ Split into {len(chunks)} recursive chunks")
            return chunks
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def create_vectorstore(
        self,
        documents: List[Document],
        collection_name: str = "quizlet_flashcards"
    ) -> VectorStore:
        """
        Create and populate vector store with documents.
        
        Args:
            documents: List of documents to embed and store
            collection_name: Name for the vector store collection
        
        Returns:
            VectorStore instance
        """
        print(f"⏳ Creating embeddings for {len(documents)} documents...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.vector_store_path
        )
        
        print(f"✓ Vector store created at {self.vector_store_path}")
        return self.vectorstore
    
    def load_existing_vectorstore(
        self,
        collection_name: str = "quizlet_flashcards"
    ) -> VectorStore:
        """Load an existing vector store."""
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.vector_store_path
        )
        print(f"✓ Loaded existing vector store from {self.vector_store_path}")
        return self.vectorstore
    
    def add_flashcards(
        self,
        file_paths: List[str],
        delimiter: str = "\t",
        chunk_strategy: str = "individual"
    ):
        """
        Add new flashcards to existing vector store.
        
        Args:
            file_paths: Paths to new Quizlet export files
            delimiter: Delimiter used in exports
            chunk_strategy: Chunking strategy to use
        """
        docs = self.load_flashcards(file_paths, delimiter, "individual")
        chunks = self.chunk_documents(docs, chunk_strategy)
        
        if self.vectorstore is None:
            raise ValueError("No vector store loaded. Use create_vectorstore() first.")
        
        self.vectorstore.add_documents(chunks)
        print(f"✓ Added {len(chunks)} new chunks to vector store")
    
    def query(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Query the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("No vector store loaded.")
        
        if filter_metadata:
            results = self.vectorstore.similarity_search(
                query, k=k, filter=filter_metadata
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        return results
