"""DocumentStore.py

Document store class for managing PDF embeddings and retrieval.

Author: Cline
version 0.1.0
"""
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class DocumentStore:
    """Manages PDF loading, embedding, and retrieval."""

    def __init__(self, persist_directory: str = "db"):
        """Initialize the document store.

        Args:
            persist_directory (str): Directory to persist the vector store.
        """
        self.__persist_directory = persist_directory
        self.__embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.__vectorstore = Chroma(
            collection_name="koios_collection",
            persist_directory=self.__persist_directory,
            embedding_function=self.__embeddings
        )

    def add_pdf(self, file_path: str) -> None:
        """Load a PDF, split it into chunks, and add to the vector store.

        Args:
            file_path (str): Path to the PDF file.
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        self.__vectorstore.add_documents(splits)

    def search(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant documents.

        Args:
            query (str): The search query.
            k (int): Number of documents to retrieve.

        Returns:
            List[Document]: List of relevant documents.
        """
        return self.__vectorstore.similarity_search(query, k=k)

    def get_all_documents(self) -> List[str]:
        """Get a list of unique source filenames in the store.

        Returns:
            List[str]: List of filenames.
        """
        results = self.__vectorstore.get()
        metadatas = results.get("metadatas", [])
        sources = set()
        for meta in metadatas:
            if meta and "source" in meta:
                sources.add(os.path.basename(meta["source"]))
        return list(sources)

    def clear_all_documents(self) -> None:
        """Remove all documents from the vector store."""
        results = self.__vectorstore.get()
        ids = results.get("ids", [])
        if ids:
            self.__vectorstore.delete(ids=ids)
