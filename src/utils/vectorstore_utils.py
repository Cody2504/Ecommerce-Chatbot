from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from src.utils.document_processor import create_optimized_vectorstore, process_documents

def load_documents(data_path):
    """Loads documents from the specified path."""
    loader = DirectoryLoader(data_path, glob='*.txt')
    docs = loader.load()
    return docs

def create_vectorstore(docs, embeddings, persist_directory):
    """Creates and persists a vector store with optimized document chunking."""
    return create_optimized_vectorstore(docs, embeddings, persist_directory)