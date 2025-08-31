# app/rag/vectorstore.py
import os
import faiss
import pickle
from typing import List, Optional
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from app.models.embeddings import EmbeddingWrapper

VECTOR_DIR = "data/vectorstore"

def _get_vector_folder_path(collection_id: str) -> str:
    """Get the folder path for the collection"""
    # Clean the collection_id to avoid filesystem issues
    clean_id = collection_id.replace(" ", "_").replace(":", "_")
    return os.path.join(VECTOR_DIR, clean_id)

def load_vectorstore(collection_id: str, embedding_model) -> Optional[FAISS]:
    """
    Load existing FAISS index if available.
    """
    folder_path = _get_vector_folder_path(collection_id)
    
    print(f"🔍 Looking for vector store at: {folder_path}")
    print(f"🔍 Directory exists: {os.path.exists(folder_path)}")
    
    if os.path.exists(folder_path):
        # Check what files are in the directory
        files = os.listdir(folder_path) if os.path.exists(folder_path) else []
        print(f"🔍 Files in vector store directory: {files}")
        
        # FAISS typically saves as index.faiss and index.pkl
        required_files = ['index.faiss', 'index.pkl']
        missing_files = [f for f in required_files if f not in files]
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return None
        
        try:
            print(f"🔄 Loading existing vector store from: {folder_path}")
            vectorstore = FAISS.load_local(
                folder_path=folder_path,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Successfully loaded vector store")
            return vectorstore
        except Exception as e:
            print(f"❌ Error loading existing vector store: {e}")
            print(f"❌ Exception type: {type(e).__name__}")
            return None
    else:
        print(f"📁 No existing vector store found at: {folder_path}")
        # Let's also check what directories exist in VECTOR_DIR
        if os.path.exists(VECTOR_DIR):
            existing_dirs = os.listdir(VECTOR_DIR)
            print(f"📁 Existing collections: {existing_dirs}")
        return None

def add_documents_to_store(docs: List[Document], collection_id: str) -> int:
    """
    Embed documents and add to FAISS vector store.
    """
    if not docs:
        raise ValueError("No documents provided to add to store")
    
    print(f"🔄 Processing {len(docs)} documents for collection: {collection_id}")
    
    # Create directories
    os.makedirs(VECTOR_DIR, exist_ok=True)
    folder_path = _get_vector_folder_path(collection_id)
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"📁 Vector store path: {folder_path}")
    
    # Initialize embedding model
    embedding_wrapper = EmbeddingWrapper()
    embedding_model = embedding_wrapper._emb
    
    # Try to load existing store
    faiss_store = load_vectorstore(collection_id, embedding_model)
    
    if faiss_store is None:
        # Create new vector store from documents
        print(f"🔄 Creating new vector store for {collection_id}")
        try:
            faiss_store = FAISS.from_documents(docs, embedding_model)
            print(f"✅ Successfully created new vector store")
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    else:
        # Add to existing store
        print(f"🔄 Adding to existing vector store for {collection_id}")
        try:
            faiss_store.add_documents(docs)
            print(f"✅ Successfully added documents to existing store")
        except Exception as e:
            print(f"❌ Error adding to existing vector store: {e}")
            raise
    
    # Save the store
    try:
        faiss_store.save_local(folder_path=folder_path)
        print(f"✅ Vector store saved to: {folder_path}")
        
        # Verify the save was successful
        saved_files = os.listdir(folder_path)
        print(f"✅ Saved files: {saved_files}")
        
    except Exception as e:
        print(f"❌ Error saving vector store: {e}")
        raise
    
    print(f"✅ Total documents processed: {len(docs)}")
    return len(docs)

def search_similar_docs(query: str, collection_id: str, k: int = 4) -> List[Document]:
    """
    Search top-k similar chunks for a query.
    """
    print(f"🔍 Searching for: '{query}' in collection: {collection_id}")
    print(f"🔍 Requested top-k: {k}")
    
    # Initialize embedding model (same as used during ingestion)
    embedding_wrapper = EmbeddingWrapper()
    embedding_model = embedding_wrapper._emb
    
    # Load vector store
    faiss_store = load_vectorstore(collection_id, embedding_model)
    
    if faiss_store is None:
        error_msg = f"No vector store found for collection: {collection_id}"
        print(f"❌ {error_msg}")
        
        # Provide debugging info
        folder_path = _get_vector_folder_path(collection_id)
        print(f"❌ Expected path: {folder_path}")
        
        if os.path.exists(VECTOR_DIR):
            existing_collections = os.listdir(VECTOR_DIR)
            print(f"❌ Available collections: {existing_collections}")
        
        raise ValueError(error_msg)
    
    try:
        results = faiss_store.similarity_search(query, k=k)
        print(f"✅ Found {len(results)} similar documents")
        
        # Log some info about the results
        for i, doc in enumerate(results):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  Result {i+1}: {preview}...")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during similarity search: {e}")
        raise

def list_collections() -> List[str]:
    """
    List all available collections in the vector store.
    """
    if not os.path.exists(VECTOR_DIR):
        return []
    
    collections = []
    for item in os.listdir(VECTOR_DIR):
        item_path = os.path.join(VECTOR_DIR, item)
        if os.path.isdir(item_path):
            collections.append(item)
    
    return collections

def get_collection_info(collection_id: str) -> dict:
    """
    Get information about a specific collection.
    """
    folder_path = _get_vector_folder_path(collection_id)
    
    info = {
        "collection_id": collection_id,
        "folder_path": folder_path,
        "exists": os.path.exists(folder_path),
        "files": [],
        "size_bytes": 0
    }
    
    if info["exists"]:
        info["files"] = os.listdir(folder_path)
        for file in info["files"]:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                info["size_bytes"] += os.path.getsize(file_path)
    
    return info