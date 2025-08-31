# app/rag/ingest.py
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

UPLOAD_DIR = "data/uploads"

def prepare_chunks(file_id: str) -> List:
    """
    Prepare document chunks with better error handling
    """
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_id}.pdf not found in uploads folder")

    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"✅ Pages extracted: {len(pages)}")
        
        # Check if pages were actually extracted
        if not pages:
            raise ValueError(f"No content could be extracted from {file_id}.pdf")
        
        # Check if pages have content
        total_content = "".join([page.page_content for page in pages])
        if not total_content.strip():
            raise ValueError(f"No text content found in {file_id}.pdf")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)
        print(f"✅ Chunks created: {len(chunks)}")
        
        # Validate chunks
        if not chunks:
            raise ValueError("No chunks could be created from the document")
        
        # Filter out empty chunks
        valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        print(f"✅ Valid chunks after filtering: {len(valid_chunks)}")
        
        return valid_chunks
        
    except Exception as e:
        print(f"❌ Error processing {file_id}.pdf: {str(e)}")
        raise