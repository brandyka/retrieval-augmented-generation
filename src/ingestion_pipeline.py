from ingestions.data_loader import DataLoader
from chunking.data_splitter import Chunking
from embedding.embedder import Embedding
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_PATH = os.path.join(SRC_DIR, "data", "company_profile.pdf")

def load_document(document_path):
    load_document = DataLoader(document_path)

    if document_path.lower().endswith(".pdf"):
        document = load_document.load_pdf()
    else:
        print("File except PDF not supported")

    return document

def chunking_document(document, type=None, chunk_size: int = None, overlap: int = None, seperators: list = None, breakpoint_threshold_type: str = None, breakpoint_threshold_amount: int = None):
    chunker = Chunking(document)

    if type == "recursive":
        return chunker.recursive_chunk(chunk_size, overlap, seperators)
    elif type == "semantic":
        return chunker.semantic_chunk(breakpoint_threshold_type, breakpoint_threshold_amount)
    else:
        print("Chunking type not supported")
        return None
    

def save_to_vector_db(chunks):
    docs_embedder = Embedding(chunks)
    vector_db = docs_embedder.save_to_db()

    return vector_db

def main():
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCUMENT_PATH = os.path.join(SRC_DIR, "data", "company_profile.pdf")

    document = load_document(DOCUMENT_PATH)
    print(f"Loaded document with {len(document)} pages.")

    chunks = chunking_document(document, type="recursive", chunk_size=512, overlap=100, seperators=["\n\n", "\n", " ", ""])
    print(f"Created {len(chunks)} chunks using recursive chunking.")

    vector_db = save_to_vector_db(chunks)
    print("Saved chunks to vector database.")
    return vector_db


if __name__ == "__main__":
    main()