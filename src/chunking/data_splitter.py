from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from models.embedding_model import model
from collections import defaultdict
import hashlib
import os

class Chunking:
    def __init__(self, document):
        self.document = document

    def add_chunk_id(self, chunks):
        counter = defaultdict(int)

        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            key = (source, page)

            idx = counter[key]
            counter[key] += 1

            raw = f"{source}|{page}|{idx}|{chunk.page_content}"
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

        return chunks

    def recursive_chunk(self, chunk_size: int, chunk_overlap: int, separators: list):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        chunks = text_splitter.split_documents(self.document)
        return self.add_chunk_id(chunks)

    def semantic_chunk(self, breakpoint_threshold_type: str, breakpoint_threshold_amount: int):
        semantic_chunker = SemanticChunker(
            embeddings=model,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
        chunks = semantic_chunker.split_documents(self.document)
        return self.add_chunk_id(chunks)
