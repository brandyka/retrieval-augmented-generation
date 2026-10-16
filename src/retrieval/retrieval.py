from models.embedding_model import model
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
import os

SRC_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VECTOR_DB_DIR = os.path.join(SRC_DIR, "vectordb")


class Retrieval:
    def __init__(self, query, doc: list = None):
        self.query = query
        self.doc = doc

    def similarity_search(self, k=3):
        vector_db = Chroma(
            collection_name="document",
            persist_directory=VECTOR_DB_DIR,
            embedding_function=model,
        )

        print("Database path:", VECTOR_DB_DIR)
        print("Active collection:", vector_db._collection.name)
        print("Document count:", vector_db._collection.count())

        results = vector_db.similarity_search_with_score(
            self.query,
            k=k,
        )

        print("Retrieved documents:", len(results))
        return results

    def keyword_search(self):
        retriever = BM25Retriever.from_documents(self.doc)
        return retriever

        
