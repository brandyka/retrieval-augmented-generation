from models.embedding_model import model
from langchain_chroma import Chroma
import os

class Embedding:
    def __init__(self, documents ):
        self.documents = documents

    def embedder(self):
        embeddings = model.embed_documents(self.documents)
        return embeddings
    
    def save_to_db(self):
            persist_directory = os.path.join(
                os.getcwd(),
                "vectordb"
            )

            os.makedirs(persist_directory, exist_ok=True)

            vector_db = Chroma(
                embedding_function=model,
                collection_name="document",
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )

            vector_db.add_documents(self.documents)

            return vector_db

#Embedding docs
#Embedding = Embedding(docs)
#save_to_db = Embedding.save_to_db(Embedding.embedder())

#Embedding query
#QueryEmbedding = Embedding(query_docs)
#query_embeddings = QueryEmbedding.embedder()
