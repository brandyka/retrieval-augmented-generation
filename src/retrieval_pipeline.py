from retrieval.retrieval import Retrieval
from ingestion_pipeline import load_document, chunking_document
from dotenv import load_dotenv
import os

load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER")
FILE_NAME = os.getenv("FILE_NAME")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_PATH = os.path.join(SRC_DIR, DATA_FOLDER, FILE_NAME)


query = "Perusahaan apa ini ?"

#Semantic Search
#retrieval = Retrieval(query)
#results = retrieval.similarity_search(k=3)

#print("user query:", query)

#for res, score in results:
#    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

#BM25 Search
doc_load = load_document(DOCUMENT_PATH)
chunk_doc = chunking_document(doc_load, type="recursive", chunk_size=512, overlap=100, seperators=["\n\n", "\n", " ", ""])

bm25_retrieval = Retrieval(query, chunk_doc)

retriever = bm25_retrieval.keyword_search()

result_bm25 = retriever.invoke(query)

for doc in result_bm25:
    print(doc.page_content)
