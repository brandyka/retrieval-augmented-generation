from retrieval.retrieval import Retrieval

query = "Perusahaan apa ini ?"

retrieval = Retrieval(query)
results = retrieval.similarity_search(k=3)

print("user query:", query)

for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")