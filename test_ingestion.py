from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriver
def test_pipeline():

    file_path = "data/sample_docs/LLM_Observatory_Documentation.pdf"

    print("---loading PDF---")
    pages = load_pdf(file_path)
    print(f"Successfuly loaded {len(pages)} pages from {file_path}")

    print("\n--- 2. Chunking Documents ---")
    chunks = chunk_documents(pages, chunk_size=300, chunk_overlap=30)
    print(f"Generated {len(chunks)} total chunks.")

    print("\n--- . Initializing Dense Retriever & Qdrant ---")
    retriever = DenseRetriever()

    print("\n--- . Indexing Chunks into Qdrant ---")
    retriever.index_chunks(chunks)

    print("\n--- . Executing Dense Vector Search ---")
    query = "What is the main topic discussed in the document?"
    results = retriever.search(query, top_k=2)



    for i, res in enumerate(results, start=1):
        print(f"\nResult #{i} (Score: {res['score']}):")
        print(f"Source: {res['source']} (Page {res['page_number']})")
        print(f"ID:     {res['chunk_id']}")
        print(f"Text:   {res['text'][:150]}...")


    print("\n--- 4. Testing Sparse Retrieval (BM25) ---")
    sparse = SparseRetriver()
    sparse.index_chunks(chunks)
    sparse_results = sparse.search("schéma en étoile", top_k=2)

    print("\nTop Sparse Match:")
    print(f"ID: {sparse_results[0]['chunk_id']} | Score: {sparse_results[0]['score']}")
    print(f"Text: {sparse_results[0]['text'][:100]}...")
if __name__ == "__main__":
    test_pipeline()