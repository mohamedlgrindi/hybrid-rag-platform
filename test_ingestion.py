from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_documents


def test_pipeline():

    file_path = "data/sample_docs/LLM_Observatory_Documentation.pdf"

    print("---loading PDF---")
    pages = load_pdf(file_path)
    print(f"Successfuly loaded {len(pages)} pages from {file_path}")

    print("\n--- 2. Chunking Documents ---")
    chunks = chunk_documents(pages, chunk_size=300, chunk_overlap=30)
    print(f"Generated {len(chunks)} total chunks.")


    print("\n--- 3. Inspecting First Chunk Output ---")
    first_chunk = chunks[0]
    print(f"Chunk ID: {first_chunk['chunk_id']}")
    print(f"Source:   {first_chunk['source']} (Page {first_chunk['page_number']})")
    print(f"Text Preview:\n{first_chunk['chunk_text'][:150]}...\n")

if __name__ == "__main__":
    test_pipeline()