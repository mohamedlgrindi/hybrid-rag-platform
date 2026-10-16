from typing import List , Dict ,Any
from rank_bm25 import BM25Okapi


class SparseRetriver: 
    def __init__(self):

        self.chunks:List[Dict[str,Any]] = []

        self.bm25 : BM25Okapi = None

    def tokenize(self,text : str)->List[str]:

        """
        Tokenizes text by converting to lowercase and splitting on whitespace.
        Example: "Star Schema" -> ["star", "schema"]
        """

        return text.lower().split()

    def index_chunks(self, chunks: List[Dict[str,Any]]):

        """
        Tokenizes all chunk passages and builds the BM25Okapi inverted index.
        """
        if not chunks:
            print("No chunks provided for BM25 indexing.")


        self.chunks = chunks


        corpus = [self.tokenize(c["chunk_text"]) for c in self.chunks]
        print(f"Building BM25 sparse index for {len(corpus)} chunks...")

        self.bm25 = BM25Okapi(corpus)
        print("BM25 indexing complete")




    def search(self , query: str , top_k : int = 5)-> List[Dict[str,Any]]:
        """
        Tokenizes query, computes BM25 relevance scores for all chunks,
        and returns top_k matching chunks formatted uniformly with Qdrant outputs.
        """

        if not self.bm25 or not self.chunks:
            print("BM25 index is empty. Please index chunks first.")
            return []


        tokenized_query = self.tokenize(query)

        # 2. Compute BM25 numerical scores for all chunks in the index
        # Returns a 1D list of scores corresponding to self.chunks indices
        scores = self.bm25.get_scores(tokenized_query)

        scored_chunks = list(zip(range(len(self.chunks)),scores))
        sorted_chunks = sorted(scored_chunks, key= lambda x : x[1],reverse = True)

        results = []

        for idx , score in sorted_chunks:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "score" : round(float(score),4),
                "text": chunk["chunk_text"],
                "source" : chunk["source"],
                "page_number" : chunk["page_number"]
            })


        return results