import uuid
from sentence_transformers import SentenceTransformer
from typing import List , Dict , Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams , Distance, PointStruct
from app.config import  settings

class DenseRetriever:
    def __init__(self):
        print(f"loading embedding model : {settings.EMBEDDINGS_MODEL_NAME}...")
        self.model = SentenceTransformer(settings.EMBEDDINGS_MODEL_NAME )

        self.client = QdrantClient(host = settings.QDRQNT_HOST , port = settings.QDRQNT_PORT)
        self.collection_name = settings.COLLECTION_NAME

        self.create_collection_if_not_exists()


    def create_collection_if_not_exists(self):
        """Creates Qdrant collection if is does not already exist."""    

        collections = [col.name for col in self.client.get_collections().collections]

        if self.collection_name not in collections:
            print(f"Create Qdrant collection {self.collection_name}")
            self.client.create_collection(
                collection_name =self.collection_name,
                vector_config = VectorParams(
                    size = 384,
                    distance = Distance.COSINE
                )
            )


    def index_chunks(self , chunks : List[Dict[str,Any]]):
        """ Embeds text chunks into dense vectors and upsert them into Qdrant"""

        if not chunks:
            print("No chunks provided for indexing")
            return

        text = [c["chunk_text"] for c in chunks]
        print(f"Generate dense emedding fo {len(text)} chunks...")
        embeddings = self.model.encode(text,show_progress_bar=True)

        points = []

        for idx, (chunk, vector) in enumerate(zip(chunks,embeddings)):

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,chunk["chunk_id"]))

            points.append(
                PointStruct(
                    id=point_id,
                    vector = vector.tolist(),
                    payload = {
                        "chunk_id" : chunk["chunk_id"],
                        "text" : chunk["chunk_text"],
                        "source" : chunk["source"],
                        "page_number" : chunk["page_number"]
                    }
                )
            )

        print(f"Upserting {len(points)} points into Qdrant collection '{self.collection_name}'...")
        self.client.upsert(collection_name = self.collection_name, points  = points)
        print("Indexing complete!")


    def search(self,query:str , top_k : int =5)-> List[Dict[str,Any]]:

        """
        Converts a search query into a dense vector and searches Qdrant.
        """
        query_vector = self.model.encode(query).tolist()

        search_results = self.client.search(

            collection_name = self.collection_name,
            query_vector = query_vector,
            limit = top_k
            )

        results = []

        for hit in search_results:
            results.append(
                {
                    "chunk_id":hit.payload["chunk_id"],
                    "score": round(hit.score,4),
                    "text" : hit.payload["text"],
                    "source": hit.payload["source"],
                    "page_number": hit.payload["page_number"]
                }
            )

        return results

