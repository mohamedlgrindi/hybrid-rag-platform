from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    PROJECT_NAME: str = "Enterprise Hybrid-RAG Platform"
    VERSION: str = "1.0.0"

    # Qdrant Vector DB Settings
    QDRQNT_HOST: str = "localhost"
    QDRQNT_PORT: int = 6333
    COLLECTION_NAME: str = "knowledge_base"

    # Embedding & Reranking Model Specifications
    # BAAI/bge-small-en-v1.5 produces high quality 384-dimensional dense vectors
    EMBEDDINGS_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    # BAAI/bge-reranker-base is a Cross-Encoder model for re-scoring search results
    RERANKER_MODEL_NAME: str = "BAAI/bge-rera,ker-base"

    class Config:
        env_file = ".env"

settings = Settings()        