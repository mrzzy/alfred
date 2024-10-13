#
# Alfred
# Vector Embeddings
#

from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field


class Embedding(BaseModel):
    provider: str
    model: str


def build_embedding(spec: Embedding) -> Embeddings:
    """Build Embeddings instance from given spec specification."""

    # setup embedding model
    if spec.provider == "sentence_transformers":

        class SentenceTransformerEmbeddings(Embeddings):
            def __init__(self, model_name: str):
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.model.encode(texts).tolist()

            def embed_query(self, text: str) -> list[float]:
                return self.model.encode([text])[0]

        return SentenceTransformerEmbeddings(spec.model)
    elif spec.provider == "voyageai":
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(model=spec.model, batch_size=128)
    else:
        raise ValueError(f"Unsupported embedding provider: {spec.provider})")
