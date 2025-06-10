from pydantic import BaseModel, Field, PrivateAttr
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
import chromadb
import os
from crewai.tools import BaseTool


class TextQueryInput(BaseModel):
    text_query: str = Field(..., description="Text query for product search")
    top_k: int = Field(default=5, description="Number of results to return")


class TextQueryTool(BaseTool):
    name: str = "Text Product Search Tool"
    description: str = "Searches products using text queries with CLIP embeddings"
    args_schema: type[BaseModel] = TextQueryInput

    _text_model: SentenceTransformer = PrivateAttr()
    _chroma_collection: chromadb.Collection = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        project_root = os.path.dirname(os.path.abspath(__file__))
        chroma_path = os.path.join(project_root, "..", "..", "chroma_store_text")
        chroma_path = os.path.abspath(chroma_path)  # Normalize to full absolute path

        object.__setattr__(self, '_text_model',
                           SentenceTransformer('all-mpnet-base-v2'))
        object.__setattr__(self, '_chroma_collection',
                           chromadb.PersistentClient(path=chroma_path)
                           .get_collection("ecommerce_text"))


    def _run(self, text_query: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            emb = self._text_model.encode(text_query).tolist()
            results = self._chroma_collection.query(
                query_embeddings=[emb],
                n_results=top_k
            )

            return {
                'search_type': 'text_only',
                'results': [{
                    'product_id': pid,
                    'metadata': meta,
                    'similarity_score': 1 - dist
                } for pid, meta, dist in zip(
                    results['ids'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )]
            }
        except Exception as e:
            return {'error': f"Text search failed: {str(e)}"}
