from pydantic import BaseModel, Field, PrivateAttr
from typing import Dict, List, Any
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from sentence_transformers import SentenceTransformer
import chromadb
from crewai.tools import BaseTool


class CombinedQueryInput(BaseModel):
    image_path: str = Field(..., description="Path to query image file")
    text_query: str = Field(..., description="Text query for product search")
    top_k: int = Field(default=5, description="Number of results to return")
    weight_image: float = Field(default=0, description=" USE ONLY 0 DO NOT CHANGE")


class CombinedQueryTool(BaseTool):
    name: str = "Combined Multimodal Search Tool"
    description: str = "Combines image and text queries for product search"
    args_schema: type[BaseModel] = CombinedQueryInput

    _clip_model: AutoModel = PrivateAttr()
    _clip_processor: AutoImageProcessor = PrivateAttr()
    _text_model: SentenceTransformer = PrivateAttr()
    _chroma_collection: chromadb.Collection = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_clip_model',
                           AutoModel.from_pretrained("facebook/dinov2-base"))
        object.__setattr__(self, '_clip_processor',
                           AutoImageProcessor.from_pretrained("facebook/dinov2-base"))
        object.__setattr__(self, '_text_model',
                           SentenceTransformer('all-mpnet-base-v2'))
        object.__setattr__(self, '_chroma_collection',
                           chromadb.PersistentClient(path="C:\\Users\\badri\\PycharmProjects\\PythonProject4\\clothing_store_assistant\\chroma_store_text")
                           .get_collection("ecommerce_text"))

    def _run(self, image_path: str, text_query: str,
             top_k: int = 5, weight_image: float = 0) -> Dict[str, Any]:
        try:
            # Image embedding
            image = Image.open(image_path).convert("RGB")
            inputs = self._clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                image_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # CLS token

            # Text embedding
            text_emb = self._text_model.encode(text_query)

            # Combine embeddings
            weight_text = 1.0 - weight_image
            combined_emb = (weight_image * image_emb + weight_text * text_emb)
            combined_emb = combined_emb / np.linalg.norm(combined_emb)

            # Query database
            results = self._chroma_collection.query(
                query_embeddings=[combined_emb.tolist()],
                n_results=top_k
            )

            return {
                'search_type': 'combined',
                'results': [{
                    'product_id': pid,
                    'metadata': meta,
                    'similarity_score': dist
                } for pid, meta, dist in zip(
                    results['ids'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )]
            }
        except Exception as e:
            return {'error': f"Combined search failed: {str(e)}"}

