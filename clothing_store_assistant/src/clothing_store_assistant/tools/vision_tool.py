from crewai.tools import BaseTool
from google import genai
from pydantic import BaseModel, Field, PrivateAttr
from google.genai import types
import os

g_api_key = os.environ['GEMINI_API_KEY']
class ClothingDescriptorInput(BaseModel):
    image_path: str = Field(..., description="Path to the clothing image file.")


class ClothingDescriptorOutput(BaseModel):
    description: str = Field(..., description="Detailed description of the clothing in the image.")

class GeminiClothingDescriptorTool(BaseTool):
    name: str = "GeminiClothingDescriptorTool"
    description: str = (
        "Analyzes a clothing image and returns a detailed fashion-oriented description "
        "for providing it to a fashion designer who will have a further look at it to provide suggestions. Describes garments, styles, patterns, textures, etc."
    )
    args_schema: type[BaseModel] = ClothingDescriptorInput
    output_schema: type[BaseModel] = ClothingDescriptorOutput

    # Declare client as a PrivateAttr
    _client: genai.Client = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the client and assign as a private attribute
        object.__setattr__(self, '_client', genai.Client(api_key=g_api_key))

    def _run(self, image_path: str) -> str:
        try:
            uploaded_file = self._client.files.upload(file=image_path)

            with open(image_path, "rb") as f:
                img2_bytes = f.read()


            # Compose prompt
            prompt = (
                "You are a fashion expert analyzing a clothing item for visual search. "
                "Describe the item in detail for cataloging purposes. Your description should include: "
                "- Any visible branding, logos, or accessories. "
                "- Pattern. "
                "- Type of garment (e.g., shirt, pants, jacket). "
                "- Material, texture, and base color. "
                "- Style elements (e.g., fit, collar type, length, sleeve type). "
                "- Gender orientation (male, female, unisex) based on cut, styling, and overall design. "
                "Use fashion-industry standard terminology. "
                "Conclude with a structured summary in this format: "
                "Includes [Brand] if identifiable. It features a [pattern] pattern. "
                "It is a [baseColour] [articleType] designed for [gender]. "
                "Best used in [usage] during [season]."
            )

            # Generate content from the model
            response = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt,uploaded_file, types.Part.from_bytes(
            data=img2_bytes,
            mime_type='image/jpg'
        )]
            )

            # Return the description as JSON
            return ClothingDescriptorOutput(description=response.text).model_dump_json(indent=2)

        except Exception as e:
            # Return error message in output schema format
            return ClothingDescriptorOutput(description=f"Error: {e}").model_dump_json(indent=2)
