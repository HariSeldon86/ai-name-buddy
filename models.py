from pydantic import BaseModel, Field
from typing import List, Optional


class Word(BaseModel):
    keyword: str = Field(..., description="The keyword text")
    abbreviation: str = Field(..., description="The abbreviation of the keyword")
    description: Optional[str] = Field(None, description="A brief description of the keyword")

    def embed_format(self) -> str:
        """Returns a string representation suitable for embedding."""
        return f"Keyword: {self.keyword} | Abbreviation: {self.abbreviation} | Description: {self.description or ''}"