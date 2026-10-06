from datetime import datetime, timezone
from uuid import uuid4

from app.input.schemas import NormalizedPost


class TextLoader:
    """
    Converts raw user-provided text into a NormalizedPost.
    """

    def load(self, text: str) -> NormalizedPost:
        if not isinstance(text, str):
            raise TypeError("Text input must be a string.")

        # Remove unnecessary whitespace
        cleaned_text = " ".join(text.split())

        if not cleaned_text:
            raise ValueError("Text input cannot be empty.")

        return NormalizedPost(
            post_id=f"text_{uuid4().hex[:12]}",
            platform="text",
            source_url=None,
            author=None,
            title=None,
            text=cleaned_text,
            images=[],
            comments=[],
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            metadata={
                "input_type": "text",
                "source": "user_input",
            },
        )