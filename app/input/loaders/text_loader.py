from datetime import datetime, timezone
import re

from app.input.schemas import NormalizedPost


class TextLoader:

    def load(self, text: str) -> NormalizedPost:

        if not isinstance(text, str):
            raise TypeError("Text input must be a string")

        normalized_text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if not normalized_text:
            raise ValueError(
                "Text input cannot be empty"
            )

        return NormalizedPost(
            post_id="text_input",
            platform="text",
            text=normalized_text,
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            metadata={
                "source_type": "direct_text"
            }
        )