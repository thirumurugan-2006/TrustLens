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

        from app.input.schemas import Platform, PostType, PostContent, AcquisitionMethod

        return NormalizedPost(
            post_id="text_input",
            platform=Platform.text,
            post_type=PostType.text,
            content=PostContent(text=normalized_text),
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            metadata={
                "source_type": "direct_text"
            },
            acquisition=AcquisitionMethod.copied_text
        )