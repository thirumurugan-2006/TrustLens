import os
import re
from typing import Optional

import praw
from dotenv import load_dotenv

from app.input.schemas import NormalizedComment, NormalizedPost
from app.input.adapters.base_adapter import BasePlatformAdapter, PlatformAccessError

load_dotenv()


class RedditAccessError(PlatformAccessError):
    """Raised when Reddit API access is unavailable."""
    def __init__(self, message: str):
        super().__init__(message, platform="reddit")


class RedditAdapter(BasePlatformAdapter):
    """
    Reddit input adapter for TrustLens.

    Uses Reddit's official API through PRAW.
    Does NOT scrape Reddit pages.
    """

    def __init__(self):
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = os.getenv(
            "REDDIT_USER_AGENT",
            "TrustLens/0.1"
        )

        self.reddit = None

        if self.client_id and self.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                    check_for_async=False,
                )

                # We only need read-only access.
                self.reddit.read_only = True

            except Exception as exc:
                raise RedditAccessError(
                    f"Could not initialize Reddit API client: {exc}"
                ) from exc

    def is_available(self) -> bool:
        """Return True if Reddit API credentials are configured."""
        return self.reddit is not None

    def health_check(self) -> dict:
        if self.is_available():
            return {"platform": "reddit", "status": "available", "reason": "API configured"}
        return {"platform": "reddit", "status": "unavailable", "reason": "API credentials missing"}

    def can_handle(self, url: str) -> bool:
        return self.validate_url(url)

    def fetch(self, url: str) -> NormalizedPost:
        return self.fetch_post(url)

    @staticmethod
    def extract_post_id(url: str) -> Optional[str]:
        """
        Extract a Reddit post ID from a Reddit URL.

        Example:
        https://www.reddit.com/r/python/comments/abc123/example/
        -> abc123
        """

        pattern = r"/comments/([a-zA-Z0-9]+)"

        match = re.search(pattern, url)

        if match:
            return match.group(1)

        return None

    @staticmethod
    def validate_url(url: str) -> bool:
        """Check whether the URL looks like a Reddit post URL."""

        return bool(
            re.match(
                r"^https?://(www\.)?reddit\.com/r/[^/]+/comments/",
                url,
                re.IGNORECASE,
            )
        )

    def fetch_post(
        self,
        url: str,
        comment_limit: int = None,
    ) -> NormalizedPost:
        if comment_limit is None:
            comment_limit = int(os.getenv("REDDIT_COMMENT_LIMIT", "20"))

        if not self.validate_url(url):
            raise ValueError(
                "Invalid Reddit post URL."
            )

        if not self.is_available():
            raise RedditAccessError(
                "Reddit API access is not available. "
                "Configure approved Reddit API credentials first."
            )

        try:
            submission = self.reddit.submission(url=url)

            # Expand comment tree.
            submission.comments.replace_more(limit=0)

            comments = []

            for comment in submission.comments.list()[:comment_limit]:

                if not comment.body:
                    continue

                comments.append(
                    NormalizedComment(
                        comment_id=str(comment.id),
                        parent_id=self._get_parent_id(comment),
                        author=self._get_author(comment),
                        text=comment.body,
                        timestamp=self._get_timestamp(
                            comment.created_utc
                        ),
                        score=getattr(comment, "score", None),
                        metadata={
                            "subreddit": str(
                                submission.subreddit
                            )
                        },
                    )
                )

            return NormalizedPost(
                post_id=str(submission.id),
                platform="reddit",
                source_url=url,
                author=self._get_author(submission),
                title=submission.title or "",
                text=submission.selftext or "",
                images=self._extract_images(submission),
                comments=comments,
                timestamp=self._get_timestamp(
                    submission.created_utc
                ),
                metadata={
                    "subreddit": str(submission.subreddit),
                    "score": getattr(
                        submission,
                        "score",
                        None,
                    ),
                    "num_comments": getattr(
                        submission,
                        "num_comments",
                        None,
                    ),
                    "upvote_ratio": getattr(
                        submission,
                        "upvote_ratio",
                        None,
                    ),
                    "is_self": getattr(
                        submission,
                        "is_self",
                        None,
                    ),
                },
            )

        except RedditAccessError:
            raise

        except Exception as exc:
            raise RedditAccessError(
                f"Failed to retrieve Reddit post: {exc}"
            ) from exc

    @staticmethod
    def _get_author(obj):
        """Safely obtain an author's username."""

        try:
            if obj.author is None:
                return None

            return str(obj.author.name)

        except Exception:
            return None

    @staticmethod
    def _get_parent_id(comment):
        """Return the parent comment ID when available."""

        try:
            parent_id = comment.parent_id

            if parent_id:
                return parent_id.split("_")[-1]

        except Exception:
            pass

        return None

    @staticmethod
    def _get_timestamp(timestamp):
        """Convert Unix timestamp to ISO format."""

        from datetime import datetime, timezone

        if timestamp is None:
            return None

        return datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc,
        ).isoformat()

    @staticmethod
    def _extract_images(submission):
        """
        Extract image URLs where available.

        This does not download images yet.
        """

        images = []

        try:
            url = submission.url

            if url and any(
                url.lower().endswith(ext)
                for ext in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                ]
            ):
                images.append(url)

        except Exception:
            pass

        return images