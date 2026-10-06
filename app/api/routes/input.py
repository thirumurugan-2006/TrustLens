import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas.responses import RedditInputRequest, TextInputRequest, PlatformInputRequest
from app.input.adapters.reddit_adapter import RedditAccessError
from app.input.adapters.linkedin_adapter import LinkedInAccessError
from app.input.adapters.base_adapter import PlatformAccessError
from app.pipeline.input_pipeline import InputPipeline

router = APIRouter()
pipeline = InputPipeline()


@router.post("/input/text")
def text_input(request: TextInputRequest):
    try:
        return pipeline.process("text", request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/input/screenshot")
async def upload_screenshot(file: UploadFile | None = File(default=None)):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Screenshot file is required.")

    suffix = Path(file.filename).suffix or ".upload"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = temporary_file.name
            while chunk := await file.read(1024 * 1024):
                temporary_file.write(chunk)

        return pipeline.process("screenshot", temporary_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        if temporary_path:
            try:
                os.remove(temporary_path)
            except OSError:
                pass


@router.post("/input/reddit")
def reddit_input(request: RedditInputRequest):
    try:
        return pipeline.process("reddit_url", request.url)
    except RedditAccessError as exc:
        raise HTTPException(
            status_code=503,
            detail="Reddit API access is currently unavailable.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/input/platform")
def platform_input(request: PlatformInputRequest):
    try:
        return pipeline.process("platform_url", request.url)
    except PlatformAccessError as exc:
        # Graceful degradation for unavailable platforms
        return {
            "status": "unavailable",
            "platform": exc.platform,
            "reason": str(exc),
            "fallback_available": True,
            "fallback_options": ["screenshot", "direct_text"]
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
