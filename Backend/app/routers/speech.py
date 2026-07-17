import logging

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from ..schemas import (
    SpeechToTextResponse,
    TextToSpeechRequest,
)
from ..services.speech_to_text import (
    SpeechToTextService,
)
from ..services.text_to_speech import (
    TextToSpeechService,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api",
    tags=["speech"],
)


speech_to_text_service = SpeechToTextService()

text_to_speech_service = TextToSpeechService()


@router.post(
    "/stt",
    response_model=SpeechToTextResponse,
)
async def speech_to_text(
    audio: UploadFile = File(...),

    language_code: str = Form(
        default="en-IN",
    ),
) -> SpeechToTextResponse:
    """
    Convert an uploaded browser recording into text
    using Google Cloud Speech-to-Text Chirp 3.
    """

    try:
        audio_content = await audio.read()

        if not audio_content:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded audio recording is empty."
                ),
            )

        logger.info(
            "STT request: filename=%s type=%s size=%s language=%s",
            audio.filename,
            audio.content_type,
            len(audio_content),
            language_code,
        )

        transcript = await run_in_threadpool(
            speech_to_text_service.transcribe,
            audio_content,
            language_code,
        )

        return SpeechToTextResponse(
            transcript=transcript,
            language_code=language_code,
        )

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Speech-to-text request failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Speech transcription failed.",
        ) from error

    finally:
        await audio.close()


@router.post("/tts")
async def text_to_speech(
    request: TextToSpeechRequest,
) -> Response:
    """
    Convert assistant text into WAV audio.
    """

    try:
        clean_text = request.text.strip()

        if not clean_text:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty.",
            )

        logger.info(
            "TTS request: characters=%s language=%s",
            len(clean_text),
            request.language_code,
        )

        audio_content = await run_in_threadpool(
            text_to_speech_service.synthesize,
            clean_text,
            request.language_code,
        )

        if not audio_content:
            raise RuntimeError(
                "The TTS service returned empty audio."
            )

        logger.info(
            "TTS generated %s bytes.",
            len(audio_content),
        )

        return Response(
            content=audio_content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": (
                    'inline; filename="assistant-response.wav"'
                ),
                "Cache-Control": "no-store",
            },
        )

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Text-to-speech request failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Voice generation failed.",
        ) from error