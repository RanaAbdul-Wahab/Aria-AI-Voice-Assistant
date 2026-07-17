import logging
import os
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from pydantic import (
    BaseModel,
    Field,
)

from .agent_runtime import AgentRuntime
from .agents.master_agent import (
    master_agent,
)
from .config import get_settings
from .routers.speech import (
    router as speech_router,
)


logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    __name__
)


settings = get_settings()

master_runtime = AgentRuntime(
    master_agent
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=5000,
    )

    user_id: str = Field(
        default="voice-ai-user",
        min_length=1,
        max_length=200,
    )

    session_id: str | None = Field(
        default=None,
        max_length=250,
    )


class ChatResponse(BaseModel):
    answer: str
    agent: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    project_id: str
    location: str
    model: str


TTS_MAX_RESPONSE_BYTES = int(
    os.getenv(
        "TTS_MAX_RESPONSE_BYTES",
        "1800",
    )
)


def limit_text_for_tts(
    text: str,
    max_bytes: int = TTS_MAX_RESPONSE_BYTES,
) -> str:
    """
    Keep an answer inside the configured UTF-8 size limit.
    """

    clean_text = re.sub(
        r"[ \t]+",
        " ",
        text,
    ).strip()

    clean_text = re.sub(
        r"\n{3,}",
        "\n\n",
        clean_text,
    )

    if (
        len(clean_text.encode("utf-8"))
        <= max_bytes
    ):
        return clean_text

    ending = (
        "\n\nThe response was shortened "
        "for voice playback."
    )

    selected_words: list[str] = []

    for word in clean_text.split():
        candidate = " ".join(
            [
                *selected_words,
                word,
            ]
        )

        candidate_with_ending = (
            candidate + ending
        )

        if (
            len(
                candidate_with_ending.encode(
                    "utf-8"
                )
            )
            > max_bytes
        ):
            break

        selected_words.append(
            word
        )

    shortened_text = " ".join(
        selected_words
    ).rstrip(
        " ,;:-"
    )

    last_sentence_end = max(
        shortened_text.rfind("."),
        shortened_text.rfind("?"),
        shortened_text.rfind("!"),
    )

    if (
        last_sentence_end
        > len(shortened_text) * 0.6
    ):
        shortened_text = (
            shortened_text[
                : last_sentence_end + 1
            ]
        )

    return shortened_text + ending


@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    logger.info(
        "Starting Voice AI Assistant backend."
    )

    yield

    logger.info(
        "Closing Voice AI Assistant backend."
    )

    await master_runtime.close()


app = FastAPI(
    title="Multi-Agent Voice AI API",
    version="1.0.0",

    description=(
        "FastAPI backend for the Master Agent, "
        "company-policy RAG, web search, "
        "Speech-to-Text and Text-to-Speech."
    ),

    lifespan=lifespan,
)


allowed_origins = list(
    dict.fromkeys(
        [
            settings.frontend_origin,
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=(
        allowed_origins
    ),

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Adds:
# POST /api/stt
# POST /api/tts
app.include_router(
    speech_router
)


@app.get(
    "/",
)
def root() -> dict[str, str]:
    return {
        "message": (
            "Multi-Agent Voice AI backend "
            "is running."
        ),
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        project_id=(
            settings.project_id
        ),
        location=(
            settings.location
        ),
        model=(
            settings.model_id
        ),
    )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
async def chat_with_master_agent(
    request: ChatRequest,
) -> ChatResponse:
    session_id = (
        request.session_id
        or str(uuid.uuid4())
    )

    try:
        answer = await master_runtime.ask(
            question=request.question,

            user_id=request.user_id,

            session_id=session_id,
        )

        answer = limit_text_for_tts(
            answer
        )

        return ChatResponse(
            answer=answer,
            agent="master_agent",
            session_id=session_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Master Agent request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The assistant could not process "
                "the request."
            ),
        ) from error