from pydantic import BaseModel, Field


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
    agent: str = "master_agent"
    session_id: str


class TextToSpeechRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
    )

    language_code: str = Field(
        default="en-IN",
        min_length=2,
        max_length=20,
    )


class SpeechToTextResponse(BaseModel):
    transcript: str
    language_code: str


class HealthResponse(BaseModel):
    status: str
    project_id: str
    location: str
    model: str