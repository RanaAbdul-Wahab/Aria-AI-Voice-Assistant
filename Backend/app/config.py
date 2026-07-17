import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


# Demo_project/Backend
BACKEND_DIRECTORY = (
    Path(__file__).resolve().parent.parent
)

ENV_FILE = BACKEND_DIRECTORY / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)


def get_first_value(
    *variable_names: str,
    default: str = "",
) -> str:
    """
    Return the first available non-empty environment value.
    """

    for variable_name in variable_names:
        value = os.getenv(
            variable_name,
            "",
        ).strip()

        if value:
            return value

    return default.strip()


def get_integer(
    variable_name: str,
    default: int,
) -> int:
    raw_value = os.getenv(
        variable_name,
        str(default),
    ).strip()

    try:
        return int(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{variable_name} must be an integer. "
            f"Current value: {raw_value}"
        ) from error


def get_float(
    variable_name: str,
    default: float,
) -> float:
    raw_value = os.getenv(
        variable_name,
        str(default),
    ).strip()

    try:
        return float(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{variable_name} must be a number. "
            f"Current value: {raw_value}"
        ) from error


@dataclass(frozen=True)
class Settings:
    project_id: str
    location: str
    model_id: str

    rag_corpus_name: str
    rag_top_k: int
    rag_distance_threshold: float

    frontend_origin: str
    app_name: str

    @property
    def rag_corpus(self) -> str:
        """
        Backward-compatible alias for older code that uses
        settings.rag_corpus.
        """

        return self.rag_corpus_name

    @classmethod
    def from_environment(
        cls,
    ) -> "Settings":
        project_id = get_first_value(
            "GOOGLE_CLOUD_PROJECT",
            "PROJECT_ID",
        )

        location = get_first_value(
            "GOOGLE_CLOUD_LOCATION",
            "LOCATION",
            "VERTEX_LOCATION",
            default="europe-west3",
        )

        model_id = get_first_value(
            "MODEL_ID",
            "GOOGLE_MODEL",
            default="gemini-2.5-flash",
        )

        rag_corpus_name = get_first_value(
            "RAG_CORPUS_NAME",
            "RAG_CORPUS",
            "RAG_CORPUS_RESOURCE_NAME",
            "VERTEX_RAG_CORPUS",
        )

        frontend_origin = get_first_value(
            "FRONTEND_ORIGIN",
            default="http://localhost:5173",
        )

        app_name = get_first_value(
            "ADK_APP_NAME",
            default="voice_ai_assistant",
        )

        missing_variables: list[str] = []

        if not project_id:
            missing_variables.append(
                "GOOGLE_CLOUD_PROJECT"
            )

        if not rag_corpus_name:
            missing_variables.append(
                "RAG_CORPUS_NAME"
            )

        if missing_variables:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing_variables)
                + ". Add them to Backend/.env."
            )

        return cls(
            project_id=project_id,
            location=location,
            model_id=model_id,

            rag_corpus_name=(
                rag_corpus_name
            ),

            rag_top_k=get_integer(
                "RAG_TOP_K",
                5,
            ),

            rag_distance_threshold=get_float(
                "RAG_DISTANCE_THRESHOLD",
                0.6,
            ),

            frontend_origin=(
                frontend_origin
            ),

            app_name=app_name,
        )


settings = Settings.from_environment()


# Ensure ADK and Google Gen AI use Vertex AI.
os.environ["GOOGLE_CLOUD_PROJECT"] = (
    settings.project_id
)

os.environ["GOOGLE_CLOUD_LOCATION"] = (
    settings.location
)

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = (
    "TRUE"
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the shared application settings.
    """

    return settings


__all__ = [
    "Settings",
    "settings",
    "get_settings",
]