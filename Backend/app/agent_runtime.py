import asyncio
import logging
from typing import Optional

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import (
    InMemorySessionService,
)
from google.genai import types

from .config import settings


logger = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(
        self,
        agent: BaseAgent,
    ) -> None:
        self.agent = agent

        self.app_name = (
            settings.app_name
        )

        self.session_service = (
            InMemorySessionService()
        )

        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=(
                self.session_service
            ),
        )

        self._session_creation_lock = (
            asyncio.Lock()
        )


    async def ensure_session(
        self,
        user_id: str,
        session_id: str,
    ) -> None:
        existing_session = (
            await self.session_service
            .get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )
        )

        if existing_session is not None:
            return

        async with (
            self._session_creation_lock
        ):
            existing_session = (
                await self.session_service
                .get_session(
                    app_name=self.app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
            )

            if (
                existing_session
                is not None
            ):
                return

            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )

            logger.info(
                "Created ADK session: "
                "user_id=%s session_id=%s",
                user_id,
                session_id,
            )


    @staticmethod
    def extract_text_from_event(
        event,
    ) -> str:
        content = getattr(
            event,
            "content",
            None,
        )

        if content is None:
            return ""

        parts = getattr(
            content,
            "parts",
            None,
        )

        if not parts:
            return ""

        text_parts: list[str] = []

        for part in parts:
            text = getattr(
                part,
                "text",
                None,
            )

            if text:
                text_parts.append(
                    text
                )

        return "".join(
            text_parts
        ).strip()


    async def ask(
        self,
        question: str,
        user_id: str,
        session_id: str,
    ) -> str:
        clean_question = (
            question.strip()
        )

        if not clean_question:
            raise ValueError(
                "Question cannot be empty."
            )

        await self.ensure_session(
            user_id=user_id,
            session_id=session_id,
        )

        message = types.Content(
            role="user",

            parts=[
                types.Part(
                    text=clean_question,
                ),
            ],
        )

        final_response = ""

        logger.info(
            "Starting Master Agent request: "
            "user_id=%s session_id=%s",
            user_id,
            session_id,
        )

        events = self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        )

        async for event in events:
            if not event.is_final_response():
                continue

            event_text = (
                self.extract_text_from_event(
                    event
                )
            )

            if event_text:
                final_response = (
                    event_text
                )

        if not final_response:
            raise RuntimeError(
                "The agent completed without "
                "returning a text response."
            )

        logger.info(
            "Master Agent request completed."
        )

        return final_response


    async def close(
        self,
    ) -> None:
        close_method = getattr(
            self.runner,
            "close",
            None,
        )

        if close_method is None:
            return

        result = close_method()

        if asyncio.iscoroutine(
            result
        ):
            await result


__all__ = [
    "AgentRuntime",
]