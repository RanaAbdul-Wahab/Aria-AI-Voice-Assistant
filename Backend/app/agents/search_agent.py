from google.adk.agents import Agent
from google.adk.tools import google_search
from google.genai import types

from ..config import settings


search_agent = Agent(
    name="current_web_search_agent",

    model=settings.model_id,

    description=(
        "Searches the public web only when current, latest, recent, "
        "legal, market, news or external public information is "
        "required. It must not handle greetings or internal company "
        "policy questions."
    ),

    instruction="""
You are the current public-information specialist.

RULES

1. Use Google Search only when the question requires information
   that is current, latest, recent, legal, public, external, market
   related or news related.

2. Do not answer questions about internal company policy documents.

3. Use reliable and relevant public sources.

4. Do not perform repeated searches for the same information.

5. Mention when information is uncertain or varies by jurisdiction.

6. Keep the final answer concise and suitable for voice playback.

7. Normally respond in no more than 100 words.
""",

    tools=[
        google_search,
    ],

    generate_content_config=(
        types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=400,

            thinking_config=(
                types.ThinkingConfig(
                    thinking_budget=0,
                )
            ),
        )
    ),
)


__all__ = [
    "search_agent",
]