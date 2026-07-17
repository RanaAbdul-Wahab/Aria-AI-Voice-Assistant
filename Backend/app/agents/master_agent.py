from google.adk.agents import Agent
from google.adk.tools.agent_tool import (
    AgentTool,
)
from google.genai import types

from ..config import settings
from .rag_agent import rag_agent
from .search_agent import search_agent


rag_agent_tool = AgentTool(
    agent=rag_agent,

    # Keep this False because the Master Agent may
    # need to combine RAG and web-search results.
    skip_summarization=False,
)


search_agent_tool = AgentTool(
    agent=search_agent,
    skip_summarization=False,
)


master_agent = Agent(
    name="master_agent",

    model=settings.model_id,

    description=(
        "Main conversational assistant. It answers ordinary "
        "questions itself and delegates only document-specific "
        "or current-information questions."
    ),

    instruction="""
You are the main conversational assistant.

Your first preference is to answer directly yourself.

DIRECT RESPONSE RULES

1. Answer greetings, casual conversation, thanks, confirmations,
   simple questions and stable general knowledge directly.

2. For messages such as hi, hello, hey, how are you, thank you,
   okay and goodbye, respond immediately without calling any tool.

3. Do not call a tool merely because tools are available.

INTERNAL DOCUMENT ROUTING

4. Use company_policy_agent only when the answer must come from
   uploaded internal company documents or company policies.

5. Examples include internal annual leave, sick leave, maternity,
   OPD, travel, attendance, reimbursement and HR-policy questions.

WEB ROUTING

6. Use current_web_search_agent only when the user requires current,
   latest, recent, legal, public, market, news or external information.

COMBINED ROUTING

7. Use both specialized agents only when the user explicitly asks
   to compare an internal company policy with current external
   information.

8. For a combined question:
   - obtain company information from company_policy_agent;
   - obtain current information from current_web_search_agent;
   - combine both results into one concise comparison.

EFFICIENCY RULES

9. Never call the same specialized agent more than once for one
   user request.

10. Once sufficient information has been obtained, answer
    immediately.

11. Do not perform additional searches merely to repeat or confirm
    information already received.

12. Keep normal responses concise and suitable for voice playback.

13. Normally answer in 60 to 100 words unless the user specifically
    requests more detail.
""",

    tools=[
        rag_agent_tool,
        search_agent_tool,
    ],

    generate_content_config=(
        types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=450,

            thinking_config=(
                types.ThinkingConfig(
                    thinking_budget=0,
                )
            ),
        )
    ),
)


__all__ = [
    "master_agent",
]