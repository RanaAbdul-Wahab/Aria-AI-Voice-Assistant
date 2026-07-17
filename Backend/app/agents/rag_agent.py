import logging

import vertexai
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import (
    VertexAiRagRetrieval,
)
from google.genai import types
from vertexai.preview import rag

from ..config import settings


logger = logging.getLogger(__name__)


vertexai.init(
    project=settings.project_id,
    location=settings.location,
)


company_policy_retrieval = VertexAiRagRetrieval(
    name="retrieve_company_policy_documents",

    description=(
        "Retrieve passages only from uploaded internal company "
        "documents and company policies. Use this only for internal "
        "leave, maternity, sick leave, annual leave, OPD, travel, "
        "attendance, reimbursement, HR or other company-policy "
        "questions. Do not use it for greetings, casual conversation, "
        "general knowledge, news or public web information."
    ),

    rag_resources=[
        rag.RagResource(
            rag_corpus=(
                settings.rag_corpus_name
            ),
        ),
    ],

    similarity_top_k=(
        settings.rag_top_k
    ),

    vector_distance_threshold=(
        settings.rag_distance_threshold
    ),
)


rag_agent = Agent(
    name="company_policy_agent",

    model=settings.model_id,

    description=(
        "Answers only questions requiring uploaded internal company "
        "policy documents. It must not handle greetings, casual chat, "
        "general knowledge or current public information."
    ),

    instruction="""
You are the internal company-policy specialist.

You receive questions that specifically require uploaded internal
company documents.

RULES

1. Use retrieve_company_policy_documents for the user's question.

2. Base your answer only on information retrieved from the uploaded
   documents.

3. Do not use general knowledge to invent company rules.

4. If the retrieved passages do not contain the answer, clearly say
   that the information was not found in the uploaded documents.

5. Mention the relevant policy name when it is available.

6. Do not call the retrieval tool repeatedly for the same question.

7. Keep the answer concise, clear and suitable for spoken audio.

8. Normally respond in no more than 100 words.
""",

    tools=[
        company_policy_retrieval,
    ],

    generate_content_config=(
        types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=400,

            # Reduces latency for Gemini 2.5 Flash.
            thinking_config=(
                types.ThinkingConfig(
                    thinking_budget=0,
                )
            ),
        )
    ),
)


__all__ = [
    "rag_agent",
]