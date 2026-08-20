"""Critic agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = AgentName.CRITIC.value

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        settings = get_settings()
        self.llm_client = llm_client or LLMClient(default_model=settings.mistral_supervisor_model)

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append verification notes."""
        if not state.final_answer:
            return state

        system_prompt = (
            "You are a Quality Assurance & Critic Agent. Review the final report against the research sources. "
            "Verify accuracy, check for potential hallucinations or missing citations, and suggest quick improvements."
        )
        user_prompt = f"Query: {state.request.query}\n\nFinal Report:\n{state.final_answer}"

        llm_res = self.llm_client.complete(system_prompt, user_prompt)
        review = llm_res.content.strip()

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=review,
                metadata={"status": "verified"},
            )
        )
        state.add_trace_event("critic_complete", {"review_length": len(review)})
        return state
