"""Analyst agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = AgentName.ANALYST.value

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        settings = get_settings()
        self.llm_client = llm_client or LLMClient(default_model=settings.mistral_analyst_model)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.analysis_notes."""
        research_notes = state.research_notes or "No research notes provided."
        logger.info("Analyst processing research notes.")

        system_prompt = (
            "You are a critical Analyst Agent. Read the provided research notes, extract main technical claims, "
            "evaluate evidence strength, compare viewpoints, identify gaps, and outline key conclusions."
        )
        user_prompt = f"Target Query: {state.request.query}\nAudience: {state.request.audience}\n\nResearch Notes:\n{research_notes}"

        llm_res = self.llm_client.complete(system_prompt, user_prompt)
        analysis = llm_res.content.strip()

        state.analysis_notes = analysis
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=analysis,
                metadata={"research_notes_length": len(research_notes)},
            )
        )
        state.add_trace_event("analyst_complete", {"analysis_length": len(analysis)})
        return state
