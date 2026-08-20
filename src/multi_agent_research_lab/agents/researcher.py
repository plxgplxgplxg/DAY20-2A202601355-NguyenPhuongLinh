"""Researcher agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = AgentName.RESEARCHER.value

    def __init__(
        self, llm_client: LLMClient | None = None, search_client: SearchClient | None = None
    ) -> None:
        settings = get_settings()
        self.llm_client = llm_client or LLMClient(default_model=settings.mistral_researcher_model)
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.sources and state.research_notes."""
        query = state.request.query
        logger.info("Researcher searching for query: %s", query)

        sources = self.search_client.search(query, max_results=state.request.max_sources)
        state.sources = sources

        sources_text = "\n".join(
            [
                f"- Title: {s.title}\n  URL: {s.url or 'N/A'}\n  Snippet: {s.snippet}"
                for s in sources
            ]
        )

        system_prompt = (
            "You are a meticulous Researcher Agent. Given a research query and raw search results, "
            "synthesize comprehensive research notes highlighting key technical facts, findings, and references."
        )
        user_prompt = f"Query: {query}\n\nSources:\n{sources_text}"

        llm_res = self.llm_client.complete(system_prompt, user_prompt)
        notes = llm_res.content.strip()

        state.research_notes = notes
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=notes,
                metadata={"source_count": len(sources)},
            )
        )
        state.add_trace_event("researcher_complete", {"source_count": len(sources)})
        return state
