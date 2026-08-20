"""Writer agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes with inline citations."""

    name = AgentName.WRITER.value

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        settings = get_settings()
        self.llm_client = llm_client or LLMClient(default_model=settings.mistral_writer_model)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.final_answer."""
        logger.info("Writer synthesizing final report.")
        sources_summary = "\n".join([f"- [{s.title}]({s.url or 'N/A'})" for s in state.sources])

        system_prompt = (
            "You are a professional Technical Writer Agent. Synthesize research notes and analysis notes "
            "into a well-structured, clear markdown report tailored for the target audience. "
            "IMPORTANT: Include inline citations and reference the provided sources."
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research Notes:\n{state.research_notes or 'N/A'}\n\n"
            f"Analysis Notes:\n{state.analysis_notes or 'N/A'}\n\n"
            f"Sources:\n{sources_summary}"
        )

        llm_res = self.llm_client.complete(system_prompt, user_prompt)
        final_answer = llm_res.content.strip()

        # Ensure source references section exists if not already present
        if "References" not in final_answer and state.sources:
            final_answer += "\n\n### References\n" + sources_summary

        state.final_answer = final_answer
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=final_answer,
                metadata={"final_length": len(final_answer)},
            )
        )
        state.add_trace_event("writer_complete", {"final_length": len(final_answer)})
        return state
