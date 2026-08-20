"""Supervisor / router implementation for multi-agent workflow."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = AgentName.SUPERVISOR.value

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        settings = get_settings()
        self.llm_client = llm_client or LLMClient(default_model=settings.mistral_supervisor_model)
        self.max_iterations = settings.max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect current state and decide the next step (researcher, analyst, writer, done)."""
        if state.iteration >= self.max_iterations:
            logger.warning(
                "Reached max iterations (%d). Forcing route to writer or done.", self.max_iterations
            )
            next_step = AgentName.WRITER.value if not state.final_answer else "done"
            state.record_route(next_step)
            return state

        # Explicit deterministic routing fallback check
        if not state.sources and not state.research_notes:
            next_step = AgentName.RESEARCHER.value
        elif not state.analysis_notes:
            next_step = AgentName.ANALYST.value
        elif not state.final_answer:
            next_step = AgentName.WRITER.value
        else:
            next_step = "done"

        # Try LLM-assisted decision if LLM client available
        system_prompt = (
            "You are a Supervisor Agent in a Multi-Agent Research System. "
            "Your job is to analyze current research progress and decide the next agent step. "
            "Available next steps: 'researcher', 'analyst', 'writer', 'done'. "
            "Rules:\n"
            "1. If sources/research_notes are missing or insufficient -> 'researcher'\n"
            "2. If research_notes exist but analysis_notes are missing -> 'analyst'\n"
            "3. If analysis_notes exist but final_answer is missing -> 'writer'\n"
            "4. If final_answer is complete -> 'done'\n"
            "Respond ONLY with one of the exact string words: researcher, analyst, writer, done."
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Current iteration: {state.iteration}\n"
            f"Has sources: {len(state.sources) > 0}\n"
            f"Has research notes: {state.research_notes is not None}\n"
            f"Has analysis notes: {state.analysis_notes is not None}\n"
            f"Has final answer: {state.final_answer is not None}\n"
            f"Previous routes: {state.route_history}\n"
            f"Default suggestion: {next_step}"
        )

        try:
            llm_res = self.llm_client.complete(system_prompt, user_prompt)
            decision = llm_res.content.strip().lower()
            for valid in [
                AgentName.RESEARCHER.value,
                AgentName.ANALYST.value,
                AgentName.WRITER.value,
                "done",
            ]:
                if valid in decision:
                    next_step = valid
                    break
        except Exception as exc:
            logger.warning(
                "Supervisor LLM decision failed: %s. Using fallback step: %s", exc, next_step
            )

        state.record_route(next_step)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Routed workflow to step: {next_step}",
                metadata={"iteration": state.iteration, "next_step": next_step},
            )
        )
        state.add_trace_event(
            "supervisor_decision", {"next_step": next_step, "iteration": state.iteration}
        )
        return state
