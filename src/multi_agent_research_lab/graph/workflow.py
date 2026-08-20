"""Multi-Agent workflow implementation using graph execution pattern."""

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and executes the multi-agent orchestration loop.

    Supported nodes: supervisor, researcher, analyst, writer, critic.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

        self.agents = {
            AgentName.RESEARCHER.value: self.researcher,
            AgentName.ANALYST.value: self.analyst,
            AgentName.WRITER.value: self.writer,
            AgentName.CRITIC.value: self.critic,
        }

    def build(self) -> dict[str, Any]:
        """Return workflow graph representation map."""
        return {
            "nodes": [
                AgentName.SUPERVISOR.value,
                AgentName.RESEARCHER.value,
                AgentName.ANALYST.value,
                AgentName.WRITER.value,
                AgentName.CRITIC.value,
            ],
            "max_iterations": self.settings.max_iterations,
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the multi-agent graph loop until Supervisor outputs 'done' or max iterations reached."""
        with trace_span("multi_agent_workflow", {"query": state.request.query}) as span:
            logger.info("Starting Multi-Agent Workflow execution.")

            while state.iteration < self.settings.max_iterations:
                with trace_span("supervisor_node", {"iteration": state.iteration}):
                    state = self.supervisor.run(state)

                next_step = state.route_history[-1] if state.route_history else "done"
                logger.info("Iteration %d: Supervisor routed to '%s'", state.iteration, next_step)

                if next_step == "done" or next_step not in self.agents:
                    logger.info(
                        "Workflow completed or requested termination at step '%s'.", next_step
                    )
                    break

                worker_agent = self.agents[next_step]
                with trace_span(f"worker_node_{next_step}", {"iteration": state.iteration}):
                    try:
                        state = worker_agent.run(state)
                    except Exception as exc:
                        error_msg = f"Worker agent '{next_step}' failed: {exc}"
                        logger.error(error_msg)
                        state.errors.append(error_msg)
                        break

            # If final answer was written, optional critic pass
            if state.final_answer and AgentName.CRITIC.value not in [
                r for r in state.route_history
            ]:
                with trace_span("critic_node", {"iteration": state.iteration}):
                    state = self.critic.run(state)

            span["attributes"]["total_iterations"] = state.iteration
            span["attributes"]["route_history"] = state.route_history
            return state
