"""Unit tests for multi-agent system components."""

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_quality_score,
)
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_supervisor_routing_policy() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain GraphRAG architecture"))
    supervisor = SupervisorAgent()

    # Step 1: No sources -> Route to researcher
    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    # Step 2: Add research notes -> Route to analyst
    state.sources = [
        SourceDocument(
            title="GraphRAG Paper", snippet="GraphRAG integrates knowledge graphs with RAG."
        )
    ]
    state.research_notes = "GraphRAG uses knowledge graphs."
    state = supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    # Step 3: Add analysis notes -> Route to writer
    state.analysis_notes = "GraphRAG significantly improves complex reasoning."
    state = supervisor.run(state)
    assert state.route_history[-1] == "writer"


def test_workflow_execution() -> None:
    query = ResearchQuery(query="Research GraphRAG state-of-the-art")
    state = ResearchState(request=query)
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    assert result.final_answer is not None
    assert len(result.route_history) > 0
    assert result.sources is not None


def test_citation_coverage_and_quality_metrics() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [
        SourceDocument(
            title="GraphRAG Deep Dive", url="https://arxiv.org/abs/1234.5678", snippet="Snippet"
        )
    ]
    state.research_notes = "Notes"
    state.analysis_notes = "Analysis"
    state.final_answer = "According to [GraphRAG Deep Dive](https://arxiv.org/abs/1234.5678), GraphRAG improves RAG performance."

    coverage = compute_citation_coverage(state)
    quality = compute_quality_score(state)

    assert coverage == 1.0
    assert quality > 7.0
