"""Benchmark implementation for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate citation coverage ratio (percentage of sources cited in final answer)."""
    if not state.sources or not state.final_answer:
        return 0.0

    cited_count = 0
    final_text_lower = state.final_answer.lower()
    for src in state.sources:
        title_words = [w.lower() for w in src.title.split() if len(w) > 3]
        if (
            src.url
            and src.url.lower() in final_text_lower
            or any(w in final_text_lower for w in title_words)
        ):
            cited_count += 1

    return min(1.0, cited_count / len(state.sources))


def compute_quality_score(state: ResearchState) -> float:
    """Calculate synthetic quality score out of 10 based on output completeness."""
    if not state.final_answer:
        return 0.0

    score = 5.0
    if state.sources:
        score += 1.5
    if state.research_notes:
        score += 1.0
    if state.analysis_notes:
        score += 1.5
    if len(state.final_answer) > 300:
        score += 1.0

    return min(10.0, score)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, quality, citation coverage, cost, and failure rate."""

    started = perf_counter()
    failure = 0.0
    try:
        state = runner(query)
    except Exception as exc:
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
        failure = 1.0

    latency = perf_counter() - started
    coverage = compute_citation_coverage(state)
    quality = compute_quality_score(state)
    cost = 0.0  # Mistral Free tier

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=failure,
        notes=f"Iterations: {state.iteration}, Route history: {' -> '.join(state.route_history)}",
    )
    return state, metrics
