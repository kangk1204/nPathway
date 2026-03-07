"""Planning and orchestration helpers for the pathway agent workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AgentSpec:
    """Single agent definition used in the orchestration plan."""

    agent_id: str
    name: str
    objective: str
    key_actions: tuple[str, ...]
    deliverables: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


class PathwayAgentFramework:
    """Dependency-aware workflow manager for pathway discovery agents."""

    def __init__(self, agents: Iterable[AgentSpec]) -> None:
        self.agents: list[AgentSpec] = list(agents)
        self._agent_index: dict[str, AgentSpec] = {agent.agent_id: agent for agent in self.agents}
        if len(self.agents) != len(self._agent_index):
            raise ValueError("Agent IDs must be unique.")
        self.validate()

    def validate(self) -> None:
        """Validate dependency references and cycle constraints."""
        for agent in self.agents:
            missing = [dep for dep in agent.dependencies if dep not in self._agent_index]
            if missing:
                raise ValueError(
                    f"Agent '{agent.agent_id}' has missing dependencies: {', '.join(missing)}"
                )

        # Trigger cycle detection through topological ordering.
        self.execution_order()

    def execution_order(self) -> list[str]:
        """Return a deterministic topological order for agents."""
        indegree = {agent.agent_id: 0 for agent in self.agents}
        adjacency: dict[str, list[str]] = {agent.agent_id: [] for agent in self.agents}
        insertion_rank = {agent.agent_id: idx for idx, agent in enumerate(self.agents)}

        for agent in self.agents:
            for dep in agent.dependencies:
                indegree[agent.agent_id] += 1
                adjacency[dep].append(agent.agent_id)

        queue = [agent_id for agent_id, deg in indegree.items() if deg == 0]
        queue.sort(key=lambda aid: insertion_rank[aid])

        ordered: list[str] = []
        while queue:
            current = queue.pop(0)
            ordered.append(current)
            for nxt in adjacency[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
                    queue.sort(key=lambda aid: insertion_rank[aid])

        if len(ordered) != len(self.agents):
            raise ValueError("Cycle detected in agent dependencies.")

        return ordered

    def ready_agents(self, completed: Iterable[str]) -> list[str]:
        """Return agents that are ready to run for a set of completed agent IDs."""
        done = set(completed)
        ready: list[str] = []
        for agent_id in self.execution_order():
            if agent_id in done:
                continue
            agent = self._agent_index[agent_id]
            if all(dep in done for dep in agent.dependencies):
                ready.append(agent_id)
        return ready

    def to_records(self) -> list[dict[str, object]]:
        """Return serializable plan records."""
        records: list[dict[str, object]] = []
        for order, agent_id in enumerate(self.execution_order(), start=1):
            agent = self._agent_index[agent_id]
            record = asdict(agent)
            record["order"] = order
            records.append(record)
        return records

    def render_markdown(self, generated_on: date | None = None) -> str:
        """Render a concise markdown execution plan for the framework."""
        stamp = generated_on or date.today()
        lines: list[str] = [
            "# Pathway Discovery Multi-Agent Framework",
            "",
            f"- Generated on: {stamp.isoformat()}",
            f"- Agent count: {len(self.agents)}",
            "",
            "## Execution Order",
        ]

        for agent_id in self.execution_order():
            agent = self._agent_index[agent_id]
            deps = ", ".join(agent.dependencies) if agent.dependencies else "-"
            lines.extend(
                [
                    f"### {agent.name} (`{agent.agent_id}`)",
                    f"- Objective: {agent.objective}",
                    f"- Depends on: {deps}",
                    "- Key actions:",
                ]
            )
            lines.extend([f"  - {action}" for action in agent.key_actions])
            lines.append("- Deliverables:")
            lines.extend([f"  - {artifact}" for artifact in agent.deliverables])
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def write_outputs(
        self,
        markdown_path: str | Path,
        json_path: str | Path,
        generated_on: date | None = None,
    ) -> tuple[Path, Path]:
        """Write markdown and JSON planning outputs."""
        md_path = Path(markdown_path)
        js_path = Path(json_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        js_path.parent.mkdir(parents=True, exist_ok=True)

        markdown = self.render_markdown(generated_on=generated_on)
        md_path.write_text(markdown, encoding="utf-8")

        import json

        payload = {
            "generated_on": (generated_on or date.today()).isoformat(),
            "agents": self.to_records(),
        }
        js_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return md_path, js_path


def build_default_pathway_framework() -> PathwayAgentFramework:
    """Create the default five-agent pathway discovery framework."""
    agents = [
        AgentSpec(
            agent_id="literature_mining",
            name="Agent 1 - Literature Mining",
            objective="Track latest pathway discovery methods and evidence trends.",
            key_actions=(
                "Search PubMed/arXiv/bioRxiv with fixed weekly query sets.",
                "Extract algorithm families, benchmark settings, and failure modes.",
                "Score novelty/reproducibility risks and hand off priorities.",
            ),
            deliverables=(
                "docs/literature/latest_trends_report.md",
                "results/literature/paper_matrix.csv",
                "results/literature/method_priorities.json",
            ),
        ),
        AgentSpec(
            agent_id="dataset_preprocessing",
            name="Agent 2 - Dataset Preprocessing",
            objective="Apply dataset-specific QC and normalization with full provenance.",
            key_actions=(
                "Run protocol-aware QC thresholds per dataset.",
                "Standardize gene identifiers and batch metadata.",
                "Export analysis-ready matrices with reproducible logs.",
            ),
            deliverables=(
                "results/preprocessing/preprocessing_manifest.json",
                "results/preprocessing/qc_summary.csv",
                "results/preprocessing/analysis_ready/",
            ),
        ),
        AgentSpec(
            agent_id="hardcore_developer",
            name="Agent 3 - Hardcore Developer",
            objective="Build and optimize the pathway discovery stack with GPU efficiency.",
            key_actions=(
                "Implement and benchmark modern DL/ML program discovery methods.",
                "Tune GPU memory usage, throughput, and numerical stability.",
                "Package reproducible training/evaluation scripts with tests.",
            ),
            deliverables=(
                "src/npathway/discovery/",
                "results/benchmarks/developer_benchmark_summary.csv",
                "results/benchmarks/gpu_profiling_report.md",
            ),
            dependencies=("literature_mining", "dataset_preprocessing"),
        ),
        AgentSpec(
            agent_id="supervisor_engineering",
            name="Agent 4 - Supervisor (Engineering)",
            objective="Coordinate execution order, quality gates, and integration readiness.",
            key_actions=(
                "Operate stage gates and dependency checks across all agents.",
                "Track blockers, risks, and completion status for each milestone.",
                "Approve releases only when metrics and tests pass.",
            ),
            deliverables=(
                "docs/supervisor/engineering_status_board.md",
                "docs/supervisor/integration_gate_checklist.md",
            ),
            dependencies=("hardcore_developer",),
        ),
        AgentSpec(
            agent_id="supervisor_publication",
            name="Agent 5 - Supervisor (Paper/Figure)",
            objective="Turn validated outputs into manuscript-ready text and figure packs.",
            key_actions=(
                "Draft method/results narratives linked to validated experiments.",
                "Generate figure specs and track source-of-truth data paths.",
                "Prepare submission-safe claim wording and evidence mapping.",
            ),
            deliverables=(
                "paper/main.tex",
                "results/figures/figure_manifest.yaml",
                "docs/supervisor/manuscript_evidence_map.md",
            ),
            dependencies=("hardcore_developer", "supervisor_engineering"),
        ),
    ]
    return PathwayAgentFramework(agents)
