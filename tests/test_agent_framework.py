"""Tests for pathway agent framework orchestration."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npathway.agents.framework import AgentSpec, PathwayAgentFramework, build_default_pathway_framework


def test_default_framework_execution_order() -> None:
    """Default framework should preserve the intended dependency order."""
    framework = build_default_pathway_framework()
    order = framework.execution_order()

    assert order.index("literature_mining") < order.index("hardcore_developer")
    assert order.index("dataset_preprocessing") < order.index("hardcore_developer")
    assert order.index("hardcore_developer") < order.index("supervisor_engineering")
    assert order.index("hardcore_developer") < order.index("supervisor_publication")
    assert order.index("supervisor_engineering") < order.index("supervisor_publication")


def test_ready_agents_updates_from_completed() -> None:
    """Ready set should change as dependencies become completed."""
    framework = build_default_pathway_framework()

    initial_ready = framework.ready_agents(completed=[])
    assert initial_ready == ["literature_mining", "dataset_preprocessing"]

    after_parallel = framework.ready_agents(completed=["literature_mining", "dataset_preprocessing"])
    assert after_parallel == ["hardcore_developer"]


def test_missing_dependency_is_rejected() -> None:
    """Framework should raise when an agent references unknown dependency."""
    with pytest.raises(ValueError, match="missing dependencies"):
        PathwayAgentFramework(
            [
                AgentSpec(
                    agent_id="a",
                    name="A",
                    objective="A",
                    key_actions=("x",),
                    deliverables=("d",),
                    dependencies=("missing",),
                )
            ]
        )


def test_cycle_is_rejected() -> None:
    """Framework should raise on cyclic dependency graphs."""
    with pytest.raises(ValueError, match="Cycle detected"):
        PathwayAgentFramework(
            [
                AgentSpec(
                    agent_id="a",
                    name="A",
                    objective="A",
                    key_actions=("x",),
                    deliverables=("d",),
                    dependencies=("b",),
                ),
                AgentSpec(
                    agent_id="b",
                    name="B",
                    objective="B",
                    key_actions=("x",),
                    deliverables=("d",),
                    dependencies=("a",),
                ),
            ]
        )


def test_write_outputs(tmp_path) -> None:
    """Framework should write markdown and JSON outputs."""
    framework = build_default_pathway_framework()
    md = tmp_path / "plan.md"
    js = tmp_path / "plan.json"

    md_path, js_path = framework.write_outputs(md, js, generated_on=date(2026, 3, 4))
    assert md_path.exists()
    assert js_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    assert "Pathway Discovery Multi-Agent Framework" in md_text
    assert "Agent 1 - Literature Mining" in md_text

    payload = json.loads(js_path.read_text(encoding="utf-8"))
    assert payload["generated_on"] == "2026-03-04"
    assert len(payload["agents"]) == 5
