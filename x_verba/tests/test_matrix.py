"""
X-Verba Test Case Matrix v0.4.0

Validates engine.py v0.4.0 outputs and the full pipeline:
    engine.scan() -> cli.py -> writer.py + OutputFormatter

Run from this folder:
    python -m pytest test_matrix.py -v

Test categories:
  1. Engine output structure (graphs / inventories / metrics / algorithms)
  2. Graph structures (AgentGraph, DecisionGraph)
  3. Inventory accuracy (AI, agent, decision)
  4. Metrics validity (coverage, tendency, gamma variants)
  5. Algorithm results (pagerank, critical path, propagation)
  6. End-to-end pipeline (OutputFormatter, OutputWriter, CLI)

Edge cases (empty repo / no AI integrations) are covered separately —
see TestEmptyRepo at the bottom.
"""
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from x_verba.engine import ScanEngine, OutputFormatter, TendencyState
from x_verba.writer import OutputWriter
from x_verba.cli import main


TEST_REPO_PATH = Path(__file__).parent.parent / "_test_sample"
EMPTY_REPO_PATH = Path(__file__).parent.parent / "_test_empty"


@pytest.fixture(scope="module")
def results():
    """Run the full v0.4.0 scan once and share it across tests in this module."""
    return ScanEngine().scan(str(TEST_REPO_PATH))


# ── Category 1: Engine output structure (v0.4.0) ──────────────────────────────

class TestEngineOutputStructure:

    def test_preserved_keys(self, results):
        """v0.3.0 keys must still be present and populated."""
        for key in ("primitives", "summary", "gaps", "drift_classes", "legion_matches"):
            assert key in results

        assert results["files_scanned"] > 0
        assert results["primitives"]["ai_integrations"], "expected AI integrations in _test_sample"

    def test_new_v0_4_0_keys(self, results):
        """v0.4.0 must add graphs, inventories, metrics, algorithms."""
        assert "graphs" in results
        assert "agent_graph" in results["graphs"]
        assert "decision_graph" in results["graphs"]

        assert "inventories" in results
        assert "ai" in results["inventories"]
        assert "agent" in results["inventories"]
        assert "decision" in results["inventories"]

        assert "metrics" in results
        assert "coverage" in results["metrics"]
        assert "tendency" in results["metrics"]
        assert "gamma_variants" in results["metrics"]

        assert "algorithms" in results
        assert "pagerank" in results["algorithms"]
        assert "critical_path" in results["algorithms"]
        assert "propagation" in results["algorithms"]

    def test_summary_folds_in_v0_4_0_metrics(self, results):
        """_build_v0_4_0_summary() must fold the new metrics into results['summary']."""
        summary = results["summary"]
        for key in ("ai_inventory", "agent_inventory", "decision_inventory",
                    "coverage", "tendency", "gamma_variants", "top_decisions"):
            assert key in summary, f"summary missing v0.4.0 key: {key}"


# ── Category 2: Graph structures ──────────────────────────────────────────────

class TestGraphStructures:

    def test_agent_graph_structure(self, results):
        agent_graph = results["graphs"]["agent_graph"]
        assert hasattr(agent_graph, "nodes")
        assert hasattr(agent_graph, "edges")
        assert hasattr(agent_graph, "chains")
        assert hasattr(agent_graph, "clusters")
        assert isinstance(agent_graph.nodes, list)
        assert isinstance(agent_graph.edges, list)

    def test_decision_graph_structure(self, results):
        decision_graph = results["graphs"]["decision_graph"]
        assert hasattr(decision_graph, "nodes")
        assert hasattr(decision_graph, "edges")
        assert len(decision_graph.nodes) > 0, "_test_sample must have decision points"

        for location, node in decision_graph.nodes.items():
            # location is "file:line"
            assert ":" in location
            assert hasattr(node, "criticality")
            assert hasattr(node, "governed")
            # Criticality is a weighted product (impact x blast_radius x
            # irreversibility x (1 + governance_gap)) and can exceed 1.0 for
            # ungoverned high-impact consequences — only non-negativity holds.
            assert node.criticality >= 0.0


# ── Category 3: Inventory accuracy ────────────────────────────────────────────

class TestInventories:

    def test_ai_inventory(self, results):
        ai_inv = results["inventories"]["ai"]
        assert hasattr(ai_inv, "total")
        assert hasattr(ai_inv, "by_provider")
        assert hasattr(ai_inv, "governed")
        assert hasattr(ai_inv, "ungoverned")
        assert ai_inv.total == ai_inv.governed + ai_inv.ungoverned
        assert ai_inv.total == sum(ai_inv.by_provider.values())

    def test_agent_inventory(self, results):
        agent_inv = results["inventories"]["agent"]
        assert hasattr(agent_inv, "total_agents")
        assert hasattr(agent_inv, "total_handovers")
        assert hasattr(agent_inv, "governed_handovers")
        assert hasattr(agent_inv, "total_chains")

        if agent_inv.total_handovers > 0:
            assert agent_inv.governed_handovers <= agent_inv.total_handovers
            assert (agent_inv.governed_handovers + agent_inv.ungoverned_handovers
                    == agent_inv.total_handovers)

        assert (agent_inv.fully_governed_chains
                + agent_inv.partially_governed_chains
                + agent_inv.ungoverned_chains) <= agent_inv.total_chains \
            if agent_inv.total_chains else True

    def test_decision_inventory(self, results):
        dec_inv = results["inventories"]["decision"]
        assert hasattr(dec_inv, "total")
        assert hasattr(dec_inv, "by_consequence_type")
        assert hasattr(dec_inv, "by_criticality")
        assert hasattr(dec_inv, "critical_total")
        assert dec_inv.critical_total <= dec_inv.total
        assert dec_inv.total == len(results["graphs"]["decision_graph"].nodes)


# ── Category 4: Metrics validity ──────────────────────────────────────────────

class TestMetrics:

    def test_governance_coverage(self, results):
        coverage = results["metrics"]["coverage"]
        assert hasattr(coverage, "overall_percent")
        assert 0 <= coverage.overall_percent <= 100
        assert hasattr(coverage, "by_decision_type")
        assert hasattr(coverage, "by_consequence_type")
        assert hasattr(coverage, "critical_coverage")
        assert 0 <= coverage.critical_coverage <= 100

        for pct in coverage.by_decision_type.values():
            assert 0 <= pct <= 100

    def test_tendency_indicators(self, results):
        tendency = results["metrics"]["tendency"]
        assert hasattr(tendency, "state")
        assert isinstance(tendency.state, TendencyState)
        assert tendency.state.value in ["stable", "emerging", "amplifying", "critical", "failure"]

        assert hasattr(tendency, "t_amplification_active")
        assert isinstance(tendency.t_amplification_active, bool)

        assert hasattr(tendency, "pre_node_proximity")
        assert isinstance(tendency.pre_node_proximity, str)

        assert 0.0 <= tendency.score

    def test_gamma_variants(self, results):
        gamma = results["metrics"]["gamma_variants"]
        assert hasattr(gamma, "overall")
        assert hasattr(gamma, "by_decision_type")
        assert hasattr(gamma, "by_consequence_type")
        assert hasattr(gamma, "critical")
        assert hasattr(gamma, "agent_handover")
        assert hasattr(gamma, "agent_chain")
        assert hasattr(gamma, "cluster")

        for gamma_value in (gamma.overall, gamma.critical, gamma.agent_handover,
                             gamma.agent_chain, gamma.cluster):
            assert gamma_value is not None
            assert 0.0 <= gamma_value.value <= 1.0
            assert gamma_value.status in ("ABOVE_THRESHOLD", "PARTIAL_COVERAGE", "BELOW_THRESHOLD")


# ── Category 5: Algorithm results ─────────────────────────────────────────────

class TestAlgorithms:

    def test_pagerank_output(self, results):
        pagerank = results["algorithms"]["pagerank"]
        assert isinstance(pagerank, dict)
        assert pagerank, "_test_sample's decision graph should produce a non-empty pagerank"
        for location, score in pagerank.items():
            assert isinstance(location, str)
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0

    def test_critical_path_output(self, results):
        critical_path = results["algorithms"]["critical_path"]
        assert isinstance(critical_path, list)
        for item in critical_path:
            assert isinstance(item, str)
            assert ":" in item  # file:line format

    def test_propagation_potential_output(self, results):
        propagation = results["algorithms"]["propagation"]
        assert isinstance(propagation, dict)
        for decision, reachable in propagation.items():
            assert isinstance(decision, str)
            assert isinstance(reachable, dict)
            for downstream_node, path in reachable.items():
                assert isinstance(downstream_node, str)
                assert isinstance(path, list)


# ── Category 6: End-to-end pipeline ───────────────────────────────────────────

class TestEndToEnd:

    def test_engine_to_output_formatter_text(self, results):
        formatter = OutputFormatter()
        text_report = formatter.format_report(results, fmt="text")
        assert isinstance(text_report, str)
        assert len(text_report) > 100
        assert "X-VERBA SCAN RESULTS" in text_report
        assert "GOVERNANCE SCORECARD" in text_report

    def test_engine_to_output_formatter_json(self, results):
        formatter = OutputFormatter()
        json_report = formatter.format_report(results, fmt="json")
        assert isinstance(json_report, str)
        parsed = json.loads(json_report)
        assert "summary" in parsed
        assert "graphs" in parsed

    def test_engine_to_writer_yaml(self, results):
        writer = OutputWriter(results, "yaml")
        yaml_content = writer._render_yaml()
        assert isinstance(yaml_content, str)
        assert "X-VERBA GOVERNANCE CONTRACT" in yaml_content
        assert "system_identity" in yaml_content

        # New v0.4.0 reference sections
        assert "SECTION 10 — GOVERNANCE INTELLIGENCE" in yaml_content
        assert "SECTION 11 — AGENT GOVERNANCE" in yaml_content
        assert "SECTION 12 — CRITICAL DECISIONS" in yaml_content

        # Whole file must still be valid, loadable YAML
        data = yaml.safe_load(yaml_content)
        assert "governance_intelligence" in data
        assert "agent_governance" in data
        assert "critical_decisions" in data

    def test_engine_to_writer_json_and_md(self, results):
        # JSON and Markdown renderers must not error on v0.4.0 results
        assert json.loads(OutputWriter(results, "json")._render_json())
        md = OutputWriter(results, "md")._render_markdown()
        assert "# X-Verba Governance Report" in md

    def test_cli_scan_text(self):
        runner = CliRunner()
        result = runner.invoke(main, ["scan", str(TEST_REPO_PATH), "--format", "text",
                                       "--context-profile", "general"])
        assert result.exit_code == 0, result.output
        assert "Files scanned" in result.output
        assert "Tendency" in result.output

    def test_cli_scan_json(self, tmp_path):
        runner = CliRunner()
        out_file = tmp_path / "report.json"
        result = runner.invoke(main, [
            "scan", str(TEST_REPO_PATH),
            "--format", "json",
            "--context-profile", "general",
            "--output", str(out_file),
        ])
        assert result.exit_code == 0, result.output
        assert out_file.exists()
        parsed = json.loads(out_file.read_text(encoding="utf-8"))
        assert "summary" in parsed
        assert "graphs" in parsed


# ── Edge cases: empty repo / no AI integrations ───────────────────────────────

@pytest.fixture(scope="module")
def empty_results():
    if not EMPTY_REPO_PATH.exists():
        pytest.skip(f"{EMPTY_REPO_PATH} does not exist")
    return ScanEngine(context_profile="ai-app").scan(str(EMPTY_REPO_PATH))


class TestEmptyRepo:
    """
    With the default 'ai-app' context profile, a repo with no AI integrations
    still runs the full Pass 3-16 structural governance analysis — the
    'ai-app' profile filters which findings are flagged as AI-adjacent, it
    does not skip analysis. `no_ai_context` records that no AI integrations
    were found, but Gamma, decision points, and v0.4.0 keys are still computed.
    """

    def test_scan_produces_gamma_even_without_ai(self, empty_results):
        summary = empty_results["summary"]
        assert summary["ai_integrations_detected"] == 0
        assert empty_results["no_ai_context"] is True

        # Gamma must always be a float, never None / NO_AI_INTEGRATIONS
        assert summary["structural_gamma"] is not None
        assert isinstance(summary["structural_gamma"], float)
        assert summary["governance_status"] != "NO_AI_INTEGRATIONS"

        # Backwards-compatible keys still present
        for key in ("gaps", "drift_classes", "legion_matches", "decision_points",
                    "agent_handovers", "terminal_states"):
            assert key in empty_results

        # v0.4.0 keys ARE computed — analysis is not skipped for non-AI repos
        for key in ("graphs", "inventories", "metrics", "algorithms"):
            assert key in empty_results

    def test_writer_handles_no_ai_context_results(self, empty_results):
        """OutputWriter must still produce a full contract for a non-AI repo,
        since structural governance analysis ran regardless of AI presence."""
        writer = OutputWriter(empty_results, "yaml")
        yaml_content = writer._render_yaml()
        data = yaml.safe_load(yaml_content)

        assert "system_identity" in data
        assert "scan_summary" in data
        assert "governance_intelligence" in data

    def test_formatter_handles_no_ai_context_results(self, empty_results):
        formatter = OutputFormatter()
        text_report = formatter.format_report(empty_results, fmt="text")
        assert isinstance(text_report, str)
        assert len(text_report) > 0

        json_report = formatter.format_report(empty_results, fmt="json")
        assert json.loads(json_report)


# ── Domain model extraction (VERBA Phase 2) ──────────────────────────────────

class TestDomainModelExtraction:
    """to_dict() of each Phase-2 model must reproduce the exact dict shape
    that engine.py produced before the dataclass promotion (regression gate
    for 'no new behaviour')."""

    def test_pre_node_to_dict(self):
        from x_verba.models import PreNode
        pn = PreNode(type="human_approval", strength=0.75, evidence_line="if approved:")
        d = pn.to_dict()
        assert d == {"type": "human_approval", "strength": 0.75, "evidence_line": "if approved:"}

    def test_terminal_state_to_dict(self):
        from x_verba.models import TerminalState
        ts = TerminalState(
            id="ts-001", type="silent_failure", location="main.py:42",
            severity="critical", plain_english="Exception swallowed silently.",
            consequence="Errors hidden from operators.",
            recommended_action="Log and escalate.",
        )
        d = ts.to_dict()
        assert d["id"] == "ts-001"
        assert d["type"] == "silent_failure"
        assert d["severity"] == "critical"
        assert "extra" not in d

    def test_invariant_to_dict(self):
        from x_verba.models import Invariant
        inv = Invariant(
            location="auth.py:10", type="authorization",
            pattern="role_check", line_content="if not user.is_admin:", near_ai_call=True,
        )
        d = inv.to_dict()
        assert d["location"] == "auth.py:10"
        assert d["near_ai_call"] is True
        assert "extra" not in d

    def test_governance_gap_to_dict_merges_extra(self):
        from x_verba.models import GovernanceGap
        gap = GovernanceGap(
            id="gap-001", type="missing_pre_node", location="main.py:10",
            severity="critical", plain_english="No Pre-Node before AI call.",
            what_is_missing="Pre-Node", consequence="Uncontrolled AI execution.",
            verba_term="PreNode", recommended_action="Add a human approval gate.",
            extra={"verba_explanation": "VE-001", "ai_integration_ref": "openai:main.py:10"},
        )
        d = gap.to_dict()
        # extra fields merged in at top level, no "extra" key
        assert d["verba_explanation"] == "VE-001"
        assert d["ai_integration_ref"] == "openai:main.py:10"
        assert "extra" not in d
        # core fields still present
        assert d["id"] == "gap-001"
        assert d["type"] == "missing_pre_node"

    def test_governance_gap_no_extra(self):
        from x_verba.models import GovernanceGap
        gap = GovernanceGap(
            id="gap-002", type="informal_invariant", location="api.py:5",
            severity="medium", plain_english="Informal check.",
            what_is_missing="Formal Invariant", consequence="Bypassed.",
            verba_term="Invariant", recommended_action="Formalise.",
        )
        d = gap.to_dict()
        assert "extra" not in d
        assert d["id"] == "gap-002"

    def test_real_scan_constraints_have_invariant_shape(self, results):
        """engine.py's _detect_constraints now produces Invariant.to_dict() dicts."""
        constraints = results.get("primitives", {}).get("constraints", [])
        if not constraints:
            pytest.skip("No constraints in _test_sample — skip shape check")
        c = constraints[0]
        for key in ("location", "type", "pattern", "line_content", "near_ai_call"):
            assert key in c, f"Invariant key '{key}' missing from constraints[0]"

    def test_real_scan_terminal_states_have_shape(self, results):
        """engine.py's _detect_terminal_states now produces TerminalState.to_dict() dicts."""
        ts = results.get("terminal_states", [])
        if not ts:
            pytest.skip("No terminal_states in _test_sample — skip shape check")
        t = ts[0]
        for key in ("id", "type", "location", "severity", "plain_english",
                    "consequence", "recommended_action"):
            assert key in t, f"TerminalState key '{key}' missing from terminal_states[0]"

    def test_real_scan_gaps_have_no_extra_key(self, results):
        """GovernanceGap.to_dict() must not expose the internal 'extra' dict."""
        gaps = results.get("gaps", [])
        for gap in gaps:
            assert "extra" not in gap, f"'extra' key leaked into gap {gap.get('id')}"


# ── Baseline storage (VERBA Phase 3) ─────────────────────────────────────────

class TestBaseline:
    """BaselineStore save/load/archive round-trips, using the real scan results."""

    def test_save_and_load_round_trip(self, results, tmp_path):
        from x_verba.baseline import BaselineStore
        store = BaselineStore(tmp_path)
        saved = store.save(results)
        assert saved.exists()

        loaded = store.load()
        # Canonical Gamma must survive JSON round-trip
        b_gamma = loaded["summary"]["gamma_variants"]["overall"]
        r_gamma = results["summary"]["gamma_variants"]["overall"]
        assert abs(b_gamma["value"] - r_gamma["value"]) < 1e-9
        assert b_gamma["status"] == r_gamma["status"]

    def test_load_explicit_path(self, results, tmp_path):
        from x_verba.baseline import BaselineStore
        store = BaselineStore(tmp_path)
        saved_path = store.save(results)
        loaded = store.load(saved_path)
        assert "summary" in loaded

    def test_load_raises_when_no_baseline(self, tmp_path):
        from x_verba.baseline import BaselineStore, BaselineNotFoundError
        store = BaselineStore(tmp_path / "nonexistent")
        with pytest.raises(BaselineNotFoundError, match="No governance baseline"):
            store.load()

    def test_archive_sequential_numbering(self, results, tmp_path):
        from x_verba.baseline import BaselineStore
        store = BaselineStore(tmp_path)
        a1 = store.archive(results)
        a2 = store.archive(results)
        a3 = store.archive(results)
        assert a1.name == "scan-001.json"
        assert a2.name == "scan-002.json"
        assert a3.name == "scan-003.json"
        for p in (a1, a2, a3):
            assert p.exists()

    def test_save_produces_valid_json(self, results, tmp_path):
        from x_verba.baseline import BaselineStore
        store = BaselineStore(tmp_path)
        store.save(results)
        text = store.baseline_path.read_text(encoding="utf-8")
        loaded = json.loads(text)
        assert isinstance(loaded, dict)
        assert "summary" in loaded


# ── Governance Verification (VERBA Phase 4) ───────────────────────────────────

class TestVerification:
    """GovernanceVerificationEngine.compare() coverage for every comparison row
    plus overall_status/has_critical_regressions/passed semantics."""

    def test_stable_same_vs_same(self, results):
        from x_verba.engine import OutputFormatter
        from x_verba.qa_engine import GovernanceVerificationEngine
        safe = OutputFormatter._json_safe(results)
        vr = GovernanceVerificationEngine().compare(safe, safe)
        assert vr.overall_status == "STABLE"
        assert vr.passed is True
        assert vr.has_critical_regressions is False

    def test_regressed_critical_findings_increase(self, results):
        import copy
        from x_verba.engine import OutputFormatter
        from x_verba.qa_engine import GovernanceVerificationEngine
        baseline = OutputFormatter._json_safe(results)
        current = copy.deepcopy(baseline)
        current["summary"]["critical"] = baseline["summary"]["critical"] + 1
        vr = GovernanceVerificationEngine().compare(baseline, current)
        assert vr.overall_status == "REGRESSED"
        assert vr.has_critical_regressions is True
        assert vr.passed is False

    def test_improved_critical_findings_decrease(self, results):
        import copy
        from x_verba.engine import OutputFormatter
        from x_verba.qa_engine import GovernanceVerificationEngine
        baseline = OutputFormatter._json_safe(results)
        current = copy.deepcopy(baseline)
        # increase baseline critical so current looks lower
        baseline["summary"]["critical"] = (baseline["summary"]["critical"] or 0) + 2
        vr = GovernanceVerificationEngine().compare(baseline, current)
        assert vr.overall_status == "IMPROVED"
        assert vr.passed is True

    def test_gamma_status_downgrade_is_critical(self, results):
        import copy
        from x_verba.engine import OutputFormatter
        from x_verba.qa_engine import GovernanceVerificationEngine
        baseline = OutputFormatter._json_safe(results)
        current = copy.deepcopy(baseline)
        # force baseline to ABOVE_THRESHOLD so current looks like a downgrade
        baseline["summary"]["gamma_variants"]["overall"]["status"] = "ABOVE_THRESHOLD"
        baseline["summary"]["gamma_variants"]["overall"]["value"] = 0.95
        current["summary"]["gamma_variants"]["overall"]["status"] = "BELOW_THRESHOLD"
        current["summary"]["gamma_variants"]["overall"]["value"] = 0.30
        current["summary"]["critical"] = baseline["summary"]["critical"]  # no other change
        vr = GovernanceVerificationEngine().compare(baseline, current)
        # status downgrade is CRITICAL severity REGRESSED direction
        gamma_delta = next(d for d in vr.deltas if d.metric == "structural_gamma")
        from x_verba.models import Severity, DeltaDirection
        assert gamma_delta.severity == Severity.CRITICAL
        assert gamma_delta.direction == DeltaDirection.REGRESSED
        assert vr.has_critical_regressions is True

    def test_new_ai_provider_is_medium_new(self, results):
        import copy
        from x_verba.engine import OutputFormatter
        from x_verba.qa_engine import GovernanceVerificationEngine
        baseline = OutputFormatter._json_safe(results)
        current = copy.deepcopy(baseline)
        current["summary"]["ai_inventory"]["by_provider"]["novel_provider"] = 2
        vr = GovernanceVerificationEngine().compare(baseline, current)
        from x_verba.models import Severity, DeltaDirection
        new_deltas = [d for d in vr.deltas if d.direction == DeltaDirection.NEW
                      and "novel_provider" in d.metric]
        assert len(new_deltas) == 1
        assert new_deltas[0].severity == Severity.MEDIUM

    def test_verification_result_to_dict_shape(self, results):
        from x_verba.engine import OutputFormatter
        from x_verba.qa_engine import GovernanceVerificationEngine
        safe = OutputFormatter._json_safe(results)
        vr = GovernanceVerificationEngine().compare(safe, safe)
        d = vr.to_dict()
        for key in ("deltas", "overall_status", "passed", "has_critical_regressions",
                    "regressions", "improvements"):
            assert key in d
        assert isinstance(d["deltas"], list)
        assert isinstance(d["regressions"], list)
        assert isinstance(d["improvements"], list)


# ── End-to-end: scan --save-baseline / scan --compare ─────────────────────────

class TestCLIVerificationFlow:
    """CliRunner integration: save-baseline → compare → STABLE."""

    def test_scan_save_baseline_then_compare(self, tmp_path):
        runner = CliRunner()

        # Step 1: scan and save baseline
        result = runner.invoke(main, [
            "scan", str(TEST_REPO_PATH),
            "--save-baseline",
            "--output", str(tmp_path / "report.txt"),
        ])
        assert result.exit_code == 0, result.output
        baseline_file = TEST_REPO_PATH / ".verba" / "governance-baseline.json"
        assert baseline_file.exists()

        # Step 2: compare against the same baseline — should be STABLE
        result = runner.invoke(main, [
            "scan", str(TEST_REPO_PATH),
            "--compare", str(baseline_file),
            "--output", str(tmp_path / "report2.txt"),
        ])
        assert result.exit_code == 0, result.output
        assert "STABLE" in result.output or "No governance changes" in result.output


class TestDriftClassification:
    """DC/Legion detection and QA recommendation tests."""

    def test_legion_matches_key_present(self, results):
        """legion_matches key is always present in scan results."""
        assert "legion_matches" in results
        assert isinstance(results["legion_matches"], list)

    def test_tier1_confidence_labels_are_valid(self, results):
        """Every Tier 1 legion match has confidence HIGH or MEDIUM."""
        for match in results.get("legion_matches", []):
            conf = match.get("confidence")
            assert conf in ("HIGH", "MEDIUM", "SPECULATIVE"), (
                f"Unexpected confidence value: {conf!r} in {match}"
            )

    def test_tier2_speculative_labeled_correctly(self, results):
        """SPECULATIVE matches have a matched_pattern field."""
        speculative = [
            m for m in results.get("legion_matches", [])
            if m.get("confidence") == "SPECULATIVE"
        ]
        for match in speculative:
            assert "matched_pattern" in match, (
                f"SPECULATIVE match missing matched_pattern: {match}"
            )

    def test_dc_qa_recommendations_from_legion_matches(self):
        """_dc_qa_recommendations returns correct test cases for known DC codes."""
        from x_verba.qa_engine import _dc_qa_recommendations
        matches = [
            {
                "dc_code": "DC-E13",
                "legion_code": "L1",
                "legion_name": "Agent Handover No Pre-Node",
                "confidence": "HIGH",
                "location": "test_file.py:42",
            },
            {
                "dc_code": "DC-I11",
                "legion_code": "L3",
                "legion_name": "Metric Saturation",
                "confidence": "MEDIUM",
                "location": "test_file.py:88",
            },
        ]
        recs = _dc_qa_recommendations(matches)
        assert len(recs) == 2
        dc_codes = {r["dc_code"] for r in recs}
        assert "DC-E13" in dc_codes
        assert "DC-I11" in dc_codes
        for rec in recs:
            assert rec["confidence"] in ("HIGH", "MEDIUM")
            assert len(rec["tests"]) >= 1
            assert "test" in rec["tests"][0]
            assert "description" in rec["tests"][0]

    def test_dc_qa_recommendations_speculative_no_tests(self):
        """SPECULATIVE matches produce a note but no specific test cases."""
        from x_verba.qa_engine import _dc_qa_recommendations
        matches = [
            {
                "dc_code": "DC-X99",
                "legion_code": "L1",
                "legion_name": "Unknown Legion",
                "confidence": "SPECULATIVE",
                "matched_pattern": "some_pattern",
                "location": "test_file.py:10",
            }
        ]
        recs = _dc_qa_recommendations(matches)
        assert len(recs) == 1
        assert recs[0]["confidence"] == "SPECULATIVE"
        assert "note" in recs[0]
        assert recs[0]["tests"] == []


class TestGovernanceNodes:
    """Writer Section 3 — Candidate Governance Node inference and YAML output."""

    def test_infer_embed_keyword_is_high(self):
        from x_verba.writer import _infer_candidate_node
        ai = {"line_content": "service.generate_raw_embeddings(texts)", "provider": "openai", "output_destination": None}
        name, conf = _infer_candidate_node(ai)
        assert name == "EMBEDDINGS_GENERATED"
        assert conf == "HIGH"

    def test_infer_classify_keyword_is_high(self):
        from x_verba.writer import _infer_candidate_node
        ai = {"line_content": "model.classify(input_text)", "provider": "anthropic", "output_destination": None}
        name, conf = _infer_candidate_node(ai)
        assert name == "CONTENT_CLASSIFIED"
        assert conf == "HIGH"

    def test_infer_known_provider_no_keyword_is_medium(self):
        from x_verba.writer import _infer_candidate_node
        ai = {"line_content": "client.chat.completions.create(messages=messages)", "provider": "openai", "output_destination": None}
        name, conf = _infer_candidate_node(ai)
        assert name == "AI_RESPONSE_GENERATED"
        assert conf == "MEDIUM"

    def test_infer_user_destination_is_medium(self):
        from x_verba.writer import _infer_candidate_node
        ai = {"line_content": "model.run(prompt)", "provider": "unknown_provider", "output_destination": "user_response"}
        name, conf = _infer_candidate_node(ai)
        assert name == "AI_RESPONSE_GENERATED"
        assert conf == "MEDIUM"

    def test_infer_no_signal_is_low(self):
        from x_verba.writer import _infer_candidate_node
        ai = {"line_content": "custom_model.process(data)", "provider": "custom", "output_destination": None}
        name, conf = _infer_candidate_node(ai)
        assert name == "AI_OPERATION_EXECUTED"
        assert conf == "LOW"

    def test_nodes_yaml_uses_node_ids_not_ai_ids(self, results):
        """Rendered YAML must use NODE-001 keys, not AI-001."""
        writer = OutputWriter(results, "yaml")
        parsed = yaml.safe_load(writer._render_yaml())
        nodes = parsed.get("nodes", [])
        if not nodes or "note" in nodes[0]:
            pytest.skip("No nodes in _test_sample")
        first_key = list(nodes[0].keys())[0]
        assert first_key.startswith("NODE-"), f"Expected NODE-XXX key, got {first_key!r}"

    def test_nodes_yaml_has_candidate_node_fields(self, results):
        """Every node entry must have candidate_node, inference_confidence, confirmed_node_name."""
        writer = OutputWriter(results, "yaml")
        parsed = yaml.safe_load(writer._render_yaml())
        nodes = parsed.get("nodes", [])
        if not nodes or "note" in nodes[0]:
            pytest.skip("No nodes in _test_sample")
        first_node = list(nodes[0].values())[0]
        for field in ("candidate_node", "inference_confidence", "implementation",
                      "trigger", "confirmed_node_name"):
            assert field in first_node, f"Node missing expected field: {field!r}"
        assert first_node["confirmed_node_name"] is None

    def test_pre_nodes_reference_node_ids(self, results):
        """Pre-Node keys must be PN-NODE-XXX and node_ref must point to NODE-XXX."""
        writer = OutputWriter(results, "yaml")
        parsed = yaml.safe_load(writer._render_yaml())
        pre_nodes = parsed.get("pre_nodes", [])
        if not pre_nodes or "note" in pre_nodes[0]:
            pytest.skip("No pre_nodes in _test_sample")
        first_key = list(pre_nodes[0].keys())[0]
        assert first_key.startswith("PN-NODE-"), f"Expected PN-NODE-XXX, got {first_key!r}"
        first_pn = list(pre_nodes[0].values())[0]
        assert first_pn["node_ref"].startswith("NODE-"), \
            f"node_ref should be NODE-XXX, got {first_pn['node_ref']!r}"

    def test_section_3_header_says_governance_nodes(self, results):
        """Section 3 must say GOVERNANCE NODES and must not contain the old framing."""
        writer = OutputWriter(results, "yaml")
        yaml_content = writer._render_yaml()
        assert "SECTION 3 — GOVERNANCE NODES" in yaml_content
        assert "Every AI call is a Node" not in yaml_content


class TestLegionSchema:
    """Legion operational semantics — schema, evidence layer, determinism, lifecycle."""

    def test_legion_schema_serializable(self):
        """Legion.to_dict() preserves all legacy keys and adds new schema fields."""
        import json
        from x_verba.models import Legion

        legion = Legion(
            id="abc1234567890abc",
            dc_code="DC-E13",
            dc_name="Agent Cascade",
            legion_code="L1",
            legion_name="Ungated Agent Handover",
            description="Agent passes output without a governing Pre-Node.",
            detection_method="structural_pattern",
            confidence_float=0.9,
            confidence="HIGH",
            evidence_type="call_graph",
            file_path="agents/orchestrator.py",
            line_number=42,
            location="agents/orchestrator.py:42",
            code_context="agent_b.run(agent_a.output)",
            observability_level="STRUCTURAL",
            canonical_hash="abc1234567890abc",
        )
        d = legion.to_dict()

        # All legacy keys must be present (backward compat)
        for key in (
            "dc_code", "dc_name", "tier", "legion_code", "legion_name",
            "location", "confidence", "evidence", "matched_pattern",
            "heuristic_description", "primary_so",
        ):
            assert key in d, f"Legacy key missing: {key!r}"

        # New schema fields must be present
        for key in (
            "id", "canonical_hash", "confidence_float", "detection_method",
            "evidence_type", "observability_level", "version",
            "false_positive_conditions", "false_negative_conditions",
        ):
            assert key in d, f"New schema key missing: {key!r}"

        # Values round-trip through JSON without error
        json.dumps(d)

        # Semantic checks
        assert d["confidence"] == "HIGH"
        assert d["confidence_float"] == 0.9
        assert d["detection_method"] == "structural_pattern"
        assert d["evidence"] == "agent_b.run(agent_a.output)"  # maps code_context → evidence

    def test_evidence_extraction_deterministic(self):
        """_extract_evidence_nodes returns identical canonical_hashes for identical input."""
        from x_verba.engine import _extract_evidence_nodes

        primitives = {
            "decision_points": [
                {"location": "foo.py:10", "type": "conditional_branch",
                 "condition": "if score > 0.8", "call": ""},
                {"location": "foo.py:20", "type": "function_call",
                 "condition": "", "call": "validate(user_input)"},
            ],
            "agent_handovers": [
                {"location": "agents.py:55", "from_agent": "PlannerAgent",
                 "to_agent": "ExecutorAgent", "pre_node_exists": False,
                 "governance_gap": "no pre-node"},
            ],
            "ai_integrations": [
                {"location": "llm.py:30", "provider": "openai",
                 "line_content": "client.chat.completions.create()",
                 "pre_node_detected": False},
            ],
        }

        ev1 = _extract_evidence_nodes(primitives)
        ev2 = _extract_evidence_nodes(primitives)

        assert len(ev1) == len(ev2) == 4
        assert [e.canonical_hash for e in ev1] == [e.canonical_hash for e in ev2]

        # Evidence types are correctly assigned
        types = [e.type for e in ev1]
        assert "cfg_node" in types
        assert "call_edge" in types
        assert "ast_pattern" in types

    def test_legion_canonical_hash_deterministic(self):
        """_compute_canonical_hash: identical inputs → identical output, always."""
        from x_verba.engine import _compute_canonical_hash

        h1 = _compute_canonical_hash("agents/foo.py", 42, "DC-E13", "L1", "agent_handover_no_prenode")
        h2 = _compute_canonical_hash("agents/foo.py", 42, "DC-E13", "L1", "agent_handover_no_prenode")
        assert h1 == h2, "Same inputs must produce the same hash"
        assert len(h1) == 16, "Hash should be 16 hex characters"

        # Different file → different hash
        h3 = _compute_canonical_hash("other/bar.py", 42, "DC-E13", "L1", "agent_handover_no_prenode")
        assert h1 != h3

        # Different line → different hash
        h4 = _compute_canonical_hash("agents/foo.py", 99, "DC-E13", "L1", "agent_handover_no_prenode")
        assert h1 != h4

        # Different DC → different hash
        h5 = _compute_canonical_hash("agents/foo.py", 42, "DC-I11", "L1", "agent_handover_no_prenode")
        assert h1 != h5

    def test_dedup_keeps_highest_confidence(self):
        """_dedup_legions retains the highest-confidence Legion per canonical_hash."""
        from x_verba.engine import _dedup_legions
        from x_verba.models import Legion

        shared_hash = "dedup_test_1234abcd"

        low = Legion(
            id=shared_hash, dc_code="DC-I11", dc_name="Evaluative Decoupling",
            legion_code="L3", legion_name="Metric Saturation", description="",
            detection_method="keyword_heuristic", confidence_float=0.3,
            confidence="SPECULATIVE", evidence_type="code_pattern",
            file_path="svc.py", line_number=5, location="svc.py:5",
            code_context="confidence_score > 0.9", observability_level="BEHAVIOURAL",
            canonical_hash=shared_hash,
        )
        high = Legion(
            id=shared_hash, dc_code="DC-I11", dc_name="Evaluative Decoupling",
            legion_code="L3", legion_name="Metric Saturation", description="",
            detection_method="structural_pattern", confidence_float=0.9,
            confidence="HIGH", evidence_type="cfg_node",
            file_path="svc.py", line_number=5, location="svc.py:5",
            code_context="confidence_score > 0.9", observability_level="STRUCTURAL",
            canonical_hash=shared_hash,
        )

        # Order: low first, then high — high should win
        result = _dedup_legions([low, high])
        assert len(result) == 1
        assert result[0].confidence_float == 0.9
        assert result[0].confidence == "HIGH"

        # Order: high first, then low — high should still win
        result2 = _dedup_legions([high, low])
        assert len(result2) == 1
        assert result2[0].confidence_float == 0.9

        # Two distinct hashes → both kept
        low2 = Legion(
            id="other_hash_5678", dc_code="DC-E13", dc_name="Agent Cascade",
            legion_code="L1", legion_name="", description="",
            detection_method="structural_pattern", confidence_float=0.9,
            confidence="HIGH", evidence_type="call_graph",
            file_path="other.py", line_number=1, location="other.py:1",
            code_context="", observability_level="STRUCTURAL",
            canonical_hash="other_hash_5678",
        )
        result3 = _dedup_legions([low, low2])
        assert len(result3) == 2


# ── Framework-detection precision: LangChain / LangGraph / OpenAI / Anthropic ──
#
# Regression coverage for the false-positive and coverage-gap fixes made to
# engine.py's AI_CALL_METHODS / AGENT_FRAMEWORK_PATTERNS / JS_AI_PATTERNS
# matching. Each "false positive" case is real, confirmed code that isn't
# actually LangChain/LangGraph/OpenAI/Anthropic but shares a call shape with
# it; each "true positive" case is genuine framework usage that must keep
# working once the false-positive fix is in place.

class TestFrameworkDetectionPrecision:

    def _scan(self, tmp_path, filename: str, content: str) -> dict:
        (tmp_path / filename).write_text(content, encoding="utf-8")
        return ScanEngine().scan(str(tmp_path))

    def _providers(self, results: dict) -> list:
        return [a["provider"] for a in results["primitives"]["ai_integrations"]]

    def _agent_frameworks(self, results: dict) -> list:
        return [
            d["framework"] for d in results["primitives"]["decision_points"]
            if d.get("type") == "agent_invocation"
        ]

    # ── LangChain — Python ─────────────────────────────────────────────

    def test_supply_chain_run_pipeline_not_langchain(self, tmp_path):
        """A variable named `supply_chain` calling .run_pipeline() must not
        be misread as LangChain's chain.run — confirmed false positive from
        raw substring matching (fixed via anchored _call_matches_pattern)."""
        results = self._scan(tmp_path, "sample.py", '''
class SupplyChain:
    def run_pipeline(self):
        return "done"

supply_chain = SupplyChain()
supply_chain.run_pipeline()
''')
        assert "langchain" not in self._providers(results)
        assert "langchain" not in self._agent_frameworks(results)

    def test_insurance_agent_not_langchain(self, tmp_path):
        """A domain 'agent'/'chain' object (.run()/.invoke()) with no
        LangChain import anywhere in the file must not be flagged —
        confirmed false positive on ordinary insurance-claims code."""
        results = self._scan(tmp_path, "insurance_agent.py", '''
class InsuranceAgent:
    def run(self, policy_id):
        return f"processed {policy_id}"

class ApprovalChain:
    def invoke(self, request):
        return "approved"

def process_claim():
    agent = InsuranceAgent()
    agent.run("POL-123")
    chain = ApprovalChain()
    chain.invoke({"amount": 500})
''')
        assert "langchain" not in self._providers(results)
        assert "langchain" not in self._agent_frameworks(results)

    def test_real_langchain_still_detected(self, tmp_path):
        """Genuine LangChain usage (AgentExecutor + agent.run/chain.invoke,
        corroborated by a real langchain import) must still be detected —
        guards against over-tightening the false-positive fixes above."""
        results = self._scan(tmp_path, "real_langchain.py", '''
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI

def run_assistant():
    llm = ChatOpenAI()
    agent = AgentExecutor(agent=llm, tools=[])
    agent.run("What's the weather?")
    chain = build_chain()
    chain.invoke({"input": "hello"})
''')
        assert "langchain" in self._providers(results)
        assert "langchain" in self._agent_frameworks(results)

    # ── LangGraph — Python ─────────────────────────────────────────────

    def test_networkx_not_langgraph(self, tmp_path):
        """NetworkX's own add_node()/add_edge() API must not be misread as
        LangGraph — confirmed false positive, same method names, unrelated
        general-purpose graph library."""
        results = self._scan(tmp_path, "networkx_sample.py", '''
import networkx as nx

def build_dependency_graph():
    graph = nx.DiGraph()
    graph.add_node("task_a")
    graph.add_node("task_b")
    graph.add_edge("task_a", "task_b")
    return graph
''')
        assert "langgraph" not in self._agent_frameworks(results)

    def test_real_langgraph_still_detected(self, tmp_path):
        """Genuine LangGraph usage (StateGraph import present) must still
        be detected."""
        results = self._scan(tmp_path, "langgraph_sample.py", '''
from langgraph.graph import StateGraph

def build_agent_graph():
    graph = StateGraph(dict)
    graph.add_node("planner", planner_fn)
    graph.add_node("executor", executor_fn)
    graph.add_edge("planner", "executor")
    return graph.compile()
''')
        frameworks = self._agent_frameworks(results)
        assert frameworks.count("langgraph") == 4  # StateGraph + 2 add_node + add_edge

    def test_langgraph_prebuilt_react_agent_detected(self, tmp_path):
        """LangGraph's high-level `create_react_agent` (langgraph.prebuilt)
        must be detected as langgraph — a real, popular usage pattern that
        never touches StateGraph/add_node/add_edge directly. Confirmed
        real-world miss: braincrew-lab/langgraph-mcp-agents uses only this
        API and had zero langgraph decision points before this pattern was
        added, despite the repo name and real LangGraph usage throughout."""
        results = self._scan(tmp_path, "react_agent_sample.py", '''
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

def build_agent():
    model = ChatOpenAI()
    agent = create_react_agent(model, tools=[])
    return agent
''')
        assert "langgraph" in self._agent_frameworks(results)

    def test_langgraph_swarm_detected(self, tmp_path):
        """`create_swarm`/`create_handoff_tool` (the official
        langgraph-ai/langgraph-swarm-py package's top-level API) must be
        detected as langgraph. Confirmed real-world miss: the package's
        own documented usage example — a downstream caller building a
        swarm of agents with handoff tools, never touching
        StateGraph/add_node directly — produced zero langgraph findings
        before these patterns were added."""
        results = self._scan(tmp_path, "swarm_sample.py", '''
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool, create_swarm

alice = create_agent(
    "openai:gpt-4o",
    tools=[create_handoff_tool(agent_name="Bob", description="Transfer to Bob")],
    name="Alice",
)
bob = create_agent(
    "openai:gpt-4o",
    tools=[create_handoff_tool(agent_name="Alice", description="Transfer to Alice")],
    name="Bob",
)

workflow = create_swarm([alice, bob], default_active_agent="Alice")
''')
        frameworks = self._agent_frameworks(results)
        assert frameworks.count("langgraph") == 3  # create_swarm + 2 create_handoff_tool

    # ── LangChain / LangGraph — JS/TS ───────────────────────────────────

    def test_langgraphjs_camelcase_detected(self, tmp_path):
        """LangGraph.js's real (camelCase) API — addNode()/addEdge() — must
        be detected, not just the Python snake_case names. Was a total
        coverage gap: only the StateGraph import itself was ever caught."""
        results = self._scan(tmp_path, "langgraph.ts", '''
import { StateGraph } from "@langchain/langgraph";

const graph = new StateGraph({ channels: {} });
graph.addNode("planner", plannerFn);
graph.addNode("executor", executorFn);
graph.addEdge("planner", "executor");
export default graph.compile();
''')
        frameworks = self._agent_frameworks(results)
        assert frameworks.count("langgraph") == 4  # StateGraph + 2 addNode + addEdge

    def test_travel_agent_ts_not_langchain(self, tmp_path):
        """A plain TS class with .invoke() methods (no LangChain import)
        must not be flagged — confirmed false positive, same class of bug
        as the Python agent/chain cases above."""
        results = self._scan(tmp_path, "travel_agent.ts", '''
class TravelAgent {
  invoke(request: string) {
    return "booked";
  }
}

class ApprovalChain {
  invoke(req: object) {
    return "approved";
  }
}

function process() {
  const agent = new TravelAgent();
  agent.invoke("flight to NYC");
  const chain = new ApprovalChain();
  chain.invoke({ amount: 500 });
}
''')
        assert "langchain" not in self._agent_frameworks(results)
        assert "langchain" not in self._providers(results)

    def test_real_langchainjs_still_detected(self, tmp_path):
        """Genuine LangChain.js usage (AgentExecutor, corroborated by a
        real langchain/@langchain import) must still be detected."""
        results = self._scan(tmp_path, "real_langchainjs.ts", '''
import { AgentExecutor } from "langchain/agents";
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI();
const agent = new AgentExecutor({ agent: llm, tools: [] });
agent.invoke({ input: "hello" });
const chain = buildChain();
chain.invoke({ input: "hello" });
''')
        assert "langchain" in self._agent_frameworks(results)
        assert "langchain" in self._providers(results)

    def test_langchainjs_universal_loader_detected(self, tmp_path):
        """LangChain.js's `initChatModel` (the 'universal model loader',
        `langchain/chat_models/universal`) must be detected as a LangChain
        AI integration. Confirmed real-world miss: a production
        LangGraph.js repo (mayooear/ai-pdf-chatbot-langchain) routes every
        LLM call through this exact function — dynamically instantiating
        whichever provider is named at runtime, with no `new ChatOpenAI(...)`
        call site to match — and had zero ai_integrations findings before
        this pattern was added, despite making real LLM calls throughout."""
        results = self._scan(tmp_path, "load_model.ts", '''
import { initChatModel } from "langchain/chat_models/universal";

export async function loadChatModel(name: string) {
  return await initChatModel(name, { temperature: 0.2 });
}
''')
        assert "langchain" in self._providers(results)

    # ── OpenAI / Anthropic — Python ──────────────────────────────────────

    def test_twilio_messages_create_not_ai(self, tmp_path):
        """Twilio's SMS SDK (client.messages.create(...)) must not be
        flagged as an AI integration — confirmed false positive, identical
        call shape to Anthropic's messages.create with zero AI involvement."""
        results = self._scan(tmp_path, "twilio_sms.py", '''
from twilio.rest import Client

def send_sms():
    client = Client("sid", "token")
    return client.messages.create(
        body="Your order shipped!", from_="+15551234567", to="+15559876543",
    )
''')
        assert self._providers(results) == []

    def test_helpdesk_chat_not_openai(self, tmp_path):
        """A helpdesk client's .chat() method (no openai import) must not
        be flagged as OpenAI."""
        results = self._scan(tmp_path, "helpdesk_bot.py", '''
class HelpdeskClient:
    def chat(self, ticket_id):
        return "responded"

def handle_ticket():
    client = HelpdeskClient()
    client.chat("TICKET-42")
''')
        assert self._providers(results) == []

    def test_real_openai_bare_chat_still_detected(self, tmp_path):
        """A bare client.chat() call, corroborated by a real openai import,
        must still be detected."""
        results = self._scan(tmp_path, "real_openai_bare.py", '''
import openai

def ask(client):
    return client.chat(messages=[])
''')
        assert len(self._providers(results)) == 1

    def test_real_anthropic_bare_messages_create_still_detected(self, tmp_path):
        """A bare client.messages.create() call, corroborated by a real
        anthropic import, must still be detected."""
        results = self._scan(tmp_path, "real_anthropic_bare.py", '''
import anthropic

def ask(anthro_client):
    return anthro_client.messages.create(model="claude-3", messages=[])
''')
        assert len(self._providers(results)) == 1

    def test_legacy_and_modern_openai_still_detected(self, tmp_path):
        """Both legacy (openai.ChatCompletion.create) and modern
        (client.chat.completions.create) OpenAI SDK usage must keep
        working — guards against the anchored-matching fix accidentally
        breaking real detection."""
        results = self._scan(tmp_path, "openai_sample.py", '''
import openai
from openai import OpenAI

def ask_legacy():
    return openai.ChatCompletion.create(model="gpt-4", messages=[])

client = OpenAI()

def ask_modern():
    return client.chat.completions.create(model="gpt-4", messages=[])
''')
        providers = self._providers(results)
        # 3 findings: openai.ChatCompletion.create, the OpenAI() constructor
        # call itself, and client.chat.completions.create.
        assert providers.count("openai") == 3


# ── --all-frameworks scope flag ───────────────────────────────────────────

class TestFrameworkScopeFlag:

    def _scan_mixed_repo(self, tmp_path, all_frameworks: bool) -> dict:
        (tmp_path / "crewai_sample.py").write_text('''
from crewai import Crew, Agent, Task

def build_crew():
    researcher = Agent(role="Researcher", goal="find facts")
    writer = Agent(role="Writer", goal="write report")
    task = Task(description="research and write")
    crew = Crew(agents=[researcher, writer], tasks=[task])
    crew.kickoff()
''', encoding="utf-8")
        (tmp_path / "openai_sample.py").write_text('''
from openai import OpenAI
client = OpenAI()
client.chat.completions.create(model="gpt-4", messages=[])
''', encoding="utf-8")
        return ScanEngine(all_frameworks=all_frameworks).scan(str(tmp_path))

    def test_default_scope_suppresses_crewai(self, tmp_path):
        """Default scan (all_frameworks=False) must not report CrewAI —
        only OpenAI/LangChain/LangGraph (DEFAULT_FRAMEWORK_SCOPE)."""
        results = self._scan_mixed_repo(tmp_path, all_frameworks=False)
        assert results["all_frameworks"] is False
        frameworks = [
            d["framework"] for d in results["primitives"]["decision_points"]
            if d.get("type") == "agent_invocation"
        ]
        assert "crewai" not in frameworks
        providers = [a["provider"] for a in results["primitives"]["ai_integrations"]]
        assert "openai" in providers

    def test_all_frameworks_flag_reports_crewai(self, tmp_path):
        """--all-frameworks (all_frameworks=True) must report CrewAI
        alongside OpenAI — same repo as above, opt-in widened scope."""
        results = self._scan_mixed_repo(tmp_path, all_frameworks=True)
        assert results["all_frameworks"] is True
        frameworks = [
            d["framework"] for d in results["primitives"]["decision_points"]
            if d.get("type") == "agent_invocation"
        ]
        assert "crewai" in frameworks
        providers = [a["provider"] for a in results["primitives"]["ai_integrations"]]
        assert "openai" in providers

    def test_default_scope_includes_openai_agents_sdk(self, tmp_path):
        """OpenAI's own official agent framework (Agent()/Runner.run(),
        tagged "openai_agents_sdk" separately from plain "openai" client
        calls) must be visible in the DEFAULT scope, not just
        --all-frameworks — it's still squarely "OpenAI SDK". Confirmed
        real-world gap before this: 3 real repos built on this SDK
        (including openai/openai-agents-python itself) had every finding
        suppressed by default."""
        (tmp_path / "agent_sample.py").write_text('''
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(agent, "hello")
''', encoding="utf-8")
        results = ScanEngine().scan(str(tmp_path))
        providers = [a["provider"] for a in results["primitives"]["ai_integrations"]]
        assert "openai_agents_sdk" in providers


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
