from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PurePosixPath
import sqlite3
from types import ModuleType

import pytest


REPORT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = REPORT_DIR / "generate_evidence.py"
EVIDENCE_PATH = REPORT_DIR / "evidence.json"
ARTIFACT_PATH = REPORT_DIR / "artifact.json"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hyperbolic_report_evidence", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def generator() -> ModuleType:
    return _load_generator()


@pytest.fixture(scope="session")
def generated(generator: ModuleType) -> dict:
    return generator.generate(REPO_ROOT, generator.DEFAULT_METADATA_ROOTS)


@pytest.fixture(scope="session")
def saved() -> dict:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_generated_file_is_current_and_byte_stable(generator: ModuleType, generated: dict) -> None:
    first = generator._json_bytes(generated)
    second = generator._json_bytes(
        generator.generate(REPO_ROOT, generator.DEFAULT_METADATA_ROOTS)
    )
    assert first == second
    assert EVIDENCE_PATH.read_bytes() == first
    artifact_bytes = generator._json_bytes(generator.build_artifact(generated))
    assert ARTIFACT_PATH.read_bytes() == artifact_bytes


def test_top_level_contract_and_bounded_dataset_inventory(saved: dict) -> None:
    assert saved["version"] == 1
    assert saved["title"] == (
        "Hyperbolic Reconstruction: Data, Model, Training and Inference Readiness"
    )
    assert saved["reportDate"] == "2026-08-14"
    assert saved["generatedAt"] == "2026-08-14T00:00:00Z"
    assert saved["status"] == "partial"

    contracts = saved["datasetContracts"]
    datasets = saved["datasets"]
    assert set(contracts) == set(datasets)
    assert sum(value["laterUse"] == "chart" for value in contracts.values()) == 11
    assert sum(value["laterUse"] == "table" for value in contracts.values()) == 6
    assert sum(value["laterUse"] == "supporting evidence" for value in contracts.values()) == 1
    assert all(isinstance(rows, list) and 0 < len(rows) <= 50 for rows in datasets.values())


def test_key_observed_planned_and_unknown_contracts(saved: dict) -> None:
    facts = saved["facts"]
    assert facts["data"] == {
        "capacityCardinalityOverflowCount": 0,
        "capacityQueryOverflowCount": 0,
        "indexedEvents": 85000,
        "indexedNodes": 3645707,
        "inventoryEvents": 1000000,
        "inventoryShards": 200,
        "observedReconstructionLevels": 6,
        "sealedTestOpened": False,
    }
    assert facts["curriculum"]["plannedSteps"] == 17500
    assert facts["curriculum"]["plannedSeenEvents"] == 70000
    assert facts["curriculum"]["plannedValidationEvents"] == 2000
    assert facts["diagnostic"] == {
        "elapsedSeconds": 132,
        "jobId": "15745941",
        "modelPreset": "small_candidate",
        "peakGpuUtilizationPercent": 11,
        "peakMemoryMiB": 554,
        "peakTemperatureC": 33,
        "phaseIndices": [0, 1, 2, 3],
        "scientificClaimsAllowed": False,
        "steps": 4,
        "telemetrySamples": 9,
        "validationBatches": 1,
        "validationEventViewEvaluations": 8,
        "validationEvents": 2,
        "validationNamedViews": 4,
    }
    assert facts["readiness"]["cpuTests"] == {
        "passed": 463,
        "skipped": 8,
        "warnings": 24,
    }
    assert facts["readiness"]["blockerCount"] == 3
    assert facts["readiness"]["scientificSubmissionAllowed"] is False
    assert facts["readiness"]["h200State"] == "not_submitted_exact_gres_unavailable"
    assert facts["readiness"]["trackedH100JobContractCount"] == 0
    assert {row["status"] for row in saved["unknowns"]} == {"unknown", "planned"}


def test_real_cpu_model_counts_and_module_composition_reconcile(saved: dict) -> None:
    models = saved["facts"]["models"]
    scale = {row["preset"]: row for row in saved["datasets"]["model_parameter_scale"]}
    assert set(models) == {"tiny_cpu", "gpu_debug", "small_candidate", "production_baseline"}
    for preset, model in models.items():
        assert model["pretrainingTrainableParameters"] == sum(
            model["pretrainingTopLevelModules"].values()
        )
        assert model["reconstructionTrainableParameters"] == sum(
            model["reconstructionTopLevelModules"].values()
        )
        assert scale[preset]["pretrainingParameters"] == model["pretrainingTrainableParameters"]
        assert scale[preset]["reconstructionParameters"] == model["reconstructionTrainableParameters"]
    assert [scale[name]["pretrainingParameters"] for name in (
        "tiny_cpu", "gpu_debug", "small_candidate", "production_baseline"
    )] == sorted(row["pretrainingParameters"] for row in scale.values())
    assert models["small_candidate"]["pretrainingTrainableParameters"] == 1395159
    assert models["small_candidate"]["reconstructionTrainableParameters"] == 2135433

    for row in saved["datasets"]["model_module_composition"]:
        count_fields = (
            "encoder",
            "relationBias",
            "contextOrRelationHead",
            "decoder",
            "levelDecoders",
            "leafPidHead",
            "auxiliaryHeads",
        )
        assert sum(row[field] for field in count_fields) == row["totalParameters"]
        assert sum(row[f"{field}Share"] for field in count_fields) == pytest.approx(1.0)


def test_sources_are_safe_and_provenance_labeled(saved: dict) -> None:
    sources = saved["sources"]
    assert sources
    assert len({source["id"] for source in sources}) == len(sources)
    for source in sources:
        path = PurePosixPath(source["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert source["provenance"] in {"tracked_repo", "allowed_reduced_metadata"}
        assert len(source["sha256"]) == 64
    assert {source["provenance"] for source in sources} == {
        "tracked_repo",
        "allowed_reduced_metadata",
    }


def test_sealed_test_and_execution_boundaries(saved: dict) -> None:
    boundary = saved["evidenceBoundary"]
    assert boundary["gpuAccessPerformed"] is False
    assert boundary["schedulerAccessPerformed"] is False
    assert boundary["jobMutationPerformed"] is False
    assert boundary["sealedTestPayloadAccessed"] is False
    assert boundary["modelCheckpointOpened"] is False
    assert saved["facts"]["data"]["sealedTestOpened"] is False
    assert all(row["sealedTestAccessed"] is False for row in saved["datasets"]["subset_ladder"])
    forbidden_suffixes = {".parquet", ".root", ".pt", ".ckpt"}
    assert all(PurePosixPath(source["path"]).suffix not in forbidden_suffixes for source in saved["sources"])
    assert all("sealed" not in source["path"].lower() for source in saved["sources"])
    no_submit = {row["field"]: row["value"] for row in saved["datasets"]["no_submit_contract_rows"]}
    assert no_submit["submission_authorized"] is False
    assert no_submit["submission_performed"] is False
    assert no_submit["sealed_test_role_access"] == "forbidden"


def test_inference_and_ablation_boundaries_are_explicit(saved: dict) -> None:
    inference = saved["facts"]["inference"]
    assert inference["motherP4"] == "daughter_sum"
    assert inference["teacherForcingMothers"] == "truth_guided_topology_reco_features"
    assert inference["inferenceMothers"] == "predicted"
    assert inference["defaultExclusivity"] == "greedy"
    assert inference["setPacking"] == "evaluation_only"
    assert inference["configuredQueries"] == 32
    assert inference["observedMaximumMothers"] == 15
    assert inference["configuredMaxCardinality"] == 16
    assert inference["observedMaximumDaughters"] == 16

    arms = {row["armId"]: row for row in saved["datasets"]["ablation_arm_rows"]}
    assert arms["flat_baseline"]["status"] == "implemented"
    assert arms["hyperbolic_node_geometry"]["status"] == "implemented"
    assert arms["context_removal"]["status"] == "implemented"
    assert arms["physical_plus_hyperbolic_relation"]["status"] == "implemented"
    assert arms["physical_relation_bias_only"]["status"] == "planned_no_named_arm"
    assert {arms[f"scheduled_sampling_{value}"]["evidenceStatus"] for value in (0, 10, 25, 50)} == {"planned"}
    assert saved["facts"]["ablations"]["crossProductsPlanned"] is False


def test_metadata_root_overrides_cannot_escape_or_silently_miss_inputs(
    generator: ModuleType, tmp_path: Path
) -> None:
    escaped = dict(generator.DEFAULT_METADATA_ROOTS)
    escaped["docs"] = str(tmp_path)
    with pytest.raises(RuntimeError, match="path escapes repository root"):
        generator.generate(REPO_ROOT, escaped)

    missing = dict(generator.DEFAULT_METADATA_ROOTS)
    missing["configs"] = "reports/hyperbolic_reconstruction_technical_report_20260814/tests"
    with pytest.raises(RuntimeError, match="required evidence is missing"):
        generator.generate(REPO_ROOT, missing)


def test_canonical_artifact_basics_and_snapshot_identity(artifact: dict, saved: dict) -> None:
    assert set(artifact) == {"surface", "manifest", "snapshot", "sources"}
    assert artifact["surface"] == "report"
    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]
    assert manifest["version"] == 1
    assert manifest["surface"] == "report"
    assert manifest["title"] == (
        "Hyperbolic Reconstruction: Data, Model, Training and Inference Readiness"
    )
    assert manifest["generatedAt"] == "2026-08-14T00:00:00Z"
    assert manifest["cards"] == []
    assert len(manifest["charts"]) == 11
    assert len(manifest["tables"]) == 6
    assert snapshot["version"] == 1
    assert snapshot["generatedAt"] == "2026-08-14T00:00:00Z"
    assert snapshot["status"] == "partial"
    assert snapshot["datasets"] == saved["datasets"]
    assert artifact["sources"] == manifest["sources"]
    assert len(snapshot["accessIssues"]) >= 7
    issue_ids = {issue["id"] for issue in snapshot["accessIssues"]}
    assert {
        "missing-full-convergence",
        "missing-held-out-quality",
        "blocker-production-source",
        "blocker-fresh-preflight",
        "blocker-review-tag-render",
        "h200-unavailable",
        "physical-only-named-arm-gap",
    }.issubset(issue_ids)


def test_visible_section_order_title_and_no_sources_section(artifact: dict) -> None:
    markdown_bodies = [
        block["body"] for block in artifact["manifest"]["blocks"]
        if block["type"] == "markdown"
    ]
    headings = [body.splitlines()[0] for body in markdown_bodies]
    peer_headings = [heading for heading in headings if heading.startswith("# ") or heading.startswith("## ")]
    assert peer_headings == [
        "# Hyperbolic Reconstruction: Data, Model, Training and Inference Readiness",
        "## Technical summary — implementation evidence is strong, scientific readiness remains partial",
        "## Key findings with visual evidence — scale and hierarchy justify set reconstruction, not long-sequence modeling",
        "## Scope, data, and definitions — promotion uses immutable nested subsets and a sealed final test",
        "## Model specification — shared relation-aware geometry stays small enough for staged single-GPU study",
        "## Training methodology and details — progressive objectives and free rollout address geometry and exposure bias",
        "## Inference design — learned proposals are bounded by exclusivity, topology, and exact daughter-sum kinematics",
        "## Limitations and robustness — the four-step V100 run proves execution, not learning",
        "## Staged training and ablation plan — isolate mechanisms sequentially and never cross-product the grid",
        "## Recommended next steps — clear provenance and execution gates before any scientific run",
        "## Further questions — scientific quality, scaling, and causal attribution remain open",
    ]
    visible = "\n".join(markdown_bodies)
    assert "## Sources" not in visible
    assert "Mixture-of-experts is deferred" in visible
    assert "Whole-set scoring and iterative within-mother pointer decoding remain deferred" in visible
    assert "physical-relation-only named arm is absent" in visible
    assert "diagnostic smoke evidence only" in visible


def _encoding_fields(chart: dict) -> set[str]:
    fields: set[str] = set()
    for encoding in chart["encodings"].values():
        values = encoding if isinstance(encoding, list) else [encoding]
        for value in values:
            if "field" in value:
                fields.add(value["field"])
            fields.update(value.get("fields", []))
    return fields


def test_chart_contracts_dataset_fields_sources_and_colors(artifact: dict) -> None:
    manifest = artifact["manifest"]
    datasets = artifact["snapshot"]["datasets"]
    source_ids = {source["id"] for source in artifact["sources"]}
    allowed_types = {
        "line", "area", "stackedArea", "bar", "horizontalBar", "stackedBar",
        "stackedBar100", "horizontalStackedBar", "horizontalStackedBar100",
        "histogram", "scatter", "heatmap", "pie", "leaderboard", "sparkline",
        "funnel", "waterfall", "boxPlot",
    }
    allowed_colors = {
        "blue", "purple", "green", "neutral", "orange", "yellow", "pink", "red"
    }
    allowed_palette_kinds = {
        "categorical", "sequential", "diverging", "semantic", "identity"
    }
    required_semantics = {
        "title", "subtitle", "intent", "question", "rationale", "comparisonContext",
        "encodings", "xAxisTitle", "yAxisTitle", "valueFormat", "unit", "sourceId",
        "maxRows", "dataset", "palette",
    }
    unsupported_legacy_fields = {"xField", "series"}
    single_measure_charts = 0
    multi_measure_charts = 0
    for chart in manifest["charts"]:
        assert required_semantics.issubset(chart)
        assert unsupported_legacy_fields.isdisjoint(chart)
        assert all(chart[field] not in (None, "", {}) for field in required_semantics)
        assert {"x", "y"}.issubset(chart["encodings"])
        assert chart["encodings"]["x"].get("field")
        y_encoding = chart["encodings"]["y"]
        assert ("field" in y_encoding) != ("fields" in y_encoding)
        if "field" in y_encoding:
            assert y_encoding["field"]
            single_measure_charts += 1
        else:
            assert len(y_encoding["fields"]) > 1
            assert len(y_encoding["fields"]) == len(set(y_encoding["fields"]))
            multi_measure_charts += 1
        assert chart["type"] in allowed_types
        assert chart["dataset"] in datasets
        assert chart["sourceId"] in source_ids
        rows = datasets[chart["dataset"]]
        fields = set().union(*(row.keys() for row in rows))
        assert _encoding_fields(chart).issubset(fields)
        assert chart["palette"]["kind"] in allowed_palette_kinds
        assert chart["palette"].get("name")
        assert set(chart["palette"]["name"].split("-")) & allowed_colors
        assert chart["maxRows"] >= len(rows)
        assert chart["comparisonContext"]["grain"]
        assert chart["comparisonContext"]["denominator"]
        assert chart["comparisonContext"]["unit"]
        if chart["type"] in {
            "bar", "horizontalBar", "stackedBar", "horizontalStackedBar",
            "horizontalStackedBar100",
        }:
            assert any(line["value"] == 0 for line in chart["referenceLines"])
    assert single_measure_charts == 2
    assert multi_measure_charts == 9


def test_every_chart_has_dedicated_explanatory_adjacency(artifact: dict) -> None:
    blocks = artifact["manifest"]["blocks"]
    chart_blocks = [index for index, block in enumerate(blocks) if block["type"] == "chart"]
    assert len(chart_blocks) == 11
    for index in chart_blocks:
        adjacent = []
        if index > 0 and blocks[index - 1]["type"] == "markdown":
            adjacent.append(blocks[index - 1]["body"])
        if index + 1 < len(blocks) and blocks[index + 1]["type"] == "markdown":
            adjacent.append(blocks[index + 1]["body"])
        assert any(
            "**Takeaway.**" in body
            and "**How to read.**" in body
            and "**Implication/caveat.**" in body
            for body in adjacent
        ), f"chart block lacks dedicated explanatory adjacency: {blocks[index]}"


def test_table_contracts_dataset_fields_and_sources(artifact: dict) -> None:
    manifest = artifact["manifest"]
    datasets = artifact["snapshot"]["datasets"]
    source_ids = {source["id"] for source in artifact["sources"]}
    assert len(manifest["tables"]) == 6
    for table in manifest["tables"]:
        assert table["dataset"] in datasets
        assert table["sourceId"] in source_ids
        assert table["title"] and table["subtitle"]
        fields = set().union(*(row.keys() for row in datasets[table["dataset"]]))
        assert {column["field"] for column in table["columns"]}.issubset(fields)
        assert table["density"] in {"dense", "spacious"}


def test_every_chart_and_table_source_executes_exact_dataset_sql(
    artifact: dict, saved: dict, generator: ModuleType
) -> None:
    manifest = artifact["manifest"]
    sources = {source["id"]: source for source in artifact["sources"]}
    evidence_json = json.dumps(
        saved,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    items = [*manifest["charts"], *manifest["tables"]]
    assert len(items) == 17
    assert len({item["dataset"] for item in items}) == 17

    with sqlite3.connect(":memory:") as connection:
        for item in items:
            assert "source" not in item
            dataset_name = item["dataset"]
            assert generator.SAFE_DATASET_NAME.fullmatch(dataset_name)
            assert item["sourceId"] == generator._dataset_source_id(dataset_name)
            source = sources[item["sourceId"]]
            query = source["query"]
            sql = query["sql"]
            assert sql == generator._dataset_select_sql(dataset_name)
            assert sql.lstrip().upper().startswith("SELECT ")
            assert ":evidence_json" in sql
            assert f"'$.datasets.{dataset_name}'" in sql
            assert "json_each" in sql.lower()
            assert "ORDER BY CAST(key AS INTEGER)" in sql
            assert query["engine"] == "sqlite3"
            assert query["dialect"] == "SQLite JSON1"
            assert query["parameters"]["evidence_json"]
            assert query["description"]
            assert query["tables_used"][0] == generator.REPORT_EVIDENCE_PATH

            result = connection.execute(
                sql,
                {"evidence_json": evidence_json},
            ).fetchall()
            selected_rows = [json.loads(row_json) for (row_json,) in result]
            assert selected_rows == saved["datasets"][dataset_name]


def test_dataset_sql_identifier_validation_fails_closed(generator: ModuleType) -> None:
    with pytest.raises(RuntimeError, match="unsafe artifact dataset identifier"):
        generator._dataset_select_sql("subset_ladder') UNION SELECT 'bad")


def test_artifact_source_paths_and_sealed_test_boundary(artifact: dict) -> None:
    source_ids = {source["id"] for source in artifact["sources"]}
    assert len(source_ids) == len(artifact["sources"])
    for source in artifact["sources"]:
        path = PurePosixPath(source["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert "sealed" not in source["path"].lower()
        for used in source.get("query", {}).get("tables_used", []):
            used_path = PurePosixPath(used)
            assert not used_path.is_absolute()
            assert ".." not in used_path.parts
            assert "sealed" not in used.lower()
    for collection in ("cards", "charts", "tables"):
        assert all(item["sourceId"] in source_ids for item in artifact["manifest"][collection])
    assert all(
        row["sealedTestAccessed"] is False
        for row in artifact["snapshot"]["datasets"]["subset_ladder"]
    )
    no_submit = {
        row["field"]: row["value"]
        for row in artifact["snapshot"]["datasets"]["no_submit_contract_rows"]
    }
    assert no_submit["sealed_test_role_access"] == "forbidden"
    assert no_submit["submission_performed"] is False
