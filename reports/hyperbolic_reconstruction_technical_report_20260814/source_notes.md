# Technical report source notes

## Evidence spine

The report answers one bounded question: is the repository ready to begin the staged scientific evaluation of hyperbolic-pretrained, level-autoregressive set reconstruction, and what is already observed versus merely planned or unknown?

The decision-useful answer is partial. Data publication, immutable train/validation selection, CPU-tested mechanics, model construction, capacity metadata, and one bounded V100 diagnostic are supported. Long convergence, free-rollout scientific quality, category/channel robustness, calibration, and sealed-test performance are absent. The next authorized action remains prerequisite clearance followed by the 35k stage; the evidence does not support larger-scale promotion.

The evidence path is:

1. Reduced-publication inventory and immutable selection metadata establish population, role, and denominator boundaries.
2. The complete-only train+validation index and capacity report establish observed structure without opening sealed test.
3. Repository model code and presets are instantiated on CPU to establish trainable parameter totals and module composition.
4. Scientific configs and the execution plan establish planned curriculum, validation, promotion, and ablation budgets.
5. Job `15745941` receipt, raw telemetry, metrics, and readiness metadata establish bounded runtime execution only.
6. Architecture documentation and reconstruction config establish teacher-forcing, rollout, daughter-sum p4, exclusivity, and checkpoint-gate contracts.
7. The blocked no-submit contract and readiness metadata establish current blockers and resource limitations.

## Dataset grain and status

| Dataset | Grain | Evidence label | Later use |
|---|---|---|---|
| `production_category_composition` | One row per production category | observed metadata | chart |
| `structural_percentiles` | One row per populated reconstruction level | observed train+validation index | chart |
| `subset_ladder` | One row per nested training selection | observed metadata | chart |
| `model_parameter_scale` | One row per CPU-instantiated preset | observed CPU instantiation | chart |
| `model_module_composition` | One row per `small_candidate` model surface | observed CPU instantiation | chart |
| `curriculum_plan` | One row for the 35k scientific pretraining plan | planned | chart |
| `diagnostic_telemetry` | One row per 15-second telemetry sample | observed diagnostic | chart |
| `inference_capacity_controls` | One row per count-based capacity control | observed train+validation index | chart |
| `staged_training_budgets` | One row per promotion stage | planned | chart |
| `readiness_gates` | One row per readiness gate, one-hot state | mixed observed/planned/unknown | chart |
| `ablation_coverage` | One row per ablation family | mixed implemented/planned | chart |
| `data_contract_rows` | One row per exact data/index contract field | observed | table |
| `model_contract_rows` | One row per exact preset/model field | observed | table |
| `diagnostic_contract_rows` | One row per exact diagnostic field | observed diagnostic | table |
| `inference_contract_rows` | One row per inference or rollout control | mixed implemented/planned | table |
| `no_submit_contract_rows` | One row per no-submit field or blocker | mixed observed/planned | table |
| `ablation_arm_rows` | One row per implemented or planned arm | mixed implemented/planned | table |
| `curriculum_phase_rows` | One row per progressive phase | planned | supporting evidence |

All datasets are bounded to at most 50 rows. For each chart/table-bound dataset, the generator executes a parameterized SQLite JSON1 `SELECT` over `evidence.json`, parses the returned row JSON in canonical array order, and requires exact row-for-row equality before artifact construction.

## Claim-to-source map

| Claim family | Primary source IDs | Derivation and checks |
|---|---|---|
| Reduced production composition | `inventory`, `selection_summary` | Inventory-entry category/event counts are recomputed and reconciled with summary shard counts. |
| Nested 35k/100k/250k ladder and fixed held-out roles | `selection_summary`, `training_readiness` | Selection rows are required to be nondecreasing; index evidence must state `sealed_test_opened=false`. |
| Structural percentiles and capacity | `capacity_small_candidate`, `training_readiness` | Per-level quantiles and maxima are extracted; query/cardinality overflow counts must be zero. |
| Model preset scale and composition | `model_config_source`, `model_source`, `ablation_source`, `pretrain_source`, `small_candidate_config` | Actual repository modules are imported from this checkout and instantiated on CPU; trainable tensor elements are grouped by top-level module. YAML/code architecture fields are cross-checked. |
| Curriculum and staged budgets | `pretrain_scientific_config`, `curriculum_stage_1`–`curriculum_stage_4`, `training_plan` | Four phase budgets must sum to configured maximum steps; seen-event budgets are parsed from the plan tables. |
| Diagnostic execution and telemetry | `diagnostic_receipt`, `diagnostic_telemetry`, `diagnostic_telemetry_summary`, `diagnostic_metrics`, `training_readiness`, `pretrain_diagnostic_config` | Receipt duration, exit status, model preset, phase indices, validation counts, sample count, and peaks must reconcile across sources. |
| Training/inference information boundary | `architecture_document`, `reconstruction_config`, `capacity_small_candidate` | Required daughter-sum p4, greedy exclusivity, evaluation-only resolver, and scheduled-sampling phrases must exist; exact numeric gates come from config/index metadata. |
| Readiness and submission block | `training_readiness`, `no_submit_contract`, `current_status` | Submission must remain unauthorized/unperformed; three blockers are required; CPU audit counts are parsed from current status. |
| Ablation coverage | `ablation_source`, named `ablation_config_*` inputs, `training_plan` | Named YAML arms are enumerated separately from planned scientific comparisons. The physical-only named-arm gap is explicit. |

`report_evidence` remains the canonical derived snapshot source for datasets that join multiple inputs. Each of the 17 chart/table datasets additionally has a dataset-specific `evidence_dataset_*` source containing the exact executed SQLite JSON1 query, parameter description, canonical evidence path, and relevant upstream source paths.

## Required-section map

| Required report role | Visible section | Evidence placement |
|---|---|---|
| 1. Title | `# Hyperbolic Reconstruction: Data, Model, Training and Inference Readiness` | Title-only markdown block |
| 2. Technical summary | Implementation evidence is strong, scientific readiness remains partial | Exact data/index table |
| 3. Key findings with visual evidence | Scale and hierarchy justify set reconstruction | Category composition and structural-percentile charts |
| 4. Scope, data, and definitions | Immutable nested subsets and sealed final test | Subset-ladder chart |
| 5. Model specification | Shared relation-aware geometry and preset scale | Parameter-scale chart, module-composition chart, exact model table |
| 6. Training methodology/details | Progressive objectives and staged promotion | Curriculum and staged-budget charts |
| 7. Inference design | Learned proposals with daughter-sum p4 and exclusivity | Capacity chart and exact inference table |
| 8. Limitations/robustness | Four-step runtime evidence is not learning | Telemetry chart and exact diagnostic table |
| 9. Staged training and ablation plan | Sequential isolation without cross-products | Coverage chart and arm registry |
| 10. Recommended next steps | Clear blockers before 35k science | Blocked no-submit table |
| 11. Further questions | Scientific quality and scaling remain open | Readiness-state chart |

Each quantitative chart is immediately preceded by a dedicated markdown explanation containing a takeaway, reading instruction, evidence note, method/assumption, and implication/caveat.

## Assumptions and omissions

- `observed` means direct extraction from tracked repository evidence or an explicitly permitted reduced-metadata file, or a CPU model instantiation from repository code. It does not imply scientific validation.
- `planned` means serialized in a tracked config or execution plan but not executed as a scientific result.
- `unknown` means the required scientific evidence is absent. Unknown is not encoded as zero.
- The reduced 1M campaign is source-task sampled and category-skewed. It is not treated as category-representative.
- Structural percentiles cover the complete-only 35k train plus fixed validation index. They do not cover sealed test and do not prove the maximum possible production structure.
- Model totals include all trainable parameters allocated by the instantiated module. In the full-revised reconstruction model, compatibility top-level modules remain counted even when the heterogeneous path does not consume them in its main forward pass.
- Channel-memory buffers are not trainable parameters and are excluded from trainable totals.
- Mother p4 is the daughter sum. Later physical fits or constraints are outside this evidence package; no unconstrained truth-momentum regression is claimed.
- The report does not include full convergence, final free-rollout reconstruction quality, calibration, rare-channel/KLM robustness, representative throughput, bootstrap intervals, or sealed-test results.
- No report filter is exposed because this is a bounded technical report rather than an exploratory dashboard.
- No visible Sources section is included. Provenance is carried by canonical `sourceId` references and artifact source metadata.

## Deferred designs and implementation boundary

- Mixture-of-experts is deferred. The implemented production model is a shared encoder/task-geometry design.
- Whole-set scoring and iterative within-mother pointer decoding are deferred designs. They are not runnable model logits, training objectives, or production inference controls.
- Bounded set packing and beam search are implemented as evaluation-only diagnostics, not default production resolution and not scientific evidence by themselves.
- A combined physical-plus-hyperbolic relation-attention arm is runnable. A standalone named physical-bias-on/hyperbolic-refinement-off arm is absent; the sequential comparison remains planned.
- Geometry variants for level encoding, radius target, and tangent scale are implemented as named configs, but their scientific ordering is unknown until matched evaluation.

## Caveats that change interpretation

- Job `15745941` is a four-step diagnostic with a two-event validation cohort. Its telemetry demonstrates bounded execution and observability only.
- Low utilization and memory in that diagnostic cannot be extrapolated to scientific throughput or capacity because the run is intentionally tiny.
- The exact H200 resource was unavailable in the observed readiness metadata. There is no tracked H100 job contract; this is not proof of external scheduler state or a general statement about hardware availability.
- CPU tests validate software contracts, formulae, masks, rollout termination, and deterministic extraction. They do not demonstrate physics improvement.
- The no-submit contract remains authoritative: production-source recovery, a fresh in-allocation preflight, and clean review/tag/render gates are unresolved.

## Reproducibility

The generator uses a fixed report date and timestamp, sorted JSON keys, bounded arrays, exact-file input allowlists, repository-containment checks, SHA-256 source identities, CPU-only model instantiation, and stdlib `sqlite3` JSON1 row extraction. Dataset identifiers must match a strict safe-name pattern; every chart/table query binds the full canonical evidence document as `:evidence_json`, orders `json_each` rows by integer array key, parses `row_json`, and fails unless the result exactly equals the bound dataset. It also fails on missing required files, unsafe root overrides, cross-source diagnostic disagreement, changed curriculum structure, sealed-test access flags, unexpected submission authorization, or stale generated bytes.

Regeneration and validation commands are recorded in `README.md`. `report.html` is now the primary delivered surface. The official portable builder recorded `ok=true`, `validation=passed`, `package=passed`, `verification=structural_only`, and counts of 31 blocks, 11 charts, and 6 tables in the verbatim `delivery_receipt.json`. Structural review confirmed the exact title, canonical embedded manifest/snapshot/sources payload, semantic fallback, and absence of external asset dependencies.

Browser QA remains incomplete: the packaging host had no compatible Chromium headless-shell, so enhanced-reader rendering, responsive viewports, source dialogs, and source interaction were not verified. The receipt's absolute `html` path identifies the local packaging host only and is not canonical artifact provenance.
