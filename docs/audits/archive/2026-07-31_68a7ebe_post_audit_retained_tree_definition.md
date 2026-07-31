# Post-audit retained-tree definition

Audit scope: current working tree based on starting SHA
`68a7ebe81b14b6c329d78de91be18b5458c49089` (2026-07-31,
Europe/Berlin).

HyperTagging reconstructs the retained/pruned tree after the documented PID
filter and contraction. It does not reconstruct every generator intermediate.
Vocabulary expansion is a separate physics ablation and was not performed.

## Reduced vocabulary

The model vocabulary is `legacy-reduced-pdg-v1`. Token 0 is explicit unknown;
the remaining 40 entries are retained species.

| Token | PDG | Meaning |
|---:|---:|---|
| 0 | 0 | unknown |
| 1 | 300553 | Upsilon(4S) |
| 2 | 22 | gamma |
| 3 | 130 | K_L0 |
| 4 | 111 | pi0 |
| 5 | 443 | J/psi |
| 6 | 310 | K_S0 |
| 7 | -11 | e+ |
| 8 | 321 | K+ |
| 9 | 211 | pi+ |
| 10 | -13 | mu+ |
| 11 | 2212 | p+ |
| 12 | 3122 | Lambda0 |
| 13 | 3222 | Sigma+ |
| 14 | 411 | D+ |
| 15 | 421 | D0 |
| 16 | 431 | D_s+ |
| 17 | 4122 | Lambda_c+ |
| 18 | 413 | D*+ |
| 19 | 423 | D*0 |
| 20 | 433 | D_s*+ |
| 21 | 511 | B0 |
| 22 | 521 | B+ |
| 23 | 531 | B_s0 |
| 24 | 11 | e- |
| 25 | -321 | K- |
| 26 | -211 | pi- |
| 27 | 13 | mu- |
| 28 | -2212 | anti-p- |
| 29 | -3122 | anti-Lambda0 |
| 30 | -3222 | anti-Sigma- |
| 31 | -411 | D- |
| 32 | -421 | anti-D0 |
| 33 | -431 | D_s- |
| 34 | -4122 | anti-Lambda_c- |
| 35 | -413 | D*- |
| 36 | -423 | anti-D*0 |
| 37 | -433 | D_s*- |
| 38 | -511 | anti-B0 |
| 39 | -521 | B- |
| 40 | -531 | anti-B_s0 |

## Static reconstructed-mother ontology

`reduced-mother-ontology-v1` permits the following reduced mother tokens before
level-specific empirical priors: 1, 4, 5, 6, 12–23, and 29–40. Equivalently,
the permitted PDGs are Upsilon(4S), pi0, J/psi, K_S0, Lambda/Sigma,
D/D_s/Lambda_c and vector partners, B/B_s, and their listed conjugates.
Unknown, gamma, K_L, charged leptons, charged kaons/pions, and (anti)protons are
leaf-only and cannot be emitted as mothers.

## Contraction and topology-only semantics

The preprocessor retains primary MC particles admitted by the reduced
vocabulary. For every retained child it links the nearest retained mother. Any
pruned generator particles between those endpoints are contracted. Therefore
the possible contracted generator PDGs are data-dependent and are not a fixed
enumeration in source. Current schema-v4 stores a `contracted_intermediate`
marker on affected retained parents and a `contracted_intermediate_path` flag
on affected children, but it does not store the complete PDG list of every
contracted generator along the path. A real-data contracted-PDG frequency
claim cannot be recovered from fixtures and requires an explicit bounded real
pilot or a future provenance extension.

Retained truth nodes without reconstructed matches are labelled
`truth_topology_only`. Reconstructed objects without a retained truth match are
labelled `unmatched_reco`. These categories and competing/copy provenance are
not synonyms.

## K_L and KLM support

K_L0 is token 3 and can occur in retained truth topology or in an explicitly
configured reconstructed `Particle` candidate array. The current direct raw
collector reads Tracks and neutral ECLClusters. It does not collect a raw
`KLMClusters` store array. Consequently direct raw-KLM/K_L input support is not
verified or complete. This is a deferred detector-input question, not evidence
for expanding the vocabulary.

## Denominators and depth

- `diagnostic_all` counts all retained mothers.
- `reconstructable_partial` counts valid retained mothers with enough retained
  reconstructed daughters even if the full generator decay is incomplete.
- `complete_only` additionally requires recursive reconstructable completeness.

The stored reconstruction level remains
`L(mother) = 1 + max L(daughter)`. Maximum reconstruction level is a tree
height. Exact root depth and pair distances are separate
`retained-tree-exact-edges-v2` quantities derived from parent links.

## Executed fixture inventory

This command wrote a new `/tmp` fixture; it did not read or overwrite production
parquet:

```text
/data/dust/user/b/boyangyu/uv_env/bin/python -c "... write_notebook_fixture_v4('/tmp/post-audit-retained-tree-fixture.parquet') ..."
```

Observed deterministic fixture scope: 2 events, 14 nodes, 8 leaves, 4 level-1
nodes, 2 level-2 nodes, maximum reconstruction level 2, 6 `complete_only`
targets, 6 `reconstructable_partial` targets, 1 unmatched reconstructed object,
0 truth-topology-only nodes, 0 retained parents marked as contracting an
intermediate, and 0 K_L nodes. These are fixture contract counts only. They are
not estimates of generator contraction frequency, unmatched-object rate,
K_L/KLM coverage, retained depth, or target-policy denominators in real data.

## Real-data status

No real parquet was supplied to this pass and no full preprocessing was run.
Real frequencies of contracted intermediates, truth-topology-only nodes,
unmatched objects, K_L/KLM inputs, levels/depths, and policy denominators are
therefore **NOT MEASURED**. The real-only trained physics notebook and the
sub-100-event pilot remain guarded against fixture substitution.
