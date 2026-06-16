# Repository Map

Compact documentation map for the unified migration repository. The original
Phase 0 inventory remains in `../REPOSITORY_MAP.md`; this file records how that
history maps onto the migrated package surface through Phase 13.

## Historical Repository Roles

| Repository | Historical role | Dataset/source | Unified status |
|---|---|---|---|
| `HyperTagging` | First Toy-MC HyperTagging studies, including BASF2 generation, preprocessing, embedding, and reconstruction experiments. | Toy-MC inputs after BASF2 generation and before preprocessing under `/home/boyang/data/MC`. | Toy-MC contracts, dry-run preprocessing adapters, selected model aliases, losses, and examples are migrated. Full BASF2/data production is not migrated. |
| `HyperTaggingColab` | Cleaner collaboration-style package for GraFEI embedding utilities, models, losses, trainer patterns, and preprocessing scripts. | GraFEI-derived data, planned original root `/home/boyang/data/graFEI`. | Package shape, losses, training-loop conventions, and reusable utility ideas informed the unified layout. |
| `graFEI` | Early full GraFEI HyperTagging workflow with embedding, link prediction, reconstruction, and full evaluation scripts. | Original GraFEI data under `/home/boyang/data/graFEI`; many historical scripts use hard-coded HPC paths. | Full reconstruction evaluation behavior from `whole_eva.py` is migrated for CPU tiny events. Full-data evaluation remains HPC-only. |
| `graFEI_reduced` | Reduced/final GraFEI workflow and primary source for many self-contained model definitions. | GraFEI original inputs and derived pair/reconstruction data. | Main source for migrated embedding, link, and reconstruction model classes plus single-level reconstruction behavior. |
| `graFEI_gpt` | GPT-like/autoregressive GraFEI branch. | GraFEI original inputs and GPT-like derived flattened data. | GPT-like masks, collate layout, losses, `GPTReconstructor`, `EmbLinker`, and a conservative `MultiGPT` are migrated. Ambiguous historical `MultiGPT` PDG/feature branch is documented but not reimplemented. |

## Unified Package Surface

- `hypertagging.data`: contracts, tiny fixtures, preprocessing command builders,
  Toy-MC/GraFEI/GPT-like adapters, and GPT-like collate helpers.
- `hypertagging.utils`: padding, checkpoint loading, device, seed, and I/O
  helpers.
- `hypertagging.losses`: embedding, reconstruction, link, and GPT-like tensor
  losses.
- `hypertagging.models`: migrated historical model classes and grouped aliases.
- `hypertagging.training`: CPU dry-run loops and link-training mode helpers.
- `hypertagging.reconstruction`: single-level reconstruction and full GraFEI
  reconstruction evaluation on tiny events.
- `hypertagging.evaluation`: GraFEI event-level full-reconstruction metrics.
- `scripts/`: CPU dry-run CLIs and preprocessing dry-run wrappers.
- `examples/`: runnable fixture-based CPU examples.

## Checkpoints, Logs, And Data

Historical repositories contain many `.pth` checkpoints and log files. The
unified repository does not copy those artifacts. Checkpoint loading helpers
preserve `model_state_dict` and `epoch` handling and default to CPU loading.

Full-data reproduction requires external data and historical checkpoint
provenance not fully verified from repository contents.

## Obsolete Or Experimental Sources

Historical backup folders, `.ipynb_checkpoints/`, `__pycache__/`, large logs,
and exploratory notebooks remain in their source repositories. They are not
migrated unless a later phase needs them for an equivalence test.

## Components Still Treated As Sensitive

Do not refactor these before stronger equivalence tests exist:

- `baum_utils` LCAG/LCA reconstruction behavior.
- GraFEI level ordering, mass tokenization, and link construction.
- Toy-MC channel/event numbering and preprocessing conventions.
- Loss weights, masking logic, and model constructor defaults.
- Full-reconstruction stopping, signal probability, and link index remapping.
- GPT-like level masks, collate layout, and autoregressive masking.
