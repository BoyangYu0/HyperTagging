"""Lazy parquet/shard loading, source-safe splits, and online normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import cast, Iterable, Iterator, Mapping, Sequence
import warnings

import torch
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, IterableDataset

from hypertagging.data.heterogeneous import (
    HeterogeneousEvent,
    collate_heterogeneous_events,
    heterogeneous_event_from_record,
)
from hypertagging.data.splitting import SourceAwareSplitConfig, stable_split_name
from hypertagging.data.streaming import StreamingMaskedFeatureNormalizer
from hypertagging.preprocessing.schema_v4 import (
    TARGET_COMPOSITE_METADATA_INDICES,
    iter_event_records_v4,
)


FEATURE_BLOCKS = ("common", "track", "cluster", "composite")
BALANCED_LEVEL_REPLAY_LEVELS = (1, 2, 3, 4, 5, 6)
BALANCED_LEVEL_REPLAY_VERSION = "balanced-level-replay-v1"
BALANCED_LEVEL_REPLAY_UID_HASH_SCHEME = "sha256-u64be-length-prefixed-utf8-v1"
_DATA_BINDING_PROVENANCE = object()
_RESOLVED_DATA_BINDING_PROVENANCE = object()


@dataclass(frozen=True)
class _AuthenticatedDatasetDataBinding:
    """Private one-read index/manifest provenance token."""

    index: object
    manifest: object | None
    _provenance: object


@dataclass(frozen=True)
class _ResolvedDatasetDataBinding:
    """Private combined token after the index-only resolution phase."""

    authenticated: _AuthenticatedDatasetDataBinding
    resolved_index: object
    _provenance: object


def _uid_sequence_sha256(uids: Sequence[str]) -> str:
    """Hash an ordered UID sequence without delimiter ambiguity."""

    digest = hashlib.sha256()
    for uid in uids:
        encoded = str(uid).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _eligible_reconstruction_target_mask(
    event: HeterogeneousEvent,
    *,
    target_level: int,
    target_policy: str,
    min_daughters: int = 2,
) -> torch.Tensor:
    """Mirror the loss target policy for one unbatched event and level."""

    if target_policy not in {
        "complete_only",
        "reconstructable_partial",
        "diagnostic_all",
    }:
        raise ValueError(f"unknown reconstruction target policy: {target_policy}")
    if target_level <= 0:
        raise ValueError("balanced replay target levels must be positive")
    if min_daughters < 0:
        raise ValueError("min_daughters must be non-negative")
    eligible = event.active.bool() & event.level_ids.eq(target_level)
    if target_policy != "diagnostic_all":
        eligible &= event.valid_reconstruction_target.bool()
    if target_policy == "complete_only":
        eligible &= event.recursive_reconstructable_complete.bool()
    if min_daughters == 0 or not bool(eligible.any()):
        return eligible
    context = event.active.bool() & event.level_ids.lt(target_level)
    context_ids = context.nonzero(as_tuple=False).flatten()
    mothers = eligible.nonzero(as_tuple=False).flatten()
    daughter_counts = event.daughter_adjacency[mothers][:, context_ids].sum(dim=-1)
    eligible[mothers[daughter_counts < min_daughters]] = False
    return eligible


@dataclass(frozen=True)
class BalancedLevelReplay:
    """Deterministic random-access replay over target-policy-eligible UID pools."""

    events_by_uid: Mapping[str, HeterogeneousEvent]
    pools_by_level: Mapping[int, tuple[str, ...]]
    levels: tuple[int, ...]
    seed: int
    target_policy: str
    min_daughters: int = 2
    _permutation_cache: dict[int, tuple[int, tuple[str, ...]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            not self.levels
            or self.levels != tuple(sorted(self.levels))
            or len(set(self.levels)) != len(self.levels)
            or any(level <= 0 for level in self.levels)
        ):
            raise ValueError(
                "balanced replay levels must be unique, positive, and increasing"
            )
        if self.target_policy not in {
            "complete_only",
            "reconstructable_partial",
            "diagnostic_all",
        }:
            raise ValueError(
                f"unknown reconstruction target policy: {self.target_policy}"
            )
        if self.min_daughters < 0:
            raise ValueError("min_daughters must be non-negative")
        if set(self.pools_by_level) != set(self.levels):
            raise ValueError("balanced replay pools do not exactly match its levels")
        for level in self.levels:
            pool = self.pools_by_level[level]
            if not pool:
                raise ValueError(f"balanced replay pool for level {level} is empty")
            if pool != tuple(sorted(pool)):
                raise ValueError(
                    f"balanced replay pool for level {level} is not UID-sorted"
                )
            missing = [uid for uid in pool if uid not in self.events_by_uid]
            if missing:
                raise ValueError(
                    f"balanced replay pool for level {level} has unknown UIDs: "
                    f"{missing[:3]}"
                )

    @property
    def materialized_event_count(self) -> int:
        return len(self.events_by_uid)

    @property
    def pool_counts(self) -> dict[int, int]:
        return {level: len(self.pools_by_level[level]) for level in self.levels}

    @property
    def pool_hashes(self) -> dict[int, str]:
        return {
            level: _uid_sequence_sha256(self.pools_by_level[level])
            for level in self.levels
        }

    def level_counts(self, *, start_slot: int, slot_count: int) -> dict[int, int]:
        """Return exact round-robin counts without materializing selections."""

        if start_slot < 0 or slot_count < 0:
            raise ValueError("balanced replay slots must be non-negative")
        quotient, remainder = divmod(slot_count, len(self.levels))
        counts = {level: quotient for level in self.levels}
        for offset in range(remainder):
            level = self.levels[(start_slot + offset) % len(self.levels)]
            counts[level] += 1
        return counts

    def contract(
        self,
        *,
        planned_slot_count: int | None = None,
        planned_start_slot: int = 0,
    ) -> dict[str, object]:
        """Return the JSON-safe sampler identity and optional finite schedule."""

        contract: dict[str, object] = {
            "version": BALANCED_LEVEL_REPLAY_VERSION,
            "levels": list(self.levels),
            "seed": int(self.seed),
            "target_policy": self.target_policy,
            "min_daughters": int(self.min_daughters),
            "materialized_train_event_count": self.materialized_event_count,
            "materialized_train_uid_sha256": _uid_sequence_sha256(
                tuple(sorted(self.events_by_uid))
            ),
            "eligible_pool_counts_by_level": {
                str(level): count for level, count in self.pool_counts.items()
            },
            "eligible_pool_uid_sha256_by_level": {
                str(level): value for level, value in self.pool_hashes.items()
            },
            "uid_hash_scheme": BALANCED_LEVEL_REPLAY_UID_HASH_SCHEME,
            "level_schedule": "global_slot_round_robin",
            "pool_order": "uid_lexicographic_before_permutation",
            "pool_permutation": "sha256_ranked_by_seed_level_cycle_uid",
            "replacement_policy": "new_seeded_permutation_on_pool_exhaustion",
        }
        if planned_slot_count is not None:
            counts = self.level_counts(
                start_slot=planned_start_slot,
                slot_count=planned_slot_count,
            )
            contract["planned_schedule"] = {
                "start_slot": int(planned_start_slot),
                "slot_count": int(planned_slot_count),
                "end_slot_exclusive": int(planned_start_slot + planned_slot_count),
                "level_counts": {str(level): count for level, count in counts.items()},
                "max_minus_min_level_count": max(counts.values())
                - min(counts.values()),
            }
        return contract

    def _permutation(self, level: int, cycle: int) -> tuple[str, ...]:
        cached = self._permutation_cache.get(level)
        if cached is not None and cached[0] == cycle:
            return cached[1]
        pool = self.pools_by_level[level]

        def rank(uid: str) -> tuple[bytes, str]:
            payload = (
                f"{BALANCED_LEVEL_REPLAY_VERSION}\0{self.seed}\0{level}\0{cycle}\0{uid}"
            ).encode("utf-8")
            return hashlib.sha256(payload).digest(), uid

        permutation = tuple(sorted(pool, key=rank))
        self._permutation_cache[level] = (cycle, permutation)
        return permutation

    def selection_at(self, global_slot: int) -> tuple[int, str]:
        """Map one absolute optimizer presentation slot to (level, event UID)."""

        if global_slot < 0:
            raise ValueError("balanced replay global_slot must be non-negative")
        level_count = len(self.levels)
        level = self.levels[global_slot % level_count]
        occurrence = global_slot // level_count
        pool_size = len(self.pools_by_level[level])
        cycle, position = divmod(occurrence, pool_size)
        return level, self._permutation(level, cycle)[position]

    def selections(
        self, *, start_slot: int, slot_count: int
    ) -> tuple[tuple[int, str], ...]:
        if start_slot < 0 or slot_count < 0:
            raise ValueError("balanced replay slots must be non-negative")
        return tuple(
            self.selection_at(slot)
            for slot in range(start_slot, start_slot + slot_count)
        )

    def collated_batch(
        self, *, start_slot: int, batch_size: int
    ) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        choices = self.selections(start_slot=start_slot, slot_count=batch_size)
        batch = collate_heterogeneous_events(
            [self.events_by_uid[uid] for _level, uid in choices]
        )
        batch["selected_target_levels"] = torch.tensor(
            [level for level, _uid in choices], dtype=torch.long
        )
        batch["balanced_replay_global_slots"] = torch.arange(
            start_slot, start_slot + batch_size, dtype=torch.long
        )
        return batch


def build_balanced_level_replay(
    events: Iterable[HeterogeneousEvent],
    *,
    target_policy: str,
    seed: int,
    levels: Sequence[int] = BALANCED_LEVEL_REPLAY_LEVELS,
    min_daughters: int = 2,
) -> BalancedLevelReplay:
    """Consume train events once and build sorted eligible UID pools."""

    ordered_levels = tuple(int(level) for level in levels)
    if not ordered_levels or len(set(ordered_levels)) != len(ordered_levels):
        raise ValueError("balanced replay levels must be non-empty and unique")
    if ordered_levels != tuple(sorted(ordered_levels)):
        raise ValueError("balanced replay levels must be strictly increasing")
    # Validate the policy even for an empty iterator.
    if target_policy not in {
        "complete_only",
        "reconstructable_partial",
        "diagnostic_all",
    }:
        raise ValueError(f"unknown reconstruction target policy: {target_policy}")
    materialized: dict[str, HeterogeneousEvent] = {}
    pools: dict[int, list[str]] = {level: [] for level in ordered_levels}
    for event in events:
        uid = str(event.event_uid)
        if uid in materialized:
            raise ValueError(
                f"balanced replay requires unique event UIDs; duplicate={uid!r}"
            )
        materialized[uid] = event
        for level in ordered_levels:
            eligible = _eligible_reconstruction_target_mask(
                event,
                target_level=level,
                target_policy=target_policy,
                min_daughters=min_daughters,
            )
            if bool(eligible.any()):
                pools[level].append(uid)
    if not materialized:
        raise ValueError("balanced replay training split is empty")
    empty_levels = [level for level, uids in pools.items() if not uids]
    if empty_levels:
        raise ValueError(
            "balanced replay has no target-policy-eligible train events for levels "
            f"{empty_levels}"
        )
    sorted_pools = {level: tuple(sorted(uids)) for level, uids in pools.items()}
    return BalancedLevelReplay(
        events_by_uid=MappingProxyType(materialized),
        pools_by_level=MappingProxyType(sorted_pools),
        levels=ordered_levels,
        seed=int(seed),
        target_policy=target_policy,
        min_daughters=int(min_daughters),
    )


@dataclass
class RealDataModule:
    input_paths: tuple[str, ...]
    normalizers: dict[str, StreamingMaskedFeatureNormalizer]
    split_manifest: dict[str, object]
    split_manifest_hash: str
    overflow_counters: dict[str, int]
    seed: int
    split_config: SourceAwareSplitConfig
    max_events: int | None = None
    max_nodes: int | None = None
    max_nodes_overflow: str = "raise"
    shuffle_buffer_size: int = 1024
    allow_legacy_conflated: bool = False
    split_overrides: dict[str, str] = field(default_factory=dict)
    source_split_overrides: dict[str, str] = field(default_factory=dict)
    selection_manifest_hash: str | None = None
    split_counts: dict[str, int] = field(default_factory=dict)
    legacy_conflated_fraction: float = 0.0
    allowed_types_by_level: dict[int, tuple[int, ...]] = field(default_factory=dict)
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    source_schema_versions: tuple[str, ...] = ()
    track_fit_policies: tuple[str, ...] = ()
    dataset_index: dict[str, object] | None = None
    _materialized_splits: dict[str, list[HeterogeneousEvent]] | None = field(
        default=None, init=False, repr=False
    )
    _balanced_level_replay_cache: dict[
        tuple[str, tuple[int, ...], int, int], BalancedLevelReplay
    ] = field(default_factory=dict, init=False, repr=False)
    balanced_level_replay_contract: dict[str, object] | None = field(
        default=None, init=False
    )

    def iter_events(
        self,
        split: str,
        *,
        shuffle: bool = False,
        epoch: int = 0,
    ) -> Iterator[HeterogeneousEvent]:
        records: Iterator[dict] = self._records()
        if shuffle and self.shuffle_buffer_size > 0:
            from hypertagging.data.streaming import BoundedShuffleBuffer

            records = iter(
                BoundedShuffleBuffer(
                    records,
                    size=self.shuffle_buffer_size,
                    seed=self.seed + epoch,
                )
            )
        for record in records:
            assigned = _assigned_split(
                record,
                self.split_config,
                event_overrides=self.split_overrides,
                source_overrides=self.source_split_overrides,
                require_source_override=self.selection_manifest_hash is not None,
            )
            if assigned != split:
                continue
            event = heterogeneous_event_from_record(record)
            if (
                self.max_nodes is not None
                and event.common_features.shape[0] > self.max_nodes
            ):
                if self.max_nodes_overflow == "drop":
                    self.overflow_counters["max_nodes_dropped"] += 1
                    continue
                raise OverflowError(
                    f"event {event.event_uid} has {event.common_features.shape[0]} nodes, "
                    f"exceeding max_nodes={self.max_nodes}"
                )
            yield event

    def _records(self) -> Iterator[dict]:
        emitted = 0
        for input_path in self.input_paths:
            for record in iter_event_records_v4(input_path):
                if self.max_events is not None and emitted >= self.max_events:
                    return
                emitted += 1
                yield record

    def batches(
        self,
        split: str,
        *,
        batch_size: int,
        shuffle: bool,
        epoch: int = 0,
        start_events: int = 0,
        cycle_epochs: bool = False,
    ) -> Iterator[dict[str, torch.Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if start_events < 0:
            raise ValueError("start_events must be non-negative")
        if cycle_epochs and self.num_workers > 0:
            raise ValueError(
                "exact presentation batching with epoch carry requires num_workers=0"
            )
        if self.num_workers > 0:
            dataset = _StreamingHeterogeneousDataset(self, split, shuffle, epoch)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_heterogeneous_events,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )
            for batch in loader:
                yield self.normalize_batch(batch)
            return
        pending: list[HeterogeneousEvent] = []
        current_epoch = int(epoch)
        skip = int(start_events)
        while True:
            for event in self.iter_events(
                split, shuffle=shuffle, epoch=current_epoch
            ):
                if skip:
                    skip -= 1
                    continue
                pending.append(event)
                if len(pending) == batch_size:
                    yield self.normalize_batch(collate_heterogeneous_events(pending))
                    pending.clear()
            if not cycle_epochs:
                if pending:
                    yield self.normalize_batch(collate_heterogeneous_events(pending))
                return
            # Presentation-based production resumes carry the terminal tail
            # into the next deterministically shuffled epoch.  This prevents a
            # non-divisible source-split tail from becoming an undersized
            # optimizer update while preserving every presentation.
            current_epoch += 1
            skip = 0

    def balanced_level_replay(
        self,
        *,
        target_policy: str,
        seed: int,
        levels: Sequence[int] = BALANCED_LEVEL_REPLAY_LEVELS,
        min_daughters: int = 2,
        planned_slot_count: int | None = None,
    ) -> BalancedLevelReplay:
        """Materialize and cache one immutable global level-balanced replay."""

        ordered_levels = tuple(int(level) for level in levels)
        key = (target_policy, ordered_levels, int(seed), int(min_daughters))
        replay = self._balanced_level_replay_cache.get(key)
        if replay is None:
            replay = build_balanced_level_replay(
                self.iter_events("train", shuffle=False),
                target_policy=target_policy,
                seed=seed,
                levels=ordered_levels,
                min_daughters=min_daughters,
            )
            self._balanced_level_replay_cache[key] = replay
        self.balanced_level_replay_contract = replay.contract(
            planned_slot_count=planned_slot_count
        )
        return replay

    def balanced_level_replay_batches(
        self,
        replay: BalancedLevelReplay,
        *,
        batch_size: int,
        start_slot: int,
        stop_slot: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield finite full replay batches over an absolute slot interval."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if start_slot < 0 or stop_slot < start_slot:
            raise ValueError("invalid balanced replay slot interval")
        if (stop_slot - start_slot) % batch_size:
            raise ValueError("balanced replay interval must contain full batches")
        for slot in range(start_slot, stop_slot, batch_size):
            yield self.normalize_batch(
                replay.collated_batch(start_slot=slot, batch_size=batch_size)
            )

    def normalize_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        result = dict(batch)
        for block, normalizer in self.normalizers.items():
            # Common/composite values remain raw until the authoritative
            # model-side runtime normalizer. Track/cluster blocks are static.
            if block in {"common", "composite"}:
                continue
            values_key = f"{block}_features"
            availability_key = f"{block}_availability"
            result[values_key] = normalizer.transform(
                result[values_key], result[availability_key]
            )
        result["node_features"] = result["common_features"]
        result["runtime_features_are_raw"] = torch.tensor(True)
        return result

    def normalization_state(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            name: normalizer.state_dict()
            for name, normalizer in self.normalizers.items()
        }

    @property
    def splits(self) -> dict[str, list[HeterogeneousEvent]]:
        """Diagnostic compatibility view; trainers intentionally do not use it."""

        if self.max_events is None or self.max_events > 10_000:
            raise RuntimeError(
                "materializing RealDataModule.splits is disabled for large datasets; "
                "use iter_events or an explicit max_events<=10000 diagnostic pilot"
            )
        if self._materialized_splits is None:
            self._materialized_splits = {
                name: list(self.iter_events(name, shuffle=False))
                for name in ("train", "validation", "test")
            }
        return self._materialized_splits

    @property
    def events(self) -> list[HeterogeneousEvent]:
        return [
            event
            for name in ("train", "validation", "test")
            for event in self.splits[name]
        ]


def preflight_dataset_index_data_binding(
    data: str | Path | Sequence[str | Path],
    dataset_index: str | Path,
    *,
    required_splits: Sequence[str] | None = None,
    target_policy: str = "complete_only",
    max_events: int | None = None,
    split_config: SourceAwareSplitConfig | None = None,
    seed: int | None = None,
    allow_legacy_conflated: bool = False,
    scientific_mode: bool = False,
    pilot_split_repair: bool = False,
) -> dict[str, object]:
    """Bind authenticated index, manifest, and caller metadata before paths."""

    binding = _preflight_dataset_index_data_binding(
        data,
        dataset_index,
        required_splits=required_splits,
        target_policy=target_policy,
        max_events=max_events,
        split_config=split_config,
        seed=seed,
        allow_legacy_conflated=allow_legacy_conflated,
        scientific_mode=scientific_mode,
        pilot_split_repair=pilot_split_repair,
    )
    resolved = _resolve_dataset_index_data_binding(binding)
    payload, _manifest = _require_resolved_dataset_data_binding(resolved)
    return dict(payload)


def _preflight_dataset_index_data_binding(
    data: str | Path | Sequence[str | Path],
    dataset_index: str | Path,
    *,
    required_splits: Sequence[str] | None = None,
    target_policy: str = "complete_only",
    max_events: int | None = None,
    split_config: SourceAwareSplitConfig | None = None,
    seed: int = 20260730,
    allow_legacy_conflated: bool = False,
    scientific_mode: bool = False,
    pilot_split_repair: bool = False,
) -> _AuthenticatedDatasetDataBinding:
    from hypertagging.data.dataset_index import _load_dataset_index_binding

    index_binding = _load_dataset_index_binding(dataset_index)
    _validate_source_role_caller_prefix(
        index_binding.payload,
        max_events=max_events,
        pilot_split_repair=pilot_split_repair,
    )
    manifest_binding = _require_source_role_manifest_binding(
        data, index_binding.payload, required_splits=required_splits
    )
    _validate_index_caller_contract(
        index_binding.payload,
        data=data,
        target_policy=target_policy,
        max_events=max_events,
        split_config=split_config,
        seed=seed,
        required_splits=required_splits,
        allow_legacy_conflated=allow_legacy_conflated,
        scientific_mode=scientific_mode,
        pilot_split_repair=pilot_split_repair,
        manifest=(manifest_binding.payload if manifest_binding is not None else None),
    )
    return _AuthenticatedDatasetDataBinding(
        index=index_binding,
        manifest=manifest_binding,
        _provenance=_DATA_BINDING_PROVENANCE,
    )


def _validate_source_role_caller_prefix(
    index: Mapping[str, object],
    *,
    max_events: int | None,
    pilot_split_repair: bool,
) -> None:
    """Reject source-role-only caller modes before opening its manifest JSON."""

    if not isinstance(pilot_split_repair, bool):
        raise ValueError("pilot_split_repair must be boolean")
    selection = index.get("selection_contract")
    if not isinstance(selection, Mapping):
        raise ValueError("dataset index selection contract is missing")
    if selection.get("mode") == "source_role_manifest":
        if max_events is not None:
            raise ValueError("training-selection manifests cannot use max_events")
        if pilot_split_repair:
            raise ValueError("selection manifests cannot use pilot_split_repair")


def _require_authenticated_dataset_data_binding(
    binding: object,
) -> tuple[dict[str, object], object | None]:
    """Validate the exact private combined token without reopening metadata."""

    if (
        type(binding) is not _AuthenticatedDatasetDataBinding
        or binding._provenance is not _DATA_BINDING_PROVENANCE
    ):
        raise ValueError("dataset data provenance binding is invalid")
    from hypertagging.data.dataset_index import _require_authenticated_index_binding

    index = _require_authenticated_index_binding(binding.index)
    manifest_binding = binding.manifest
    mode = index["selection_contract"]["mode"]
    if mode == "source_role_manifest":
        if manifest_binding is None:
            raise ValueError("source-role dataset data binding is missing its manifest")
        from hypertagging.data.training_selection import (
            _require_authenticated_manifest_binding,
        )

        _require_authenticated_manifest_binding(manifest_binding)
    elif manifest_binding is not None:
        raise ValueError("raw dataset data binding cannot contain a manifest")
    return index, manifest_binding


def _resolve_dataset_index_data_binding(
    binding: _AuthenticatedDatasetDataBinding,
) -> _ResolvedDatasetDataBinding:
    """Resolve authenticated index paths while retaining pinned manifest identity."""

    _require_authenticated_dataset_data_binding(binding)
    from hypertagging.data.dataset_index import _resolve_dataset_index_binding

    resolved_index = _resolve_dataset_index_binding(binding.index)
    return _ResolvedDatasetDataBinding(
        authenticated=binding,
        resolved_index=resolved_index,
        _provenance=_RESOLVED_DATA_BINDING_PROVENANCE,
    )


def _require_resolved_dataset_data_binding(
    binding: object,
) -> tuple[dict[str, object], object | None]:
    """Validate both levels of the private combined provenance token."""

    if (
        type(binding) is not _ResolvedDatasetDataBinding
        or binding._provenance is not _RESOLVED_DATA_BINDING_PROVENANCE
    ):
        raise ValueError("resolved dataset data provenance binding is invalid")
    index, manifest = _require_authenticated_dataset_data_binding(
        binding.authenticated
    )
    from hypertagging.data.dataset_index import _require_resolved_index_binding

    resolved_index, _paths, _shard_paths = _require_resolved_index_binding(
        binding.resolved_index
    )
    if resolved_index is not index:
        raise ValueError("resolved dataset data provenance binding is inconsistent")
    return index, manifest


def _load_training_selection_from_dataset_data_binding(
    binding: _ResolvedDatasetDataBinding,
    *,
    include_splits: Sequence[str] | None,
    required_splits: Sequence[str] | None,
):
    """Load the pinned selection through the combined token only."""

    _index, manifest = _require_resolved_dataset_data_binding(binding)
    if manifest is None:
        raise ValueError("dataset data binding does not contain a selection manifest")
    from hypertagging.data.training_selection import _load_training_selection_bound

    return _load_training_selection_bound(
        manifest,
        include_splits=include_splits,
        required_splits=required_splits,
        _index_binding=binding.resolved_index,
    )


def _load_dataset_index_from_dataset_data_binding(
    binding: _ResolvedDatasetDataBinding,
) -> dict[str, object]:
    """Verify indexed sources through the combined pinned token only."""

    _require_resolved_dataset_data_binding(binding)
    from hypertagging.data.dataset_index import _load_resolved_dataset_index_bound

    return _load_resolved_dataset_index_bound(binding.resolved_index)


def _require_source_role_manifest_binding(
    data: str | Path | Sequence[str | Path],
    index: dict[str, object],
    *,
    required_splits: Sequence[str] | None = None,
) -> object | None:
    """Reject noncanonical data arguments using index/manifest metadata only."""

    indexed_selection = index.get("selection_contract", {})
    if indexed_selection.get("mode") != "source_role_manifest":
        return None
    if not isinstance(data, (str, Path)) or Path(data).suffix.lower() != ".json":
        raise ValueError(
            "a source-role-bound dataset index requires its exact immutable "
            "training-selection manifest; raw Parquet paths are forbidden"
        )
    from hypertagging.data.training_selection import (
        HASH_FIELD,
        _load_selection_manifest_binding,
        _validate_training_selection_index_metadata_pure,
    )

    manifest_binding = _load_selection_manifest_binding(data)
    manifest = manifest_binding.payload
    if (
        manifest.get(HASH_FIELD)
        != indexed_selection.get("selection_manifest_hash")
    ):
        raise ValueError("dataset index training-selection hash mismatch")
    _validate_training_selection_index_metadata_pure(
        manifest,
        index,
        include_splits=indexed_selection.get("included_splits"),
        required_splits=required_splits,
    )
    return manifest_binding


def _validate_index_caller_contract(
    index: Mapping[str, object],
    *,
    data: str | Path | Sequence[str | Path],
    target_policy: str,
    max_events: int | None,
    split_config: SourceAwareSplitConfig | None,
    seed: int | None,
    required_splits: Sequence[str] | None,
    allow_legacy_conflated: bool,
    scientific_mode: bool,
    pilot_split_repair: bool,
    manifest: Mapping[str, object] | None,
) -> None:
    """Check caller/index identity before resolving or reading source paths."""

    if not isinstance(allow_legacy_conflated, bool):
        raise ValueError("allow_legacy_conflated must be boolean")
    if not isinstance(scientific_mode, bool):
        raise ValueError("scientific_mode must be boolean")
    if not isinstance(pilot_split_repair, bool):
        raise ValueError("pilot_split_repair must be boolean")
    if max_events is not None and (
        not isinstance(max_events, int)
        or isinstance(max_events, bool)
        or max_events <= 0
    ):
        raise ValueError("max_events must be a positive integer or None")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool)
    ):
        raise ValueError("seed must be an integer")
    if not isinstance(target_policy, str):
        raise ValueError("target policy is invalid")
    required = (
        _validate_required_splits(required_splits)
        if required_splits is not None
        else ()
    )

    selection = index.get("selection_contract")
    if not isinstance(selection, Mapping):
        raise ValueError("dataset index selection contract is missing")
    mode = selection.get("mode")
    indexed_max_events = selection.get("max_events")
    if scientific_mode:
        identity = index.get("event_identity_validation")
        if not isinstance(identity, Mapping):
            raise ValueError(
                "scientific mode requires the passed exact identity/task-binding gate"
            )
        if (
            identity.get("status") != "passed"
            or identity.get("task_binding")
            != "selection_to_sidecar_to_completion_marker_validated"
            or identity.get("sealed_test_opened") is not False
            or selection.get("included_splits") != ["train", "validation"]
        ):
            raise ValueError(
                "scientific mode requires the passed exact identity/task-binding "
                "gate with sealed test excluded"
            )
    if mode == "source_role_manifest":
        if max_events is not None:
            raise ValueError("training-selection manifests cannot use max_events")
        if pilot_split_repair:
            raise ValueError("selection manifests cannot use pilot_split_repair")
        if manifest is None:
            raise ValueError("source-role index requires its bound selection manifest")
        included = selection.get("included_splits")
        if manifest.get("manifest_hash") != selection.get("selection_manifest_hash"):
            raise ValueError("dataset index training-selection hash mismatch")
        if manifest.get("selection_mode") != "explicit_whole_shard_source_roles":
            raise ValueError("training selection mode is invalid")
        if scientific_mode and included != ["train", "validation"]:
            raise ValueError("scientific mode permits exactly train and validation roles")
    else:
        if manifest is not None:
            raise ValueError("raw dataset indexes cannot be paired with a selection manifest")
        if indexed_max_events != max_events:
            raise ValueError(
                "dataset index event-selection/max-events fingerprint mismatch"
            )
        indexed_paths = index.get("paths")
        if isinstance(data, (str, Path)):
            caller_values = [data]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            caller_values = list(data)
        else:
            raise ValueError("indexed raw data must be explicit parquet paths")
        caller_paths: list[str] = []
        for value in caller_values:
            if not isinstance(value, (str, Path)):
                raise ValueError("indexed raw data must be explicit parquet paths")
            lexical = str(value)
            candidate = Path(lexical)
            if (
                not lexical
                or "\x00" in lexical
                or "\\" in lexical
                or not candidate.is_absolute()
                or str(candidate) != lexical
                or any(part in {".", ".."} for part in candidate.parts)
                or candidate.suffix != ".parquet"
            ):
                raise ValueError(
                    "indexed raw data must use canonical absolute parquet paths"
                )
            caller_paths.append(lexical)
        if not isinstance(indexed_paths, list) or caller_paths != indexed_paths:
            raise ValueError("dataset index shard paths do not match requested data")

    if index.get("target_policy") != target_policy:
        raise ValueError(
            "dataset index target policy does not match trainer target policy; "
            "request an explicit rescan to change policy"
        )
    if split_config is not None:
        if not isinstance(split_config, SourceAwareSplitConfig):
            raise ValueError("dataset index split configuration mismatch")
        if index.get("split_config") != split_config.__dict__:
            raise ValueError("dataset index split configuration mismatch")
    elif (
        manifest is None
        and seed is not None
        and index.get("split_config") != SourceAwareSplitConfig(seed=seed).__dict__
    ):
        raise ValueError("dataset index split configuration mismatch")
    legacy_fraction = index.get("legacy_fraction")
    if not isinstance(legacy_fraction, (int, float)) or isinstance(legacy_fraction, bool):
        raise ValueError("dataset index legacy fraction is invalid")
    if float(legacy_fraction) and not allow_legacy_conflated:
        raise ValueError("dataset index reports legacy-conflated nodes")

    counts = index.get("split_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("dataset index split counts are invalid")
    for split in ("train", "validation", "test"):
        count = counts.get(split, 0)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("dataset index split counts are invalid")
    if manifest is not None:
        manifest_counts = manifest.get("split_counts")
        manifest_shards = manifest.get("split_shard_counts")
        included = selection.get("included_splits", ())
        if not isinstance(manifest_counts, Mapping) or not isinstance(manifest_shards, Mapping):
            raise ValueError("training selection split counts are invalid")
        for split in ("train", "validation", "test"):
            expected = manifest_counts.get(split, 0) if split in included else 0
            if counts.get(split, 0) != expected:
                raise ValueError("dataset index split counts disagree with training selection")
    missing = [split for split in required if counts.get(split, 0) == 0]
    if missing:
        raise ValueError(f"indexed dataset has empty required split(s) {missing}")
def _validate_required_splits(values: object) -> tuple[str, ...]:
    vocabulary = ("train", "validation", "test")
    if isinstance(values, (str, bytes, set, frozenset, Mapping)):
        raise ValueError("required_splits must be an ordered sequence")
    try:
        required = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("required_splits must be an ordered sequence") from error
    if not required:
        raise ValueError("required_splits must be nonempty")
    previous = -1
    for split in required:
        if not isinstance(split, str) or split not in vocabulary:
            raise ValueError("required_splits contains an invalid split")
        position = vocabulary.index(split)
        if position <= previous:
            raise ValueError(
                "required_splits must be a unique order-preserving subsequence"
            )
        previous = position
    return required


def build_real_data_module(
    data: str | Path | Sequence[str | Path],
    *,
    max_events: int | None = None,
    max_nodes: int | None = None,
    max_nodes_overflow: str = "raise",
    split_config: SourceAwareSplitConfig | None = None,
    seed: int = 20260730,
    pilot_split_repair: bool = False,
    allow_legacy_conflated: bool = False,
    shuffle_buffer_size: int = 1024,
    normalization_state: dict[str, dict[str, torch.Tensor]] | None = None,
    required_splits: tuple[str, ...] = ("train",),
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    dataset_index: str | Path | None = None,
    rescan_dataset: bool = False,
    target_policy: str = "complete_only",
    allow_incomplete_v4_publication: bool = False,
    scientific_mode: bool = False,
) -> RealDataModule:
    """Build a restartable streaming data module without retaining event lists."""

    from hypertagging.data.training_selection import (
        _load_selection_manifest_binding_or_none,
        _validated_training_selection_metadata,
    )

    required_splits = _validate_required_splits(required_splits)
    if scientific_mode and (dataset_index is None or rescan_dataset):
        raise ValueError(
            "scientific mode requires a promoted full-record dataset index"
        )
    if scientific_mode and "test" in required_splits:
        raise ValueError("scientific indexing/training cannot request the sealed test role")
    config = split_config or SourceAwareSplitConfig(seed=seed)
    preflight_binding = None
    resolved_binding = None
    if dataset_index is not None and not rescan_dataset:
        preflight_binding = _preflight_dataset_index_data_binding(
            data,
            dataset_index,
            required_splits=required_splits,
            target_policy=target_policy,
            max_events=max_events,
            split_config=split_config,
            seed=seed,
            allow_legacy_conflated=allow_legacy_conflated,
            scientific_mode=scientific_mode,
            pilot_split_repair=pilot_split_repair,
        )
    preflight_index = None
    preflight_manifest_binding = None
    preflight_manifest = None
    if preflight_binding is not None:
        preflight_index, preflight_manifest_binding = (
            _require_authenticated_dataset_data_binding(preflight_binding)
        )
        if preflight_manifest_binding is not None:
            preflight_manifest = preflight_manifest_binding.payload
    selection = None
    source_role_bound = bool(
        preflight_index
        and preflight_index.get("selection_contract", {}).get("mode")
        == "source_role_manifest"
    )
    manifest_binding = preflight_manifest_binding
    manifest_metadata = preflight_manifest
    if not source_role_bound and isinstance(data, (str, Path)):
        candidate = Path(data)
        if candidate.suffix == ".json":
            candidate_binding = _load_selection_manifest_binding_or_none(candidate)
            if candidate_binding is not None:
                _validated_training_selection_metadata(candidate_binding.payload)
            manifest_binding = candidate_binding
            manifest_metadata = (
                candidate_binding.payload if candidate_binding is not None else None
            )
    if manifest_metadata is not None and max_events is not None:
        raise ValueError(
            "training-selection manifests cannot be combined with raw max_events prefixes"
        )
    if manifest_metadata is not None and pilot_split_repair:
        raise ValueError("selection manifests cannot use pilot_split_repair")
    indexed_split_counts: dict[str, int] | None = None
    indexed_legacy_fraction: float | None = None
    if preflight_index is not None:
        if source_role_bound and split_config is None:
            config = SourceAwareSplitConfig(**preflight_index["split_config"])
        indexed_split_counts = {
            name: int(preflight_index["split_counts"].get(name, 0))
            for name in ("train", "validation", "test")
        }
        indexed_legacy_fraction = float(preflight_index["legacy_fraction"])
    if source_role_bound or manifest_metadata is not None:
        include_splits = None
        indexed_included = (
            preflight_index.get("selection_contract", {}).get("included_splits")
            if preflight_index is not None
            else None
        )
        if scientific_mode and indexed_included != ["train", "validation"]:
            raise ValueError(
                "scientific mode permits exactly train and validation roles"
            )
        if indexed_included:
            include_splits = tuple(str(value) for value in indexed_included)
        if preflight_binding is not None:
            resolved_binding = _resolve_dataset_index_data_binding(preflight_binding)
        if source_role_bound:
            selection = _load_training_selection_from_dataset_data_binding(
                resolved_binding,
                include_splits=include_splits,
                required_splits=required_splits,
            )
        else:
            from hypertagging.data.training_selection import (
                _load_training_selection_bound,
            )

            selection = _load_training_selection_bound(
                manifest_binding,
                include_splits=include_splits,
                required_splits=required_splits,
            )
    if scientific_mode and selection is None:
        raise ValueError(
            "scientific mode requires an immutable training-selection manifest"
        )
    if selection is not None and max_events is not None:
        raise ValueError(
            "training-selection manifests cannot be combined with raw max_events prefixes"
        )
    if selection is not None and pilot_split_repair:
        raise ValueError("selection manifests cannot use pilot_split_repair")
    if selection is not None:
        paths = list(selection.paths)
    elif preflight_binding is not None:
        if resolved_binding is None:
            resolved_binding = _resolve_dataset_index_data_binding(preflight_binding)
        _index, _manifest = _require_resolved_dataset_data_binding(resolved_binding)
        paths = _resolve_indexed_raw_caller_paths(
            data, resolved_binding.resolved_index.resolved_paths
        )
    else:
        paths = resolve_data_paths(data)
    source_split_overrides = (
        dict(selection.source_split_overrides) if selection is not None else {}
    )
    # The selection loader verifies its manifest plus sidecar/marker hashes and
    # marker-bound parquet digest reference. Avoid re-reading every selected
    # parquet payload here; the promoted full dataset index is the payload/UID
    # verification gate.
    if not allow_incomplete_v4_publication and selection is None:
        _require_complete_v4_publications(paths)
    if dataset_index is not None and not rescan_dataset:
        from hypertagging.data.dataset_index import tensor_normalizer_state

        index = _load_dataset_index_from_dataset_data_binding(resolved_binding)
        split_counts = cast(dict[str, int], indexed_split_counts)
        legacy_fraction = cast(float, indexed_legacy_fraction)
        split_manifest = {
            "seed": seed,
            "groups": index["source_groups"],
            "split_counts": split_counts,
            "pilot_split_repair": False,
            "overrides": {},
            "dataset_index_hash": index["index_hash"],
            "selection_manifest_hash": (
                selection.manifest_hash if selection is not None else None
            ),
        }
        manifest_json = json.dumps(
            split_manifest, sort_keys=True, separators=(",", ":")
        )
        module = RealDataModule(
            input_paths=tuple(str(path) for path in paths),
            normalizers={},
            split_manifest=split_manifest,
            split_manifest_hash=hashlib.sha256(manifest_json.encode()).hexdigest(),
            overflow_counters={"max_nodes_dropped": 0, "invalid_events": 0},
            seed=seed,
            split_config=config,
            max_events=max_events,
            max_nodes=max_nodes,
            max_nodes_overflow=max_nodes_overflow,
            shuffle_buffer_size=shuffle_buffer_size,
            allow_legacy_conflated=allow_legacy_conflated,
            split_counts=split_counts,
            source_split_overrides=source_split_overrides,
            selection_manifest_hash=(
                selection.manifest_hash if selection is not None else None
            ),
            legacy_conflated_fraction=legacy_fraction,
            allowed_types_by_level={
                int(level): tuple(int(token) for token in tokens)
                for level, tokens in index["allowed_types_by_level"].items()
            },
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            source_schema_versions=tuple(index["schema_versions"]),
            track_fit_policies=tuple(
                sorted(str(value) for value in index.get("track_fit_policies", []))
            ),
            dataset_index=index,
        )
        state = normalization_state or tensor_normalizer_state(index)
        for block, block_state in state.items():
            normalizer = StreamingMaskedFeatureNormalizer()
            normalizer.load_state_dict(block_state)
            module.normalizers[block] = normalizer
        return module
    split_counts = {"train": 0, "validation": 0, "test": 0}
    group_splits: dict[str, str] = {}
    source_counts: dict[str, dict[str, int]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    first_uid = ""
    first_split = ""
    total_nodes = 0
    legacy_nodes = 0
    scanned = 0
    allowed_types: dict[int, set[int]] = {}
    source_schema_versions: set[str] = set()
    track_fit_policies = _track_fit_policies_from_publications(paths)
    for path in paths:
        for record in iter_event_records_v4(path):
            if max_events is not None and scanned >= max_events:
                break
            scanned += 1
            uid = str(record["event_uid"])
            source_schema_versions.add(
                str(
                    record.get(
                        "source_schema_version", record.get("schema_version", "")
                    )
                )
            )
            split = _assigned_split(
                record,
                config,
                event_overrides={},
                source_overrides=source_split_overrides,
                require_source_override=selection is not None,
            )
            if not first_uid:
                first_uid, first_split = uid, split
            split_counts[split] += 1
            source = str(record.get("source_file", ""))
            source_counts[split][source] = source_counts[split].get(source, 0) + 1
            group = source if config.group_by_source_file and source else uid
            previous = group_splits.setdefault(group, split)
            if previous != split:
                raise ValueError(f"source group {group!r} leaks across splits")
            for node in record.get("nodes", []):
                total_nodes += 1
                legacy_nodes += int(
                    str(node.get("leaf_kinematics_mode")) == "legacy_conflated"
                )
                if (
                    split == "train"
                    and int(node.get("level", 0)) > 0
                    and bool(node.get("valid_reconstruction_target", False))
                ):
                    allowed_types.setdefault(int(node["level"]), set()).add(
                        int(node.get("pid_target_token", 0))
                    )
        if max_events is not None and scanned >= max_events:
            break
    if scanned == 0:
        raise ValueError("no events were loaded from the supplied parquet data")
    if selection is not None and split_counts != selection.split_counts:
        raise ValueError(
            "training-selection event counts disagree with scanned publications"
        )
    legacy_fraction = legacy_nodes / max(total_nodes, 1)
    if legacy_nodes and not allow_legacy_conflated:
        raise ValueError(
            "legacy-conflated v1/v2/v3 data are rejected for real training; "
            "use --allow-legacy-conflated only for diagnostic, non-data-compatible runs"
        )
    if legacy_nodes:
        warnings.warn(
            f"DIAGNOSTIC ONLY: {legacy_fraction:.2%} of nodes are legacy-conflated; "
            "data-compatible performance claims are disabled",
            RuntimeWarning,
            stacklevel=2,
        )
    overrides: dict[str, str] = {}
    missing = [name for name in required_splits if split_counts[name] == 0]
    if missing:
        if pilot_split_repair and missing == ["train"] and first_uid:
            overrides[first_uid] = "train"
            split_counts[first_split] -= 1
            split_counts["train"] += 1
        else:
            raise ValueError(
                f"source-aware split has empty required split(s) {missing}; "
                "change grouping/fractions or explicitly enable pilot_split_repair"
            )
    split_manifest = {
        "seed": seed,
        "groups": dict(sorted(group_splits.items())),
        "source_counts": source_counts,
        "split_counts": split_counts,
        "pilot_split_repair": bool(pilot_split_repair),
        "overrides": overrides,
        "source_role_overrides": dict(sorted(source_split_overrides.items())),
        "selection_manifest_hash": (
            selection.manifest_hash if selection is not None else None
        ),
    }
    manifest_json = json.dumps(split_manifest, sort_keys=True, separators=(",", ":"))
    module = RealDataModule(
        input_paths=tuple(str(path) for path in paths),
        normalizers={},
        split_manifest=split_manifest,
        split_manifest_hash=hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
        overflow_counters={"max_nodes_dropped": 0, "invalid_events": 0},
        seed=seed,
        split_config=config,
        max_events=max_events,
        max_nodes=max_nodes,
        max_nodes_overflow=max_nodes_overflow,
        shuffle_buffer_size=shuffle_buffer_size,
        allow_legacy_conflated=allow_legacy_conflated,
        split_overrides=overrides,
        source_split_overrides=source_split_overrides,
        selection_manifest_hash=(
            selection.manifest_hash if selection is not None else None
        ),
        split_counts=split_counts,
        legacy_conflated_fraction=legacy_fraction,
        allowed_types_by_level={
            level: tuple(sorted(tokens)) for level, tokens in allowed_types.items()
        },
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        source_schema_versions=tuple(sorted(source_schema_versions)),
        track_fit_policies=track_fit_policies,
    )
    if normalization_state is None:
        module.normalizers = fit_training_normalizers(module.iter_events("train"))
    else:
        for block, state in normalization_state.items():
            normalizer = StreamingMaskedFeatureNormalizer()
            normalizer.load_state_dict(state)
            module.normalizers[block] = normalizer
    return module


def _resolve_indexed_raw_caller_paths(
    data: str | Path | Sequence[str | Path],
    indexed_paths: Sequence[Path],
) -> list[Path]:
    """Resolve and bind raw caller paths before any publication/source read."""

    caller_values = [data] if isinstance(data, (str, Path)) else list(data)
    resolved = [Path(value).resolve() for value in caller_values]
    if tuple(resolved) != tuple(indexed_paths):
        raise ValueError("resolved dataset index paths do not match requested data")
    return resolved


def _require_complete_v4_publications(paths: Sequence[Path]) -> None:
    from hypertagging.data.dataset_index import _validated_completion_marker

    for path in paths:
        try:
            fields = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception as error:
            raise ValueError(f"cannot inspect parquet publication {path}") from error
        if "event_json" not in fields:
            # Legacy v1-v3 container compatibility remains available.
            continue
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        marker = path.with_suffix(path.suffix + ".complete")
        if not sidecar.is_file() or not marker.is_file():
            raise ValueError(
                f"incomplete schema-v4 publication {path}: parquet, metadata sidecar, "
                "and completion marker are required"
            )
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid schema-v4 metadata sidecar {sidecar}") from error
        _validated_completion_marker(path, metadata)


def _track_fit_policies_from_publications(
    paths: Sequence[Path],
) -> tuple[str, ...]:
    """Read the MC-independent fit contract even when no index is supplied."""

    policies: set[str] = set()
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        if not sidecar.is_file():
            continue
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid metadata sidecar {sidecar}") from error
        policy = str(metadata.get("track_fit_policy", "")).strip()
        if policy:
            policies.add(policy)
    return tuple(sorted(policies))


def fit_training_normalizers(
    events: Iterator[HeterogeneousEvent] | Sequence[HeterogeneousEvent],
) -> dict[str, StreamingMaskedFeatureNormalizer]:
    output = {block: StreamingMaskedFeatureNormalizer() for block in FEATURE_BLOCKS}
    count = 0
    for event in events:
        count += 1
        for block in FEATURE_BLOCKS:
            availability = getattr(event, f"{block}_availability")
            if block == "composite":
                availability = availability.clone()
                availability[:, list(TARGET_COMPOSITE_METADATA_INDICES)] = False
            output[block].update(
                getattr(event, f"{block}_features"),
                availability,
            )
    if count == 0:
        raise ValueError("cannot fit feature normalization without training events")
    return output


def resolve_data_paths(data: str | Path | Sequence[str | Path]) -> list[Path]:
    entries = [data] if isinstance(data, (str, Path)) else list(data)
    output: list[Path] = []
    for entry in entries:
        path = Path(entry)
        records = None
        if path.suffix == ".json":
            from hypertagging.data.training_selection import (
                _load_selection_manifest_binding_or_none,
                _load_training_selection_bound,
            )

            manifest_binding = _load_selection_manifest_binding_or_none(path)
            if manifest_binding is not None:
                output.extend(_load_training_selection_bound(manifest_binding).paths)
                continue
            records = json.loads(path.read_text(encoding="utf-8"))
        elif path.is_dir():
            output.extend(sorted(path.glob("*.parquet")))
        elif path.suffix == ".parquet":
            output.append(path)
        elif path.suffix == ".jsonl":
            text = path.read_text(encoding="utf-8")
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            raise ValueError(f"unsupported data input: {path}")
        if records is not None:
            if isinstance(records, dict):
                records = records.get("shards", records.get("entries", []))
            for record in records:
                candidate = (
                    record
                    if isinstance(record, str)
                    else (
                        record.get("output_file")
                        or record.get("output")
                        or record.get("path")
                        or record.get("parquet")
                    )
                )
                if candidate:
                    candidate_path = Path(candidate)
                    output.append(
                        candidate_path
                        if candidate_path.is_absolute()
                        else path.parent / candidate_path
                    )
    unique = sorted({path.resolve() for path in output})
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing parquet shard(s): {missing}")
    if not unique:
        raise ValueError("data input resolved to no parquet shards")
    return unique


class _StreamingHeterogeneousDataset(IterableDataset):
    def __init__(
        self,
        module: RealDataModule,
        split: str,
        shuffle: bool,
        epoch: int,
    ) -> None:
        super().__init__()
        self.paths = module.input_paths
        self.max_events = module.max_events
        self.max_nodes = module.max_nodes
        self.overflow = module.max_nodes_overflow
        self.split = split
        self.split_config = module.split_config
        self.split_overrides = module.split_overrides
        self.source_split_overrides = module.source_split_overrides
        self.require_source_split_override = module.selection_manifest_hash is not None
        self.shuffle_buffer_size = module.shuffle_buffer_size if shuffle else 0
        self.seed = module.seed + epoch

    def __iter__(self):
        from hypertagging.data.streaming import ParquetEventIterableDataset

        records = ParquetEventIterableDataset(
            self.paths,
            max_events=self.max_events,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
            split_name=self.split,
            split_config=self.split_config,
            split_overrides=self.split_overrides,
            source_split_overrides=self.source_split_overrides,
            require_source_split_override=self.require_source_split_override,
        )
        for record in records:
            event = heterogeneous_event_from_record(record)
            if (
                self.max_nodes is not None
                and event.common_features.shape[0] > self.max_nodes
            ):
                if self.overflow == "drop":
                    continue
                raise OverflowError(
                    f"event {event.event_uid} exceeds max_nodes={self.max_nodes}"
                )
            yield event


def _assigned_split(
    record: dict,
    config: SourceAwareSplitConfig,
    *,
    event_overrides: dict[str, str],
    source_overrides: dict[str, str],
    require_source_override: bool = False,
) -> str:
    event_uid = str(record["event_uid"])
    if event_uid in event_overrides:
        return event_overrides[event_uid]
    source = str(record.get("source_file", ""))
    if source in source_overrides:
        return source_overrides[source]
    if require_source_override:
        raise ValueError(
            f"record source {source!r} is absent from the training-selection roles"
        )
    return stable_split_name(record, config)


__all__ = [
    "BALANCED_LEVEL_REPLAY_LEVELS",
    "BALANCED_LEVEL_REPLAY_UID_HASH_SCHEME",
    "BALANCED_LEVEL_REPLAY_VERSION",
    "BalancedLevelReplay",
    "FEATURE_BLOCKS",
    "RealDataModule",
    "build_balanced_level_replay",
    "build_real_data_module",
    "fit_training_normalizers",
    "preflight_dataset_index_data_binding",
    "resolve_data_paths",
]
