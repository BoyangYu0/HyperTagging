from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import MethodType

import pytest
import torch

from hypertagging.data.heterogeneous import (
    HeterogeneousEvent,
    heterogeneous_from_level_event,
)
from hypertagging.data.splitting import SourceAwareSplitConfig
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.training.data_module import (
    BALANCED_LEVEL_REPLAY_LEVELS,
    RealDataModule,
    build_balanced_level_replay,
)
from hypertagging.training.reconstruction_trainer import (
    ReconstructionConfig,
    _data_order_contract,
    _selected_event_levels,
)


def _event_for_level(
    level: int,
    uid: str,
    *,
    valid: bool = True,
    recursively_complete: bool = True,
    unary: bool = False,
    active: bool = True,
) -> HeterogeneousEvent:
    base = heterogeneous_from_level_event(tiny_level_events()[0])
    level_ids = base.level_ids + (level - 1)
    valid_targets = base.valid_reconstruction_target.clone()
    valid_targets[-1] = valid
    recursive = base.recursive_reconstructable_complete.clone()
    recursive[-1] = recursively_complete
    active_nodes = base.active.clone()
    active_nodes[-1] = active
    adjacency = base.daughter_adjacency.clone()
    if unary:
        adjacency[-1].zero_()
        adjacency[-1, 0] = True
    return replace(
        base,
        event_id=sum((index + 1) * byte for index, byte in enumerate(uid.encode())),
        event_uid=uid,
        level_ids=level_ids,
        daughter_adjacency=adjacency,
        active=active_nodes,
        valid_reconstruction_target=valid_targets,
        recursive_reconstructable_complete=recursive,
        truth_root_distance=level_ids.max() - level_ids,
        full_event_max_level=torch.full_like(level_ids, int(level_ids.max())),
    )


def _six_level_events(*, per_level: int = 1) -> list[HeterogeneousEvent]:
    return [
        _event_for_level(level, f"level-{level}-event-{index}")
        for level in reversed(BALANCED_LEVEL_REPLAY_LEVELS)
        for index in reversed(range(per_level))
    ]


def test_exact_phase35_slot_budget_is_globally_level_balanced() -> None:
    replay = build_balanced_level_replay(
        iter(_six_level_events(per_level=2)),
        target_policy="complete_only",
        seed=20260904,
    )

    assert replay.pool_counts == {level: 2 for level in range(1, 7)}
    assert all(
        replay.pools_by_level[level] == tuple(sorted(replay.pools_by_level[level]))
        for level in replay.levels
    )
    counts = replay.level_counts(start_slot=0, slot_count=140_032)
    assert counts == {
        1: 23_339,
        2: 23_339,
        3: 23_339,
        4: 23_339,
        5: 23_338,
        6: 23_338,
    }

    contract = replay.contract(planned_slot_count=140_032)
    assert contract["planned_schedule"] == {
        "start_slot": 0,
        "slot_count": 140_032,
        "end_slot_exclusive": 140_032,
        "level_counts": {str(level): counts[level] for level in replay.levels},
        "max_minus_min_level_count": 1,
    }
    assert set(contract["eligible_pool_uid_sha256_by_level"]) == {
        str(level) for level in replay.levels
    }
    assert all(
        len(value) == 64
        for value in contract["eligible_pool_uid_sha256_by_level"].values()
    )


def test_seeded_pool_permutations_replace_only_after_each_exhaustion() -> None:
    events = [_event_for_level(1, f"uid-{index:02d}") for index in range(8)]
    replay_a = build_balanced_level_replay(
        iter(reversed(events)),
        target_policy="complete_only",
        seed=17,
        levels=(1,),
    )
    replay_b = build_balanced_level_replay(
        iter(events),
        target_policy="complete_only",
        seed=17,
        levels=(1,),
    )
    replay_other_seed = build_balanced_level_replay(
        iter(events),
        target_policy="complete_only",
        seed=18,
        levels=(1,),
    )

    selected_a = [
        uid for _level, uid in replay_a.selections(start_slot=0, slot_count=24)
    ]
    selected_b = [
        uid for _level, uid in replay_b.selections(start_slot=0, slot_count=24)
    ]
    selected_other = [
        uid for _level, uid in replay_other_seed.selections(start_slot=0, slot_count=24)
    ]
    assert selected_a == selected_b
    assert selected_a != selected_other
    expected_pool = set(replay_a.pools_by_level[1])
    assert all(
        set(selected_a[start : start + 8]) == expected_pool for start in (0, 8, 16)
    )


def test_resume_uses_absolute_start_step_times_batch_size() -> None:
    replay = build_balanced_level_replay(
        iter(_six_level_events(per_level=3)),
        target_policy="complete_only",
        seed=29,
    )
    batch_size = 7
    start_step = 3
    full = replay.selections(start_slot=0, slot_count=70)
    resumed = replay.selections(
        start_slot=start_step * batch_size,
        slot_count=28,
    )
    assert resumed == full[start_step * batch_size : start_step * batch_size + 28]

    batch = replay.collated_batch(
        start_slot=start_step * batch_size,
        batch_size=batch_size,
    )
    assert batch["balanced_replay_global_slots"].tolist() == list(range(21, 28))
    assert batch["selected_target_levels"].tolist() == [4, 5, 6, 1, 2, 3, 4]


def test_pools_exactly_apply_policy_activity_and_minimum_daughters() -> None:
    accepted = _event_for_level(3, "z-accepted")
    invalid = _event_for_level(3, "invalid", valid=False)
    incomplete = _event_for_level(3, "incomplete", recursively_complete=False)
    unary = _event_for_level(3, "unary", unary=True)
    inactive = _event_for_level(3, "inactive", active=False)
    events = [unary, accepted, inactive, incomplete, invalid]

    complete = build_balanced_level_replay(
        iter(events),
        target_policy="complete_only",
        seed=1,
        levels=(3,),
    )
    partial = build_balanced_level_replay(
        iter(events),
        target_policy="reconstructable_partial",
        seed=1,
        levels=(3,),
    )
    diagnostic = build_balanced_level_replay(
        iter(events),
        target_policy="diagnostic_all",
        seed=1,
        levels=(3,),
    )

    assert complete.pools_by_level[3] == ("z-accepted",)
    assert partial.pools_by_level[3] == ("incomplete", "z-accepted")
    assert diagnostic.pools_by_level[3] == (
        "incomplete",
        "invalid",
        "z-accepted",
    )
    assert "unary" not in diagnostic.pools_by_level[3]
    assert "inactive" not in diagnostic.pools_by_level[3]


def test_replay_batch_attaches_one_enforced_eligible_level_per_event() -> None:
    replay = build_balanced_level_replay(
        iter(_six_level_events()),
        target_policy="complete_only",
        seed=3,
    )
    batch = replay.collated_batch(start_slot=0, batch_size=6)
    selected = _selected_event_levels(
        batch,
        valid_levels=list(BALANCED_LEVEL_REPLAY_LEVELS),
        mode="balanced_level_replay",
        seed=3,
        step=0,
        target_policy="complete_only",
    )

    assert sum(int(mask.sum()) for mask in selected.values()) == 6
    assert Counter(batch["selected_target_levels"].tolist()) == Counter(range(1, 7))
    for event_index, level in enumerate(batch["selected_target_levels"].tolist()):
        assert bool(selected[level][event_index])
        assert (
            sum(bool(selected[candidate][event_index]) for candidate in selected) == 1
        )

    corrupted = dict(batch)
    corrupted["selected_target_levels"] = batch["selected_target_levels"].clone()
    corrupted["selected_target_levels"][0] = 6
    with pytest.raises(ValueError, match="not eligible"):
        _selected_event_levels(
            corrupted,
            valid_levels=list(BALANCED_LEVEL_REPLAY_LEVELS),
            mode="balanced_level_replay",
            seed=3,
            step=0,
            target_policy="complete_only",
        )


def test_real_data_module_materializes_the_train_split_only_once() -> None:
    events = [_event_for_level(1, "uid-b"), _event_for_level(1, "uid-a")]
    module = RealDataModule(
        input_paths=(),
        normalizers={},
        split_manifest={},
        split_manifest_hash="split-hash",
        overflow_counters={},
        seed=7,
        split_config=SourceAwareSplitConfig(seed=7),
    )
    calls = 0

    def iter_events(
        _self: RealDataModule,
        split: str,
        *,
        shuffle: bool = False,
        epoch: int = 0,
    ):
        nonlocal calls
        assert split == "train"
        assert not shuffle
        assert epoch == 0
        calls += 1
        yield from events

    module.iter_events = MethodType(iter_events, module)
    first = module.balanced_level_replay(
        target_policy="complete_only",
        seed=7,
        levels=(1,),
        planned_slot_count=5,
    )
    second = module.balanced_level_replay(
        target_policy="complete_only",
        seed=7,
        levels=(1,),
        planned_slot_count=9,
    )

    assert first is second
    assert calls == 1
    assert first.pools_by_level[1] == ("uid-a", "uid-b")
    assert module.balanced_level_replay_contract["planned_schedule"]["slot_count"] == 9
    checkpoint_contract = _data_order_contract(
        ReconstructionConfig(
            data="unused",
            output_dir="unused",
            level_sampling_mode="balanced_level_replay",
        ),
        module,
    )
    expected_bound_contract = dict(module.balanced_level_replay_contract)
    expected_bound_contract.pop("planned_schedule")
    assert checkpoint_contract["balanced_level_replay"] == expected_bound_contract
