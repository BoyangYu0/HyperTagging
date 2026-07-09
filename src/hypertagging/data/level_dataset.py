"""Dataset wrappers for level-autoregressive reconstruction."""

from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data import Dataset

from hypertagging.data.level_batch import LevelEvent
from hypertagging.data.tiny_level_fixtures import tiny_level_events


class LevelReconstructionDataset(Dataset[LevelEvent]):
    """In-memory dataset for levelized events."""

    def __init__(self, events: Sequence[LevelEvent] | None = None, *, tiny: bool = False) -> None:
        if events is None:
            events = tiny_level_events() if tiny else ()
        self._events = tuple(events)

    def __len__(self) -> int:
        return len(self._events)

    def __getitem__(self, index: int) -> LevelEvent:
        return self._events[index]
