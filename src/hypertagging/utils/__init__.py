"""Utility helpers migrated from the historical HyperTagging repositories."""

from hypertagging.utils.checkpoint import (
    get_epoch,
    get_model_state_dict,
    load_checkpoint,
    load_model_state,
    save_checkpoint,
)
from hypertagging.utils.device import cpu_device, resolve_device
from hypertagging.utils.gpu_safety import (
    assert_full_training_requires_slurm,
    assert_local_gpu_tiny_test_allowed,
    get_nvidia_smi_snapshot,
    get_squeue_snapshot,
    is_inside_slurm,
)
from hypertagging.utils.io import ensure_directory
from hypertagging.utils.padding import pad_to, pack_with_evtNum, pack_with_evt_num
from hypertagging.utils.seeds import seed_everything

__all__ = [
    "assert_full_training_requires_slurm",
    "assert_local_gpu_tiny_test_allowed",
    "cpu_device",
    "ensure_directory",
    "get_epoch",
    "get_model_state_dict",
    "get_nvidia_smi_snapshot",
    "get_squeue_snapshot",
    "is_inside_slurm",
    "load_checkpoint",
    "load_model_state",
    "pack_with_evtNum",
    "pack_with_evt_num",
    "pad_to",
    "resolve_device",
    "save_checkpoint",
    "seed_everything",
]
