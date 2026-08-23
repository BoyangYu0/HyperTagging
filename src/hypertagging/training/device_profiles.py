"""Versioned H100 NVL/V100 phase-3 batch-efficiency profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from hypertagging.training.presentation_progress import validate_batch_profile


DEVICE_PROFILE_VERSION = "ht-pretraining-1m-phase3-device-profile-v1"


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    gres: str
    batch_ladder: tuple[int, ...]
    preferred_batch_size: int
    amp_dtype: str
    grad_scaler_enabled: bool
    bf16_policy: str
    memory_headroom_mib: int | None
    throughput_evidence: str

    def contract(self) -> dict[str, object]:
        return {
            "version": DEVICE_PROFILE_VERSION,
            "name": self.name,
            "exact_gres": self.gres,
            "batch_ladder": list(self.batch_ladder),
            "preferred_batch_size": self.preferred_batch_size,
            "amp_dtype": self.amp_dtype,
            "grad_scaler_enabled": self.grad_scaler_enabled,
            "bf16_policy": self.bf16_policy,
            "memory_headroom_mib": self.memory_headroom_mib,
            "throughput_evidence": self.throughput_evidence,
        }


DEVICE_PROFILES: Mapping[str, DeviceProfile] = {
    "h100nvl": DeviceProfile(
        name="h100nvl",
        gres="gpu:h100nvl:1",
        batch_ladder=(32, 64),
        preferred_batch_size=64,
        amp_dtype="bfloat16",
        grad_scaler_enabled=False,
        bf16_policy="require_cuda_bf16_supported",
        memory_headroom_mib=4_459,
        throughput_evidence="failed_h100_batch16_peak_4459_mib_peak_utilization_54_percent",
    ),
    "v100": DeviceProfile(
        name="v100",
        gres="gpu:v100:1",
        batch_ladder=(32, 64),
        preferred_batch_size=64,
        amp_dtype="float16",
        grad_scaler_enabled=True,
        bf16_policy="forbid_bf16_use_float16_gradscaler",
        memory_headroom_mib=None,
        throughput_evidence="existing_v100_diagnostics_are_not_production_batch_memory_evidence",
    ),
}


def get_device_profile(name: str) -> DeviceProfile:
    try:
        return DEVICE_PROFILES[str(name)]
    except KeyError as error:
        raise ValueError(f"unknown phase-3 device profile: {name!r}") from error


def validate_device_profile_batch(
    name: str,
    batch_size: int,
    *,
    total_presentations: int,
    milestone_presentations: tuple[int, ...],
) -> None:
    profile = get_device_profile(name)
    if batch_size not in profile.batch_ladder:
        raise ValueError(
            f"{name} batch size {batch_size} is outside the bounded ladder "
            f"{profile.batch_ladder}"
        )
    validate_batch_profile(
        batch_size,
        total_presentations=total_presentations,
        milestone_presentations=milestone_presentations,
    )


__all__ = [
    "DEVICE_PROFILE_VERSION",
    "DEVICE_PROFILES",
    "DeviceProfile",
    "get_device_profile",
    "validate_device_profile_batch",
]
