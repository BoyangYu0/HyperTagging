from __future__ import annotations

from hypertagging.basf2_integration.module import (
    _pid_likelihood_detector_available,
)


class _Likelihood:
    def __init__(self, available: set[str]) -> None:
        self.available = available

    def isAvailable(self, detector: str) -> bool:
        if detector == "broken":
            raise RuntimeError("unavailable detector API")
        return detector in self.available

    def getLogL(self, _hypothesis: object) -> float:
        # Empty PIDLikelihood objects can still expose finite default values;
        # detector availability, not finiteness alone, is the data contract.
        return 0.0


def test_pid_likelihood_requires_at_least_one_available_detector() -> None:
    empty = _Likelihood(set())
    assert not _pid_likelihood_detector_available(
        empty, ("svd", "cdc", "broken")
    )

    populated = _Likelihood({"cdc"})
    assert _pid_likelihood_detector_available(
        populated, ("svd", "broken", "cdc")
    )


def test_pid_likelihood_without_release_availability_api_is_masked() -> None:
    class _LegacyLikelihood:
        def getLogL(self, _hypothesis: object) -> float:
            return 0.0

    assert not _pid_likelihood_detector_available(
        _LegacyLikelihood(), ("cdc",)
    )
    assert not _pid_likelihood_detector_available(None, ("cdc",))
