"""Deferred SIGUSR1 handling for atomic, optimizer-boundary checkpoints."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from types import FrameType


SLURM_REQUEUE_EXIT_CODE = 75


class PendingValidationInterrupted(RuntimeError):
    """Interrupt a read-only validation after its restart state is durable."""


class SafeBoundarySignalController:
    """Record SIGUSR1 asynchronously and let a trainer act at a safe boundary."""

    def __init__(self) -> None:
        self.requested = False
        self._previous: signal.Handlers | None = None
        self._validation_is_restartable = False

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("SIGUSR1 checkpoint handling requires the main thread")
        self._previous = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._handle)

    @property
    def installed(self) -> bool:
        return self._previous is not None

    def restore(self) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGUSR1, self._previous)
            self._previous = None

    def _handle(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True
        if self._validation_is_restartable:
            raise PendingValidationInterrupted(
                "SIGUSR1 interrupted a serialized pending validation"
            )

    @contextmanager
    def restartable_validation(self):
        """Allow SIGUSR1 to abort validation only after pending state is saved."""

        self._validation_is_restartable = True
        try:
            if self.requested:
                raise PendingValidationInterrupted(
                    "SIGUSR1 arrived before pending validation began"
                )
            yield
        finally:
            self._validation_is_restartable = False

    def exit_after_checkpoint(self) -> None:
        self.restore()
        raise SystemExit(SLURM_REQUEUE_EXIT_CODE)


def install_safe_boundary_signal_controller() -> SafeBoundarySignalController:
    controller = SafeBoundarySignalController()
    controller.install()
    return controller


__all__ = [
    "SLURM_REQUEUE_EXIT_CODE",
    "PendingValidationInterrupted",
    "SafeBoundarySignalController",
    "install_safe_boundary_signal_controller",
]
