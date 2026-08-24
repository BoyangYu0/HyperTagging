"""Resolved training configuration with explicit precedence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import argparse
import sys


def resolve_config(
    *,
    defaults: Mapping[str, Any],
    yaml_path: str | Path | None = None,
    explicit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply code defaults < YAML < explicitly supplied CLI values."""

    output = dict(defaults)
    if yaml_path:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "YAML configuration requires PyYAML; install the training dependencies"
            ) from error
        loaded = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("training YAML must contain a top-level mapping")
        unknown = sorted(set(loaded) - set(output))
        if unknown:
            raise ValueError(f"unknown YAML configuration key(s): {unknown}")
        output.update(loaded)
    if explicit:
        unknown = sorted(set(explicit) - set(output))
        if unknown:
            raise ValueError(f"unknown explicit configuration key(s): {unknown}")
        output.update({key: value for key, value in explicit.items() if value is not None})
    return output


def resolve_argparse_namespace(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
) -> argparse.Namespace:
    """Resolve parser defaults < YAML ``--config`` < explicit CLI flags."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    initial = parser.parse_args(arguments)
    defaults = vars(parser.parse_args([]))
    explicit_destinations: set[str] = set()
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    for token in arguments:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit_destinations.add(option_to_dest[option])
    explicit = {
        name: value
        for name, value in vars(initial).items()
        if name in explicit_destinations and name != "config"
    }
    resolved = resolve_config(
        defaults=defaults,
        yaml_path=getattr(initial, "config", None),
        explicit=explicit,
    )
    return argparse.Namespace(**resolved)


__all__ = ["resolve_argparse_namespace", "resolve_config"]
