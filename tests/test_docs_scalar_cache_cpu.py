"""Bounded scalar memoization must preserve privacy decisions and work charges."""
from collections import Counter
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/_ext"))
import wiki_privacy as privacy


@pytest.mark.parametrize("values", [
    ["public", 0, 0.25, None, True] * 20,
    ["%252Fhome%252FPRIVATE_USER%252Fsecret"] * 3,
    ["ghp_1234567890abcdefghijklmnop"] * 3,
    ["username=PRIVATE_USER"] * 3,
    ["needle-stays-denied"] * 3,
    ["complete canonical source body"] * 3,
    ["safe_" * 60] * 3,
])
def test_cached_and_uncached_scan_have_identical_results(tmp_path, monkeypatch, values):
    (tmp_path / "data.json").write_text(json.dumps({"values": values}))
    monkeypatch.setattr(privacy, "sensitive_literals", lambda root: {"needle-stays-denied"})
    monkeypatch.setattr(privacy, "_source_bodies", lambda root: (
        set(), ["completecanonicalsourcebody"],
    ))

    def scan():
        try:
            return privacy.validate_artifact(tmp_path, tmp_path)
        except ValueError as error:
            return type(error), str(error)

    cached = scan()
    # Only decorators constructed inside the next validation are disabled;
    # existing module-level decoder caches remain identical in both scans.
    monkeypatch.setattr(privacy, "lru_cache", lambda **kwargs: lambda function: function)
    assert scan() == cached


def test_cache_is_bounded_skips_large_values_and_expires_between_scans(tmp_path, monkeypatch):
    caches = []
    calls = Counter()
    original = privacy.violations

    def capture(**kwargs):
        def decorate(function):
            result = lru_cache(**kwargs)(function)
            caches.append(result)
            return result
        return decorate

    def counted(value, **kwargs):
        calls[value] += 1
        return original(value, **kwargs)

    monkeypatch.setattr(privacy, "lru_cache", capture)
    monkeypatch.setattr(privacy, "violations", counted)
    large = "safe_" * 60
    values = [f"public_value_{index}" for index in range(5000)]
    values += ["repeated_public_value"] * 20 + [large] * 2
    (tmp_path / "data.json").write_text(json.dumps(values))
    first = privacy.validate_artifact(tmp_path)
    assert first["privacy"] == "PASS"
    assert calls["repeated_public_value"] == 1
    assert calls[large] == 2
    assert caches[0].cache_info().maxsize == 4096
    assert caches[0].cache_info().currsize == 4096
    second = privacy.validate_artifact(tmp_path)
    assert second == first
    assert len(caches) == 2
    assert calls["repeated_public_value"] == 2
    assert calls[large] == 4


def test_cache_keys_include_all_rule_modes_and_results_are_immutable(tmp_path, monkeypatch):
    caches = []

    def capture(**kwargs):
        def decorate(function):
            result = lru_cache(**kwargs)(function)
            caches.append(result)
            return result
        return decorate

    monkeypatch.setattr(privacy, "lru_cache", capture)
    (tmp_path / "data.json").write_text('["public"]')
    privacy.validate_artifact(tmp_path)
    cached = caches[0]
    for value in ("public", "username=PRIVATE_USER", "<b>/private/path</b>"):
        for fields, markup, semantic in product((False, True), repeat=3):
            expected = tuple(privacy.violations(
                value, field_assignments=fields, html_markup=markup,
                semantic_projection=semantic,
            ))
            assert cached(value, fields, markup, semantic) == expected
            mutable = list(cached(value, fields, markup, semantic))
            mutable.append("occurrence-specific-failure")
            assert cached(value, fields, markup, semantic) == expected
