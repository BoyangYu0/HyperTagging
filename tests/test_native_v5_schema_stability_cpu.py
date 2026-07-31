import pytest

from hypertagging.preprocessing.schema_v5 import NativeNestedEventWriter, native_nested_schema_v5


def test_native_v5_schema_is_explicit_and_unknown_fields_are_rejected(tmp_path):
    assert native_nested_schema_v5().metadata[b"experimental_default_off"] == b"true"
    writer = NativeNestedEventWriter(tmp_path / "native.parquet")
    with pytest.raises(ValueError, match="unknown native-v5 event"):
        writer.write_event({"event_id": 1, "event_uid": "x", "unknown": 3})
    writer.abort()

