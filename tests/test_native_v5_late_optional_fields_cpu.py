from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.preprocessing.schema_v4 import iter_event_records_v4
from hypertagging.preprocessing.schema_v5 import NativeNestedEventWriter, iter_native_nested_v5


def test_late_optional_node_field_is_not_lost(tmp_path):
    source = write_notebook_fixture_v4(tmp_path / "source.parquet")
    events = list(iter_event_records_v4(source))
    events[0]["nodes"][0]["truth_pid_token"] = None
    events[1]["nodes"][0]["truth_pid_token"] = 7
    output = tmp_path / "native.parquet"
    with NativeNestedEventWriter(output, event_buffer_size=1) as writer:
        for event in events:
            writer.write_event(event)
    rows = list(iter_native_nested_v5(output))
    assert rows[0]["nodes"][0]["truth_pid_token"] is None
    assert rows[1]["nodes"][0]["truth_pid_token"] == 7

