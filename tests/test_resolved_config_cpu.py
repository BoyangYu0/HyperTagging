from hypertagging.training.config import resolve_config


def test_yaml_is_applied_and_explicit_values_win(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("batch_size: 9\nmax_steps: 11\nnum_workers: 3\n", encoding="utf-8")
    resolved = resolve_config(
        defaults={"batch_size": 2, "max_steps": 2, "num_workers": 0},
        yaml_path=config,
        explicit={"max_steps": 4},
    )
    assert resolved == {"batch_size": 9, "max_steps": 4, "num_workers": 3}
