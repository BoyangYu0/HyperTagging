from hypertagging.training.reconstruction_trainer import _aggregate_metric_lists


def test_micro_metrics_sum_numerators_and_denominators():
    result = _aggregate_metric_lists({
        "accuracy": [1.0, 0.0],
        "accuracy_numerator": [1.0, 9.0],
        "accuracy_denominator": [1.0, 99.0],
    })
    assert result["macro_accuracy"] == 0.5
    assert result["micro_accuracy"] == 0.1

