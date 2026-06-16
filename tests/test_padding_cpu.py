import awkward as ak
import numpy as np
import pytest

from hypertagging.utils.padding import pad_to, pack_with_evtNum, pack_with_evt_num


def test_pad_to_matches_legacy_default_padding():
    array = ak.Array([[1, 2], [3]])

    out = pad_to(array, dtype=np.int32)

    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, np.array([[1, 2], [3, 0]], dtype=np.int32))


def test_pad_to_supports_grafei_pad_length_and_fill_value():
    array = ak.Array([[1], [2, 3]])

    out = pad_to(array, fill_value=-1, pad_length=3, dtype=np.int64)

    expected = np.array([[1, -1, -1], [2, 3, -1]], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_pad_to_preserves_nested_feature_shape():
    array = ak.Array([[[1, 2], [3, 4]], [[5, 6]]])

    out = pad_to(array, fill_value=0, dtype=np.float32)

    assert out.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        out,
        np.array([[[1, 2], [3, 4]], [[5, 6], [0, 0]]], dtype=np.float32),
    )


def test_pad_to_rejects_records_like_legacy_helpers():
    array = ak.Array({"x": [[1], [2, 3]]})

    with pytest.raises(TypeError):
        pad_to(array)


def test_pack_with_evt_num_uses_legacy_counts():
    array = ak.Array({"evtNum": [1, 1, 2], "x": [10, 11, 20]})

    packed = pack_with_evt_num(array)

    assert len(packed) == 2
    np.testing.assert_array_equal(ak.num(packed).to_numpy(), np.array([2, 1]))
    np.testing.assert_array_equal(packed[0].x.to_numpy(), np.array([10, 11]))
    np.testing.assert_array_equal(pack_with_evtNum(array)[1].x.to_numpy(), np.array([20]))


def test_pad_to_preserves_gpt_v_kwargs_slice_behavior():
    array = ak.Array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]])

    out = pad_to(
        array,
        fill_value=0,
        dtype=np.float32,
        v_kwargs={
            "first": (0, -1, np.int32),
            "second": (1, -2, np.float32),
        },
    )

    assert set(out) == {"first", "second"}
    assert out["first"].dtype == np.int32
    assert out["second"].dtype == np.float32
    np.testing.assert_array_equal(
        out["first"],
        np.array([[[1, 2, 3]], [[7, 8, 9]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        out["second"],
        np.array([[[4, 5, 6]], [[-1, -2, -2]]], dtype=np.float32),
    )
